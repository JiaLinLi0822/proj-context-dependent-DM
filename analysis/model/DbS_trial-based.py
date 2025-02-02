import numpy as np
from scipy.stats import norm
from scipy import optimize
import pandas as pd
from constrNMPy import constrNMPy
import os
import multiprocessing as mp
import time

def kernel_smoothing(x, samples, bandwidth):
    '''
    using Gaussian kernel to estimate the subjective rank for given sample
    x: attribute value
    samples: samples retrieved from memory
    bandwidth: the bandwidth of the kernel, controlling the smoothness of the estimate
    '''
    # rank_values = np.array([norm.cdf((x - sample) / bandwidth) for sample in samples])
    rank_values = norm.cdf((x - samples) / bandwidth)
    rank_values = np.mean(rank_values)
    
    return rank_values

def rank(x, samples, bandwidth, min_value = 1, max_value = 5):
    '''
    rescale the rank value to the value range of [1, 5]
    '''
    min_rank = kernel_smoothing(min_value, samples, bandwidth)
    max_rank = kernel_smoothing(max_value, samples, bandwidth)
    rank = kernel_smoothing(x, samples, bandwidth)
    
    rank = (rank - min_rank) / (max_rank - min_rank)
    rank_value = min_value + rank * (max_value - min_value)
    
    return rank_value

def memory_slots(memory, current_trial, new_samples, N):
    '''
    memory_slots 更新 memory，存储 (trial_num, reward) 的元组，并仅保留最近 N 个 trial 的样本。
    
    参数:
    - memory: 当前的 memory，NumPy 数组，形状为 (n, 2)，其中 memory[:,0] 是 itrial，memory[:,1] 是 reward
    - current_trial: 当前的 trial 编号
    - new_samples: 新的 reward 样本，可以是单个值或可迭代对象
    - N: 保留最近 N 个 trial 的样本
    
    返回:
    - updated_memory: 更新后的 memory，NumPy 数组
    '''
    
    # 确保 new_samples 是 NumPy 数组
    new_samples = np.atleast_1d(new_samples)
    
    # 创建一个包含 current_trial 的数组，长度与 new_samples 相同
    new_trials = np.full(new_samples.shape, current_trial)
    
    # 将新的 (itrial, reward) 组合成一个二维数组
    new_entries = np.column_stack((new_trials, new_samples))
    
    if memory.size == 0:
        # 如果 memory 为空，直接赋值为 new_entries
        updated_memory = new_entries
    else:
        # 否则，将 new_entries 添加到 memory 的末尾
        updated_memory = np.vstack((memory, new_entries))
    
    # 计算保留的最低 itrial 值
    cutoff = current_trial - N
    
    # 使用布尔掩码仅保留 itrial > cutoff 的样本
    mask = updated_memory[:, 0] > cutoff
    updated_memory = updated_memory[mask]
    
    return updated_memory
    

def optimizefun(pars, actions, muvalue, totaldays, trial_num, rewards):

    '''
    optimizefun_thresholdVrisk specifies the Prop-V-risk model

    alpha is the power of the value function
    sigma is the standard deviation of the exploration bonus
    beta is the inverse temperature
    k and b are the parameters of the threshold
    '''
    
    Ntrials = len(np.unique(trial_num))
    
    loglik = np.zeros(Ntrials)
    i = 0
    
    k, b, sigma, beta, h, alpha = pars
    # N = int(np.round(N))
    deta = sigma / 300 # that is the resolution of the numerical integration over eta eta的微分
    eta = np.arange(-6 * sigma, 6 * sigma + deta, deta).reshape(1, -1) # that is a row vector [1, 3601]
    pmf = norm.pdf(eta, loc=0, scale=sigma) * deta
    
    memory = np.empty((0, 2))
    
    for itrial in range(Ntrials):

        # pull out the data for that trial
        trial_length = int(totaldays[i])
        trial_rstar = (np.concatenate(([np.nan], muvalue[i:i + trial_length - 1]))).reshape(-1, 1) # [trial_length, 1]
        trial_action = (actions[i:i + trial_length]).reshape(-1, 1) # [trial_length, 1]
        trial_reward = (rewards[i:i + trial_length]).reshape(-1, 1)
        
        # update the memory
        for j in range(trial_length):
            
            if len(memory) > 0 and not np.isnan(trial_rstar[j]):
                trial_rstar[j] = rank(trial_rstar[j], memory[:, 1], h) # scale the rank back to the value range of [1, 5]
            
        trial_rstar = trial_rstar * alpha # [trial_length, 1]
        
        # the shape of thresholdfix is (trial_length, 1), that is a column vector
        thresholdfix = (k * np.flip(np.arange(1, trial_length + 1)) / trial_length + b).reshape(-1, 1) # Note that the threshold is decreasing

        threshold = thresholdfix + eta
        threshold = np.clip(threshold, 1, 5)

        pexplore_eta = 1 / (1 + np.exp(-beta * (threshold - trial_rstar))) # [trial_length, 3601]

        pexplore_eta[0] = 1  # first action is surely exploration

        # loglik[itrial] = np.log(np.sum(np.prod((pexplore_eta * np.where(np.tile(trial_action, (1, len(eta))) == 1, 1, 0) + 
        #                        (1 - pexplore_eta) * np.where((np.tile(trial_action, (1, len(eta)))) == 0, 1, 0)),axis=0) * pmf))
        
        probs = (pexplore_eta ** trial_action) * ((1 - pexplore_eta) ** (1 - trial_action))
        # product over all days in this trial along axis=0
        probs_product = np.prod(probs, axis=0)  # shape: [#eta]

        # Integration over eta
        likelihood = np.sum(probs_product * pmf)
        loglik[itrial] = np.log(likelihood)

        # update memory
        trial_reward_explore_index = np.where(trial_action == 1)
        trial_reward_explore = trial_reward[trial_reward_explore_index]
        memory = memory_slots(memory, itrial, trial_reward, 1)
        
        i += trial_length
    
    minusllh = -np.sum(loglik)
    
    return minusllh

def fit_subject(file, folder, N_iter, output_dir):
    data = pd.read_csv(folder + file)
    actions = data['action'].values
    muvalue = data['muvalue'].values
    trial_num = data['trial_num'].values
    totaldays = data['totaldays'].values
    rewards = data['reward'].values

    parOpt = [] 
    fopt = []
    for _ in range(N_iter):
        pars0 = np.random.uniform(1e-4, 1, 6)
        pars0[1] = np.random.uniform(1e0, 5e0)
        # pars0[5] = np.random.uniform(1e0, 5e0)

        lb = np.array([1e-5, 1e0, 1e-5, 1e-5, 1e-5, 1e-5])
        ub = np.array([5e0, 5e0, 2e0, 1e2, 1e2, 2e0])

        pars = constrNMPy.constrNM(optimizefun, pars0, lb, ub, args=(actions, muvalue, totaldays, trial_num, rewards), 
                                    full_output=True, xtol=1e-5, ftol=1e-5, maxiter=1000, maxfun=5000, disp=False)
        
        print(pars)
        parOpt.append(pars['xopt'])
        fopt.append([pars['fopt'], pars['iter'], pars['funcalls'], pars['warnflag']])
    
    min_index = fopt.index(min(fopt))
    res = np.concatenate([[file], parOpt[min_index], fopt[min_index]])

    fitting = pd.DataFrame(parOpt, columns=['k', 'b', 'sigma', 'beta', 'h', 'alpha'])
    fitting = pd.merge(fitting, pd.DataFrame(fopt), left_index=True, right_index=True)
    fitting.columns = ['k', 'b', 'sigma', 'beta', 'h', 'alpha', 'loglik', 'nIter', 'FunEvals', 'warnflag']
    # fitting['N'] = fitting['N'].apply(lambda x: int(np.round(x)))

    fitting.to_csv(output_dir + file, index=False)
    
    return res

def modelfitting(N_iter):
    path = "/Users/lijialin/Desktop/课程/毕业设计/data/fit/"
    folder_path = [path + 'Cond3/']
    
    for folder in folder_path:
        all_files = [f for f in os.listdir(folder) if not f.startswith('._')]
        
        output_path = "/Users/lijialin/Desktop/课程/models/Dbs_trial-based/"
        output_dir = output_path + folder.split('/')[-2] + '/'
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Use multiprocessing to parallelize the fitting
        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = pool.starmap(fit_subject, [(file, folder, N_iter, output_dir) for file in all_files])
        
        # Save the final results
        res = pd.DataFrame(results, columns=['file', 'k', 'b', 'sigma', 'beta', 'h', 'alpha', 'loglik', 'nIter', 'FunEvals', 'warnflag'])
        
        # N must be an integer, use round and int
        # res['N'] = res['N'].apply(lambda x: int(np.round(x)))
        res.to_csv(output_dir + 'opt.csv', index=False)

    return

if __name__ == '__main__':
    modelfitting(N_iter=1)
