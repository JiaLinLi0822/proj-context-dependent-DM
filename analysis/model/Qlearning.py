import numpy as np
from scipy.stats import norm
from scipy import optimize
import pandas as pd
from constrNMPy import constrNMPy
import os
import multiprocessing as mp

def optimizefun(pars, actions, muvalue, totaldays, trial_num, reward):

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
    
    k, b, sigma, beta, alpha, lr = pars[0], pars[1], pars[2], pars[3], pars[4], pars[5]
    deta = sigma / 300 # that is the resolution of the numerical integration over eta eta的微分
    eta = np.arange(-6 * sigma, 6 * sigma + deta, deta).reshape(1, -1) # that is a row vector [1, 3601]
    pmf = norm.pdf(eta, loc=0, scale=sigma) * deta
    
    Q = b
    
    for itrial in range(Ntrials):

        # pull out the data for that trial
        trial_length = int(totaldays[i])
        trial_rstar = (np.concatenate(([np.nan], muvalue[i:i + trial_length - 1])) * alpha).reshape(-1, 1) # [trial_length, 1]
        trial_action = (actions[i:i + trial_length]).reshape(-1, 1) # [trial_length, 1]
        trial_reward = (reward[i:i + trial_length]).reshape(-1, 1) # [reward, 1]
        
        barx_t = np.mean(trial_reward)
        
        # the shape of thresholdfix is (trial_length, 1), that is a column vector
        thresholdfix = (k * np.flip(np.arange(1, trial_length + 1)) / trial_length + Q).reshape(-1, 1) # Note that the threshold is decreasing

        threshold = thresholdfix + eta
        threshold = np.clip(threshold, 1, 5)

        # numerically integrate over eta
        pexplore_eta = 1 / (1 + np.exp(-beta * (threshold - trial_rstar))) # [trial_length, 3601]

        pexplore_eta[0] = 1  # first action is surely exploration

        loglik[itrial] = np.log(np.sum(np.prod((pexplore_eta * np.where(np.tile(trial_action, (1, len(eta))) == 1, 1, 0) + 
                               (1 - pexplore_eta) * np.where((np.tile(trial_action, (1, len(eta)))) == 0, 1, 0)),axis=0) * pmf))
        
        Q = Q + lr*(barx_t-Q)
        i += trial_length
    
    minusllh = -np.sum(loglik)
    
    return minusllh

def fit_subject(file, folder, N_iter, output_dir):
    data = pd.read_csv(folder + file)
    actions = data['action'].values
    muvalue = data['muvalue'].values
    trial_num = data['trial_num'].values
    totaldays = data['totaldays'].values
    reward = data['reward'].values

    parOpt = [] 
    fopt = []
    for _ in range(N_iter):
        pars0 = np.random.uniform(1e-4, 1, 6)
        pars0[1] = np.random.uniform(1e0, 5e0)

        lb = np.array([1e-5, 1e0, 1e-5, 1e-5, 1e-5, 1e-5])
        ub = np.array([5e0, 5e0, 2e0, 1e2, 2e0, 1e0])

        pars = constrNMPy.constrNM(optimizefun, pars0, lb, ub, args=(actions, muvalue, totaldays, trial_num, reward), 
                                    full_output=True, xtol=1e-5, ftol=1e-5, maxiter=1000, maxfun=5000, disp=False)
        
        print(pars)
        parOpt.append(pars['xopt'])
        fopt.append([pars['fopt'], pars['iter'], pars['funcalls'], pars['warnflag']])
    
    min_index = fopt.index(min(fopt))
    res = np.concatenate([[file], parOpt[min_index], fopt[min_index]])

    fitting = pd.DataFrame(parOpt, columns=['k', 'b', 'sigma', 'beta', 'alpha', 'lr'])
    fitting = pd.merge(fitting, pd.DataFrame(fopt), left_index=True, right_index=True)
    fitting.columns = ['k', 'b', 'sigma', 'beta', 'alpha', 'lr', 'loglik', 'nIter', 'FunEvals', 'warnflag']

    fitting.to_csv(output_dir + file, index=False)
    
    return res

def modelfitting(model, N_iter):
    path = "C:/Users/ljl22/Desktop/毕业设计/data/fit/"
    folder_path = [path + 'Cond3/']
    
    for folder in folder_path:
        all_files = [f for f in os.listdir(folder) if not f.startswith('._')]
        
        if model == 1:
            output_path = "C:/Users/ljl22/Desktop/毕业设计/models/new/Qlearning/"
            output_dir = output_path + folder.split('/')[-2] + '/'
            
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Use multiprocessing to parallelize the fitting
            with mp.Pool(processes=mp.cpu_count()) as pool:
                results = pool.starmap(fit_subject, [(file, folder, N_iter, output_dir) for file in all_files])
            
            # Save the final results
            res = pd.DataFrame(results, columns=['file', 'k', 'b', 'sigma', 'beta', 'alpha', 'lr', 'loglik', 'nIter', 'FunEvals', 'warnflag'])
            res.to_csv(output_dir + 'opt.csv', index=False)

    return

if __name__ == '__main__':
    modelfitting(model=1, N_iter=5)
