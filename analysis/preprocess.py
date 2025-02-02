
## preprocessing the data
## covert to dataframe

import pandas as pd
import numpy as np
import os

path = '../prj-context-dependent-DM/data/raw/'
folder_path = [path + 'Cond1/', path +'Cond2/', path +'Cond3/', path +'Cond4/']

def json2csv(input_dir, output_dir, exoutput_dir, group):

    '''
    Convert JSON files to CSV files and drop specified columns.
    cond_index: int, the index of the condition(e.g., 1,2,3,4)
    input_dir: str, the directory containing the input JSON files
    output_dir: str, the directory to save the output CSV files
    '''

    cols = ['trial_num', 'trial_length', 'click_num', 'RT', 'action', 'reward']
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(exoutput_dir):
        os.makedirs(exoutput_dir)
    if not os.path.exists(fitoutput_dir):
        os.makedirs(fitoutput_dir)
    
    # 列出指定路径下的所有文件
    files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    
    # 遍历每个文件
    for i, file in enumerate(files):
        # 生成输入文件路径和输出文件路径
        input_file_path = os.path.join(input_dir, file)
        # output_file_path = os.path.join(output_dir, os.path.splitext(file)[0] + '.csv')
        output_file_path = os.path.join(output_dir, 'sub{}'.format(str(i+1)) + '.csv')
        exoutput_file_path = os.path.join(exoutput_dir, 'sub{}'.format(str(i+1).zfill(2)) + '.csv')
        
        # 使用pandas读取JSON文件
        data = pd.read_json(input_file_path)
        
        # 保留指定的列
        data = data[cols]
        data = generatemoreinfo(data, group) # add average reward and highest reward to the data
        extractdata = dataextract(data) # extract the data for the model
        fitdata = dataForfitting(data)
        
        # 将数据保存为CSV文件
        data.to_csv(output_file_path, index=False)
        extractdata.to_csv(exoutput_file_path, index=False)
        fitdata.to_csv(fitoutput_dir + '/sub{}'.format(str(i+1).zfill(2)) + '.csv', index=False)


def generatemoreinfo(data, group):
    '''
    (1) Calculates the highest reward so far and
    (2) Average reward for each trial (after carrying out the action and getting rewarded), and 
    (3) Adds them to the raw data matrix
    '''
     # original data:
    # 1 trial index, 2 total number of days, 3 day, 4 RT, 5 action, 6 reward
    # more info:
    # 7 highest reward so far
    # 8 average reward so far

    # 生成一个新的列，用于存储最高奖励
    data['highest_reward'] = 0
    # 生成一个新的列，用于存储平均奖励
    data['average_reward'] = 0
    # 生成一个新的列，用于存储stochasticity
    data['stochasticity'] = 0

    data['average_reward_exploration'] = 0
    data['stochasticity_exploration'] = 0

    data['last_trial_average_reward'] = 0
    data['impulse'] = 0

    i = 0
    while i < data.shape[0]:
        if data.loc[i, 'click_num'] == 1:
            data.loc[i, 'highest_reward'] = data.loc[i, 'reward'] # initialize highest reward so far
            data.loc[i, 'average_reward'] = data.loc[i, 'reward'] # initialize average reward so far
            data.loc[i, 'stochasticity'] = 0
            data.loc[i, 'average_reward_exploration'] = data.loc[i, 'reward']
            data.loc[i, 'stochasticity_exploration'] = 0
            max_reward = data.loc[i, 'reward']

            if data.loc[i, 'trial_num'] != 1:
                data.loc[i, 'last_trial_average_reward'] = data.loc[i - 1, 'average_reward_exploration']

            # if data.loc[i, 'trial_num'] != 1:
            #     if group == 1 or group == 3:
            #         if data.loc[i-1, 'average_reward'] <= 3:
            #             data.loc[i, 'impulse'] = 1
            #     else:
            #         if data.loc[i-1, 'average_reward'] <= -3:
            #             data.loc[i, 'impulse'] = 1
            # gap = [0, 0]
            # abs_gap = [0, 0]
            gap = [np.nan]
            abs_gap = [np.nan]

            j = i + 1

            while j < i + data.loc[i, 'trial_length']:
                gap.append(round(data.loc[j, 'reward'] - max_reward, 1))  # gain
                # gap.append(data.loc[j, 'reward'] - data.loc[j, 'average_reward'])
                abs_gap.append(abs(round(data.loc[j, 'reward'] - max_reward,1)))

                if data.loc[j, 'trial_num'] != 1:
                    data.loc[j, 'last_trial_average_reward'] = data.loc[j - 1, 'last_trial_average_reward']

                if group == 1 or group == 3:
                    if data.loc[j-1, 'average_reward'] <= 3:
                        data.loc[j, 'impulse'] = 1
                else:
                    if data.loc[j-1, 'average_reward'] <= -3:
                        data.loc[j, 'impulse'] = 1

                max_reward = max(max_reward, data.loc[j, 'reward'])
                data.loc[j, 'highest_reward'] = max_reward  # highest reward so far
                data.loc[j, 'average_reward'] = (data.loc[j-1, 'average_reward'] * (data.loc[j, 'click_num'] - 1) + data.loc[j, 'reward']) / data.loc[j, 'click_num']  # average reward
                
                # calculate stochasticity, where calculate the standard deviation of the reward
                reward_diff = data.loc[:j, 'reward'] - data.loc[j, 'average_reward']
                data.loc[j, 'stochasticity'] = np.sqrt(np.mean(reward_diff ** 2))

                # calculate average reward and stochasticity for exploration
                if data.loc[j, 'action'] == 1:
                    data.loc[j, 'average_reward_exploration'] = (data.loc[j-1, 'average_reward_exploration'] * (data.loc[j, 'click_num'] - 1) + data.loc[j, 'reward']) / data.loc[j, 'click_num']
                    exploration_reward_diff = data.loc[:j, 'reward'] - data.loc[j, 'average_reward_exploration']
                    data.loc[j, 'stochasticity_exploration'] = np.sqrt(np.mean(exploration_reward_diff ** 2))
                else:
                    data.loc[j, 'average_reward_exploration'] = data.loc[j-1, 'average_reward_exploration']
                    data.loc[j, 'stochasticity_exploration'] = data.loc[j-1, 'stochasticity_exploration']

                j += 1
            
            data.loc[i:j-1, 'gap'] = gap[:]
            data.loc[i:j-1, 'abs_gap'] = abs_gap[:]
            i = j
        else:
            i += 1
    return data

def dataextract(data):

    mu_origin = np.concatenate(([np.nan], data.iloc[:-1]['highest_reward'].values))
    avereward_origin = np.concatenate(([np.nan], data.iloc[:-1]['average_reward'].values))
    avereward_exploration_origin = np.concatenate(([np.nan], data.iloc[:-1]['average_reward_exploration'].values))
    stochasticity_origin = np.concatenate(([np.nan], data.iloc[:-1]['stochasticity'].values))
    stochasticity_exploration_origin = np.concatenate(([np.nan], data.iloc[:-1]['stochasticity_exploration'].values))
    # gap = data.iloc[:]['gap'].values
    # abs_gap = data.iloc[:]['abs_gap'].values
    gap_origin = np.concatenate(([np.nan], data.iloc[:-1]['gap'].values))
    abs_gap_origin = np.concatenate(([np.nan], data.iloc[:-1]['abs_gap'].values))
    reward = np.concatenate(([np.nan], data.iloc[:-1]['reward'].values))
    impulse_origin = np.concatenate(([np.nan], data.iloc[:-1]['impulse'].values))

    ind = data.loc[:, 'click_num'] > 1
    datause = data.loc[ind, :]
    actions = datause.loc[:, 'action'].values
    totaldays = datause.loc[:, 'trial_length'].values
    daysleft = (datause.loc[:, 'trial_length'] - datause.loc[:, 'click_num'] + 1).values
    avereward = avereward_origin[ind]
    avereward_exploration = avereward_exploration_origin[ind]
    stochasticity = stochasticity_origin[ind]
    stochasticity_exploration = stochasticity_exploration_origin[ind]
    # gap = gap[ind]
    # abs_gap = abs_gap[ind]
    gap = gap_origin[ind]
    abs_gap = abs_gap_origin[ind]
    muvalue = mu_origin[ind]
    reward = reward[ind]
    RT = datause.loc[:, 'RT'].values
    trial_num = datause.loc[:, 'trial_num'].values
    last_trial_average_reward = datause.loc[:, 'last_trial_average_reward'].values
    impulse = impulse_origin[ind]

    extractdata = np.column_stack((trial_num, actions, totaldays, daysleft, reward, muvalue, avereward, avereward_exploration, RT, stochasticity, stochasticity_exploration, gap, abs_gap, last_trial_average_reward, impulse))
    extractdata = pd.DataFrame(extractdata, columns=['trial_num', 'action', 'totaldays', 'daysleft', 'last_reward', 'muvalue', 'avereward', 'avereward_exploration', 'RT', 'stochasticity', 'stochasticity_exploration', 'gap', 'abs_gap', 'last_trial_average_reward', 'impulse'])
    return extractdata

def dataForfitting(data):

    ind = data.loc[:, 'click_num']
    datause = data.loc[ind, :]
    actions = data.loc[:, 'action'].values
    totaldays = data.loc[:, 'trial_length'].values
    daysleft = (data.loc[:, 'trial_length'] - data.loc[:, 'click_num'] + 1).values
    muvalue = data.loc[:, 'highest_reward'].values
    reward = data.loc[:, 'reward']
    RT = datause.loc[:, 'RT'].values
    trial_num = data.loc[:, 'trial_num'].values
    gap = data.loc[:, 'gap'].values

    fitdata = np.column_stack((trial_num, actions, totaldays, daysleft, muvalue, reward, gap, RT))
    fitdata = pd.DataFrame(fitdata, columns=['trial_num', 'action', 'totaldays', 'daysleft', 'muvalue', 'reward', 'gap', 'RT'])
    # fitdata[['reward', 'muvalue', 'gap']] = fitdata.groupby('trial_num')[['reward', 'muvalue', 'gap']].shift(1)

    return fitdata

def groupdata():

    import shutil
    # 原始数据文件夹路径
    source_folder = '/Volumes/T7 Shield/实验数据'

    # 目标文件夹路径，存储不同类别的文件
    target_folders = {
        'Cond1': '../prj-context-dependent-DM/data/raw/Cond1',
        'Cond2': '../prj-context-dependent-DM/data/raw/Cond2',
        'Cond3': '../prj-context-dependent-DM/data/raw/Cond3',
        'Cond4': '../prj-context-dependent-DM/data/raw/Cond4'
    }

    # 创建目标文件夹
    for folder_path in target_folders.values():
        os.makedirs(folder_path, exist_ok=True)

    # 遍历原始数据文件夹中的文件名
    for filename in os.listdir(source_folder):
        # 判断文件名是否以 ".json" 结尾，并且不包含 "strategy" 字符串
        if filename.endswith('.json') and 'strategy' not in filename:
            # 分割文件名
            parts = filename.split('_')
            # 提取条件信息
            condition = parts[1]
            # 确定目标文件夹路径
            target_folder = target_folders.get(condition)
            if target_folder:
                # 构建源文件路径和目标文件路径
                source_file_path = os.path.join(source_folder, filename)
                target_file_path = os.path.join(target_folder, filename)
                # 复制文件到目标文件夹
                shutil.copyfile(source_file_path, target_file_path)
                print(f'Copied {filename} to {target_folder}')
            else:
                print(f'Condition {condition} not found in target_folders')

def groupstrategy():

    import shutil
    # 原始数据文件夹路径
    source_folder = '../prj-context-dependent-DM/data/raw/'

    # 目标文件夹路径，存储不同类别的文件
    target_folders = {
        'Cond1': '../prj-context-dependent-DM/data/strategy/Cond1',
        'Cond2': '../prj-context-dependent-DM/data/strategy/Cond2',
        'Cond3': '../prj-context-dependent-DM/data/strategy/Cond3',
        'Cond4': '../prj-context-dependent-DM/data/strategy/Cond4'
    }

    # 创建目标文件夹
    for folder_path in target_folders.values():
        os.makedirs(folder_path, exist_ok=True)

    # 遍历原始数据文件夹中的文件名
    for filename in os.listdir(source_folder):
        # 判断文件名是否以 ".json" 结尾，并且不包含 "strategy" 字符串
        if filename.endswith('.json') and 'strategy' in filename:
            # 分割文件名
            parts = filename.split('_')
            # 提取条件信息
            condition = parts[1]
            # 确定目标文件夹路径
            target_folder = target_folders.get(condition)
            if target_folder:
                # 构建源文件路径和目标文件路径
                source_file_path = os.path.join(source_folder, filename)
                target_file_path = os.path.join(target_folder, filename)
                # 复制文件到目标文件夹
                shutil.copyfile(source_file_path, target_file_path)
                print(f'Copied {filename} to {target_folder}')
            else:
                print(f'Condition {condition} not found in target_folders')

# groupstrategy()

# 遍历每个条件的文件夹
for i, path in enumerate(folder_path):
    input_dir = path
    output_dir = '../prj-context-dependent-DM/data/drawFig/Cond' + str(i + 1)
    exoutput_dir = '../prj-context-dependent-DM/data/extract/Cond' + str(i + 1)
    fitoutput_dir = '../prj-context-dependent-DM/data/fit/Cond' + str(i + 1)
    json2csv(input_dir, output_dir, exoutput_dir, i + 1)


