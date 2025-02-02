import numpy as np
import pandas as pd
import scipy.io as sio
from tqdm import tqdm  # 可选，用于显示进度条

def newrandn():
    """
    生成一个随机奖励值，基于正态分布，调整均值和标准差，并限制在1到5之间。
    
    Returns:
        float: 生成的奖励值，范围在1到5之间，保留一位小数。
    """
    # 生成标准正态分布随机数，调整均值为3，标准差为0.6，并四舍五入到一位小数
    r = np.round(np.random.randn() * 0.6 + 3, 1)
    # 限制奖励值在1到5之间
    r = max(1.0, min(5.0, r))
    return r

def simulation_optimal_policy():
    """
    模拟最优策略并将结果存储为与参与者原始数据相同格式的CSV文件。
    
    主要步骤：
    1. 初始化模拟参数。
    2. 加载决策矩阵deltaVmat。
    3. 进行模拟循环，生成行动和奖励数据。
    4. 存储模拟数据并保存为CSV文件。
    """
    import numpy as np
    import pandas as pd
    import scipy.io as sio
    from tqdm import tqdm  # 可选，用于显示进度条

    # 模拟参数
    times = 100000  # 每种总天数下的重复次数
    totalday_range = range(5, 11)  # 总天数从5到10天
    num_totaldays = len(totalday_range)  # 总天数的数量

    # 设置随机数生成器种子（基于当前时间，确保每次运行结果不同）
    np.random.seed()

    # 加载deltaVmat矩阵
    try:
        mat_data = sio.loadmat('deltaV_optimal.mat')
        if 'deltaVmat' in mat_data:
            deltaVmat = mat_data['deltaVmat']
        else:
            raise KeyError("在 'deltaV_optimal.mat' 文件中未找到 'deltaVmat' 变量。")
    except FileNotFoundError:
        print("错误：未找到文件 'deltaV_optimal.mat'。请确保文件存在于当前目录中。")
        return
    except KeyError as e:
        print(e)
        return

    # 检查deltaVmat的维度
    if not isinstance(deltaVmat, np.ndarray):
        print("错误：'deltaVmat' 不是一个NumPy数组。")
        return
    if deltaVmat.ndim != 2:
        print("错误：'deltaVmat' 不是一个二维矩阵。")
        return

    # 初始化模拟数据列表
    # 使用列表而不是预分配数组，以减少内存占用和提高效率
    simudata = []

    trial = 0  # 试验编号

    # 外层循环：遍历不同的总天数
    for totalday in totalday_range:
        # 内层循环：重复times次模拟
        # 使用tqdm显示进度条（可选）
        repeats = tqdm(range(times), desc=f"Simulating totalday={totalday}") if 'tqdm' in globals() else range(times)
        for _ in repeats:
            trial += 1  # 试验编号递增

            # 第1天的模拟
            reward = newrandn()
            rstar = reward
            avereward = reward
            simudata.append([trial, totalday, 1, np.nan, 1, reward, rstar, avereward])

            # 后续天数的模拟（第2天到totalday天）
            for days in range(2, totalday + 1):
                # 计算deltaV的索引
                # MATLAB中的索引从1开始，Python从0开始
                row_idx = int(round((rstar - 1) / 0.1))
                col_idx = totalday - days

                # 确保索引在deltaVmat的范围内
                row_idx = min(max(row_idx, 0), deltaVmat.shape[0] - 1)
                col_idx = min(max(col_idx, 0), deltaVmat.shape[1] - 1)

                # 获取deltaV值
                deltaV = deltaVmat[row_idx, col_idx]

                # 决定行动（action）
                if deltaV > 0:
                    action = 1
                elif deltaV < 0:
                    action = 0
                else:
                    action = 1 if np.random.rand() > 0.5 else 0

                # 根据行动生成奖励
                reward = newrandn() if action == 1 else rstar

                # 更新平均奖励
                avereward = (avereward * (days - 1) + reward) / days

                # 更新当前最佳奖励
                rstar = max(rstar, reward)

                # 存储当前天的数据
                simudata.append([trial, totalday, days, np.nan, action, reward, rstar, avereward])

    # 转换为Pandas DataFrame
    df_simudata = pd.DataFrame(simudata, columns=[
        'trial',        # 试验编号
        'totalday',     # 总天数
        'days',         # 当前天数
        'nan',          # 未使用或占位符
        'action',       # 行动（1或0）
        'reward',       # 奖励值
        'rstar',        # 当前最佳奖励
        'avereward'     # 平均奖励
    ])

    # 保存模拟数据为CSV文件
    filename = f'optimal_policy_simu{times}.csv'
    df_simudata.to_csv(filename, index=False)
    print(f"模拟完成。数据已保存到 '{filename}'。")

if __name__ == "__main__":
    simulation_optimal_policy()