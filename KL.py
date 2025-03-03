import pandas as pd
import numpy as np

# 指定文件路径
file_path = "/home/lwh/projects/ddk/VGAE/data/BEELINE-data/inputs/scRNA-Seq/mHSC-E/noPCA1/result_r_5_k_10/preds_curr_300.txt"

# 读取数据到DataFrame中
try:
    df = pd.read_csv(file_path, header=None)

    # 假设数据在单列中
    data = df[0]

    # 将数据分为两段
    mid_index = len(data) // 2
    data1 = data[:mid_index]
    data2 = data[mid_index:]

    # 定义区间范围
    bins = np.arange(0, 1.05, 0.05)

    # 计算每段数据每个区间内的值的数量和概率
    counts1, _ = np.histogram(data1, bins)
    probabilities1 = counts1 / sum(counts1)

    counts2, _ = np.histogram(data2, bins)
    probabilities2 = counts2 / sum(counts2)

    # 保存每段数据的区间及其概率分布到ndarray
    distribution1 = np.zeros((len(bins) - 1, 3))
    distribution2 = np.zeros((len(bins) - 1, 3))
    for i in range(len(bins) - 1):
        distribution1[i] = [bins[i], bins[i + 1], probabilities1[i]]
        distribution2[i] = [bins[i], bins[i + 1], probabilities2[i]]

    # 设置numpy打印选项
    np.set_printoptions(precision=4, suppress=True)
    # 打印ndarray
    print("第一段数据的区间及其概率分布：")
    print(distribution1[: , 2])

    print("第二段数据的区间及其概率分布：")
    print(distribution2[: , 2])
    print(1)
except FileNotFoundError:
    print("指定路径的文件不存在。")
except Exception as e:
    print(f"发生错误：{e}")
