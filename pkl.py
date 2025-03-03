import pickle

# 指定文件路径
file_path = '/home/lwh/projects/ddk/VGAE/data/BEELINE-data/inputs/scRNA-Seq/mHSC-E/ceshiwendingxing/result_r_1_k_10/evaluation_metrics.pkl'

# 读取 .pkl 文件
with open(file_path, 'rb') as file:
    data = pickle.load(file)

# 打印数据
print(data)