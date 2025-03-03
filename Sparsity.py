import numpy as np
import pandas as pd
import os
'''
base_dir = os.getcwd()
data_path = base_dir+ '/data/BEELINE-data/inputs/scRNA-Seq/mHSC-GM/'
data_file = data_path + 'ExpressionData.csv'
'''
# 创建表达矩阵
datapath='D:/TF_target/vgae_pytorch-master/data/BEELINE-data/inputs/scRNA-Seq/mHSC-L/'
data_file = datapath + 'ExpressionData.csv'
expression_matrix = pd.read_csv(data_file, header=0, index_col=0, encoding='utf-8')


# 计算非零元素个数
num_nonzero_entries = np.count_nonzero(expression_matrix)
total_entries = expression_matrix.size
sparsity = 1 - (num_nonzero_entries / total_entries)

print(f"Number of nonzero entries: {num_nonzero_entries}")
print(f"Sparsity: {sparsity:.2%}")
print('mHSC-E: 0.425')
print('mDC: 0.463')
print('hHep: 0.469')
print('mESC: 0.232')
print('hESC: 0.510')
print('mHSC-GM: 0.424')
print('mHSC-L: 0.437')
print(1)
