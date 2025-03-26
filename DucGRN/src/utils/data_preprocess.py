import numpy as np
import pandas as pd
import sys
import pickle as pkl
import scipy.sparse as sp
import torch.nn.functional as F
def prepareData(data_file, label_file, reverse_flags=0):
    # TF_file = pd.read_csv(TF_path)["TF"]
    # TF_file = pd.read_csv(TF_file)
    label_file = pd.read_csv(label_file, header=0, sep=',', encoding='utf-8', usecols=[0, 1])
    shuffled_label_file = label_file.sample(frac=1, random_state=42).reset_index(drop=True)

    # 取其中的25%
    label_file = shuffled_label_file.sample(frac=0.10, random_state=42)
    # label_file.iloc[1:] = label_file.iloc[1:].applymap(lambda x: x.lower())
    '''
    with h5py.File(data_path, "r") as hdf5_file:
        df = hdf5_file["expression_log2cpm"][:, :1000]
        df = pd.DataFrame(df)
        gene_name = [name.decode('utf-8') for name in hdf5_file["gene_name"][:]]
        rnaseq_id = [name.decode('utf-8') for name in hdf5_file["rnaseq_id"][:]]
    gene_name_series = pd.Series(gene_name)

    # 获取 TF_file 中元素在 gene_name 中的索引
    indexes = gene_name_series[gene_name_series.isin(TF_file.iloc[:, 0])].index.tolist()
    data = df                
    '''
    data = pd.read_csv(data_file, header=0, index_col=0, encoding='utf-8')  ###transpose for six datasets of BEELINE
    # sampled_data = data.sample(n=20, axis=1, random_state=42)
    data.index = data.index.str.lower()

    print("Read data completed! Normalize data now!")
    feature = data.transform(lambda x: np.log(x + 1))
    print("Data normalized and logged!")

    # TF = set(TF_file)
    # TF = ', '.join(TF)
    # Adjacency matrix transformation
    labels = []
    if reverse_flags == 0:
        # var_names = list(gene_name)
        var_names = data.index.tolist()
        var_names = ', '.join(var_names)
        var_names = var_names.split(', ')

        num_genes = len(var_names)
        AM = np.zeros([num_genes, num_genes])

        '''
        for row_index, row in label_file.iterrows():
            if row[0] in var_names and row[1] in var_names:
                AM[var_names.index(row[0]), var_names.index(row[1])] = 1
                label_triplet = []
                label_triplet.append(var_names.index(row[0]))
                label_triplet.append(var_names.index(row[1]))
                label_triplet.append(1)
                labels.append(label_triplet)
        '''
        for index, row in label_file.iterrows():
            gene1_index = row['Gene1']
            gene2_index = row['Gene2']
            AM[gene1_index, gene2_index] = 1

    labels = np.array(label_file)
    # labels = np.array(labels)
    AM = AM + AM.T - np.diag(AM.diagonal())
    adj = sp.csr_matrix(AM)
    return feature, adj, var_names, labels