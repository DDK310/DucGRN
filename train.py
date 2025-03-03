import torch
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from itertools import product, permutations, combinations, combinations_with_replacement
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from tqdm import tqdm
import json
import scipy.sparse as sp
import numpy as np
import pickle
import os
import time
import h5py
import pandas as pd
#from input_data import load_data
from preprocessing import *
import define_optimizer
import args
import copy
import torch.nn as nn
# from tools import EarlyStopping
#from model import VGAE
from model import NHGATModelGAN,GCNModelVAE
import argparse
import warnings
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use('Agg')
# Train on CPU (hide GPU) due to memory constraints
#DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE = torch.device('cuda')
# 输出当前使用设备
if DEVICE.type == 'cuda':
    print("当前使用的设备是 GPU")
else:
    print("当前使用的设备是 CPU")
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='VGAE')
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--hidden_dim1', type=int, default=128)
parser.add_argument('--hidden_dim2', type=int, default=64)
parser.add_argument('--hidden_dim3', type=int, default=64)
parser.add_argument('--num_heads', type=int, default=8)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--alpha', type=float, default=0.2)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_decay', type=float, default=0.0001)
parser.add_argument('--weight_decay', type=float, default=0.0001)
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--vae_bool', type=bool, default=True)
parser.add_argument('--optimizer', type=str, default='adam')
parser.add_argument("-sample", "--sample_type", type=str,default='hHSC', help="sample models used")
args = parser.parse_args()
optimizer_name = args.optimizer
lr = args.lr
lr_decay=args.lr_decay
weight_decay=args.weight_decay
gamma=args.gamma
momentum=args.momentum
sample_type = args.sample_type


# 忽略特定类型的警告
warnings.filterwarnings("ignore", message="'dropout_adj' is deprecated")
def prepareData(data_file, label_file,reverse_flags = 0):

    #TF_file = pd.read_csv(TF_path)["TF"]
    #TF_file = pd.read_csv(TF_file)
    label_file = pd.read_csv(label_file, header=0, sep=',', encoding='utf-8', usecols=[0, 1])
    shuffled_label_file = label_file.sample(frac=1, random_state=42).reset_index(drop=True)

    # 取其中的25%
    label_file = shuffled_label_file.sample(frac=0.10, random_state=42)
    #label_file.iloc[1:] = label_file.iloc[1:].applymap(lambda x: x.lower())
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
    #sampled_data = data.sample(n=20, axis=1, random_state=42)
    data.index = data.index.str.lower()


    print("Read data completed! Normalize data now!")
    feature = data.transform(lambda x: np.log(x + 1))
    print("Data normalized and logged!")

    #TF = set(TF_file)
    #TF = ', '.join(TF)
    # Adjacency matrix transformation
    labels = []
    if reverse_flags == 0:
        #var_names = list(gene_name)
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

    '''
    labels_df = pd.DataFrame(labels, columns=['Gene1', 'Gene2', 'Relation'])
    # 拼接文件路径
    output_file_path = os.path.join(data_path, 'STRING-network_refNetwork.csv')
    # 将筛选后的标签数据保存为 CSV 文件
    labels_df.to_csv(output_file_path, index=False)
    '''
    labels = np.array(label_file)
    #labels = np.array(labels)
    AM = AM + AM.T - np.diag(AM.diagonal())
    adj = sp.csr_matrix(AM)
    return feature, adj, var_names, labels

def evaluateEPR(output, label, TFs, Genes):
	label_set = set(label['Gene1']+'|'+label['Gene2'])
	output= output.iloc[:len(label_set)]
	EPR = len(set(output['Gene1']+'|' +output['Gene2']) & label_set) / (len(label_set)**2/(len(TFs)*len(Genes)-len(TFs)))
	return EPR

def get_scores(adj_orig, edges_pos, edges_neg, adj_rec):

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    preds = []
    pos = []
    for e in edges_pos:
        # print(e)
        # print(adj_rec[e[0], e[1]])
        preds.append(adj_rec[e[0], e[1]].item())
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(adj_rec[e[0], e[1]].data)
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score

'''
def get_ranked_edges(adj):
    num_nodes = adj.shape[0]
    # 初始化排名边缘列表
    ranked_edges = []
    # 遍历邻接矩阵的每一个元素，记录排名（只考虑值大于等于0.99的元素）
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if A_pred[i][j] >= 0.99:
                ranked_edges.append((i, j, A_pred[i][j]))
    # 按照排名值从高到低排序
    ranked_edges.sort(key=lambda x: x[2], reverse=True)
    ranked_edges = pd.DataFrame(ranked_edges, columns=['Gene1', 'Gene2', 'Relation'])
    return ranked_edges
'''
def get_acc(adj_rec, labels, edges):

    preds_count = (adj_rec == 1).sum().float()
    '''
    labels_all = adj_label.to_dense().view(-1).long()
    preds_all = (adj_rec == 1 ).view(-1).long()
    #preds_tf = (adj_rec[indexes, :] > 0.5).view(-1).long()

    # 计算 labels_all 中值为1的总个数
    all_acc = (preds_all == labels_all).sum().float() / labels_all.size(0)
    '''
    all_edge_values = adj_rec[labels[:, 0], labels[:, 1]]
    all_count = torch.sum(all_edge_values == 1).item()
    all_num_edges = labels.shape[0]
    all_acc = all_count / all_num_edges

    edge_values = adj_rec[edges[:, 0], edges[:, 1]]
    # 检查这些值是否等于 1，并计算等于 1 的数量
    count = torch.sum(edge_values == 1).item()
    num_edges = test_edges.shape[0]
    # 计算准确率
    acc = count / num_edges
    return all_acc, acc, preds_count

def get_roc_score(adj_rec, adj_orig, edges_pos, edges_neg):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    adj_rec = adj_rec
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(adj_rec[e[0], e[1]])
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(adj_rec[e[0], e[1]])
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])

    labels_all = np.hstack([np.ones(len(edges_pos)), np.zeros(len(edges_neg))])
    fpr, tpr, thresholds1 = roc_curve(labels_all, preds_all)
    auroc = auc(fpr, tpr)
    roc_score = roc_auc_score(labels_all, preds_all)
    precision, recall, thresholds2 = precision_recall_curve(labels_all, preds_all, pos_label=1)
    auprc = auc(recall, precision)
    ap_score = average_precision_score(labels_all, preds_all)

    return fpr, tpr,precision, recall,auroc, auprc, roc_score, ap_score, preds_all

def get_preds_all(adj_rec, adj_orig, edges_pos, edges_neg):
    preds = []

    for e in edges_pos:
        preds.append(adj_rec[e[0], e[1]])
    preds_neg = []
    for e in edges_neg:
        preds_neg.append(adj_rec[e[0], e[1]])
    preds_all = np.hstack([preds, preds_neg])
    return preds_all

def varga_loss_function(preds, labels, dis_real, dis_fake, mu, logvar, norm, weight, sparsity_weight=0.1):
    # 对抗变分图正则化图自编码损失：生成和判别的loss
    cost = norm * F.binary_cross_entropy(preds.view(-1), labels.to_dense().view(-1), weight=weight)
    KLD = 0.5 / preds.size(0) * (1 + 2 * logvar - mu ** 2 - torch.exp(logvar) ** 2).sum(1).mean()
    dis_real_loss = F.binary_cross_entropy(dis_real, torch.ones(dis_real.shape).to(DEVICE))
    dis_fake_loss = F.binary_cross_entropy(dis_fake, torch.zeros(dis_fake.shape).to(DEVICE))
    sparsity_loss = torch.mean(torch.abs(preds))
    #
    return cost - KLD+ '''dis_real_loss + dis_fake_loss''' +sparsity_weight * sparsity_loss

def vgae_loss_function(preds, labels, mu, logvar, norm, weight):
    '''
    变分图自编码，损失函数包括两部分：
        1.生成图和原始图之间的距离度量
        2.节点表示向量分布和正态分布的KL散度
    '''
    # 负样本边的weight都为1，正样本边的weight都为pos_weight
    cost = norm * F.binary_cross_entropy_with_logits(preds.view(-1), labels.to_dense().view(-1), weight=weight)
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 / preds.size(0) * torch.mean(torch.sum(
        1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
    return cost + KLD
def remove_edges(all, part):
    all = all.detach().numpy()
    tmp = copy.deepcopy(all)
    for i in range(len(part)):
        tmp[part[i,0],part[i,1]]=np.nan
    all_new = tmp[~np.isnan(tmp)]
    all_new = torch.from_numpy(np.array(all_new))
    return all_new

def retain_edges(all, part):
    all = all.detach().numpy()
    tmp = copy.deepcopy(all)
    all_new = []
    for i in range(len(part)):
        all_new.append(tmp[part[i, 0], part[i, 1]])
    all_new = torch.from_numpy(np.array(all_new))
    return all_new

base_dir = os.getcwd()
'''
data_file = input_path + '/new_expression_data.h5'
label_file = input_path + '/TF_Target for matrix.csv`
TF_path = input_path + '/TF.csv'

'''

data_path = base_dir+ '/data/BEELINE-data/inputs/scRNA-Seq/mHSC-E/'
data_file = data_path + 'ExpressionData.csv'
label_file = data_path + 'TF_reNetwork.csv'
#label_file = base_dir+'/data/BEELINE-Networks/Networks/mouse/STRING-network.csv'

#TF_file = input_path + '/data/BEELINE-Networks/Networks/mouse/mDC-ChIP-seq-network.csv'
k_fold_num = 10
replicate_num = 10

def generate_simulated_data(num_genes, num_cells, sparsity=0.01):
    """
    生成模拟的基因表达数据、邻接矩阵、基因名称和标签
    :param num_genes: 基因数量（节点数）
    :param num_cells: 细胞数量
    :param sparsity: 邻接矩阵的稀疏性
    :return: 基因表达矩阵（features），邻接矩阵（adj_matrix），基因名称（var_names），标签（labels）
    """
    # 随机生成基因表达矩阵（稠密）
    features = np.random.rand(num_cells, num_genes)

    # 随机生成邻接矩阵（稀疏）
    adj_matrix = sp.random(num_genes, num_genes, density=sparsity, format='coo')
    adj_matrix = adj_matrix.toarray()  # 转为稠密矩阵以便后续处理
    adj_matrix = np.triu(adj_matrix, k=1)  # 保留上三角部分，避免重复边
    adj_matrix = adj_matrix + adj_matrix.T  # 对称化，确保无向图

    # 生成基因名称列表
    var_names = [f"Gene_{i+1}" for i in range(num_genes)]

    # 根据邻接矩阵生成标签列表
    labels = []
    for i in range(num_genes):
        for j in range(i + 1, num_genes):  # 遍历上三角部分
            if adj_matrix[i, j] > 0:  # 如果存在边
                labels.append([var_names[i], var_names[j], 1])  # [基因1, 基因2, 关系]

    labels = np.array(labels)  # 转为NumPy数组

    # 返回结果
    return features, sp.csr_matrix(adj_matrix), var_names, labels


for m in range(1, replicate_num + 1):
    np.random.seed(m)
    for k_th in range(1, 2):

        num_genes = 10  # 基因数量
        num_cells = 5  # 细胞数量
        features, adj_matrix, var_names, labels = generate_simulated_data(num_genes, num_cells)
        #features, adj, var_names, labels = prepareData(data_file, label_file, 0)
        dir_train_result = data_path + "/-GAN/result_r_{}_k_{}".format(str(m), str(k_fold_num))
        if not os.path.exists(dir_train_result):
            os.makedirs(dir_train_result)

        # draw_graph(adj.toarray(),filename="my_graph.png")
        pca = PCA(n_components=300)
        pca.fit(features)
        features = pd.DataFrame(pca.transform(features))  # 降维后的结果

        explained_variance_ratio = pca.explained_variance_ratio_  # 降维后的各主成分的方差值占总方差值的比例，即方差贡献率
        explained_variance = pca.explained_variance_  # 降维后的各主成分的方差值

        features = sparse_to_tuple(sp.coo_matrix(features.values))
        # features = sparse_to_tuple(features.tocoo())
        num_features = features[2][1]
        num_nodes = features[2][0]
        features_nonzero = features[1].shape[0]



        # Store original adjacency matrix (without diagonal entries) for later

        adj_orig = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
        adj_orig.eliminate_zeros()
        print("processing {} cell for k_fold {}".format(sample_type,str(k_th)))
        adj_train, train_edges, adj_val, val_edges, val_edges_false, test_edges, test_edges_false,edges_all,edges_all_false = mask_test_edges(adj,1)
        num_edges = np.size(train_edges, 0)
        edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))
        #edge_index_train = torch.randint(0, num_nodes, (2, num_edges))  # 边索引
        batch_train = torch.zeros(num_nodes, dtype=torch.long)  # 批处理索引
        adj = adj_train
        adj_norm_train = preprocess_graph(adj)
        adj_label_train = adj_train + sp.eye(adj_train.shape[0])
        adj_label_train = torch.FloatTensor(adj_label_train.toarray())
        pos_weight_train = torch.Tensor([float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()])
        norm_train = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

        adj = adj_val
        adj_norm_val = preprocess_graph(adj)
        adj_label_val = adj_val + sp.eye(adj_val.shape[0])
        adj_label_val = torch.FloatTensor(adj_label_val.toarray())
        pos_weight_val = torch.Tensor([float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()])
        norm_val = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)



        #adj_label = adj_train + sp.eye(adj_train.shape[0])
        #adj_label = sparse_to_tuple(adj_label)

        adj_norm_train = torch.sparse.FloatTensor(torch.LongTensor(adj_norm_train[0].T),
                                    torch.FloatTensor(adj_norm_train[1]),
                                    torch.Size(adj_norm_train[2]))


        adj_norm_val = torch.sparse.FloatTensor(torch.LongTensor(adj_norm_val[0].T),
                                    torch.FloatTensor(adj_norm_val[1]),
                                    torch.Size(adj_norm_val[2]))

        features = torch.sparse.FloatTensor(torch.LongTensor(features[0].T).long(),
                                            torch.FloatTensor(features[1]),
                                            torch.Size(features[2])).to_dense()

        weight_mask = adj_label_train.to_dense().view(-1) == 1
        weight_tensor = torch.ones(weight_mask.size(0))
        weight_tensor[weight_mask] = pos_weight_train

        weight_mask_val = adj_label_val.to_dense().view(-1) == 1
        weight_tensor_val = torch.ones(weight_mask_val.size(0))
        weight_tensor_val[weight_mask_val] = pos_weight_val

        # init model and optimizer
        #model = GCNModelVAE(num_features, hidden_dim1=args.hidden_dim1, hidden_dim2=args.hidden_dim2, dropout=args.dropout,
        #                    vae_bool=args.vae_bool)

        #model = VGAE(num_features, hidden_dim1=args.hidden_dim1, hidden_dim2=args.hidden_dim2, hidden_dim3=args.hidden_dim3, num_heads=args.num_heads, dropout=args.dropout, alpha=args.alpha)
        model = NHGATModelGAN(num_features, hidden_dim1=args.hidden_dim1, hidden_dim2=args.hidden_dim2, hidden_dim3=args.hidden_dim3, num_heads=args.num_heads, dropout=args.dropout, alpha=args.alpha, vae_bool=args.vae_bool)
        # define optimizer
        if optimizer_name == 'adam':
            optimizer = define_optimizer.define_optimizer_adam(model, lr=lr, weight_decay=weight_decay)

        elif optimizer_name == 'adamw':
            optimizer = define_optimizer.define_optimizer_adamw(model, lr=lr, weight_decay=weight_decay)

        elif optimizer_name == 'sgd':
            optimizer = define_optimizer.define_optimizer_sgd(model, lr=lr, momentum=momentum,
                                                              weight_decay=weight_decay)

        elif optimizer_name == 'adagrad':
            optimizer = define_optimizer.define_optimizer_adagrad(model, lr=lr, lr_decay=lr_decay,
                                                                  weight_decay=weight_decay)

        elif optimizer_name == 'rmsprop':
            optimizer = define_optimizer.define_optimizer_rmsprop(model, lr=lr, weight_decay=weight_decay,
                                                                  momentum=momentum)

        elif optimizer_name == 'adadelta':
            optimizer = define_optimizer.define_optimizer_adadelta(model, lr=lr, weight_decay=weight_decay)

        else:
            raise NameError('No define optimization function name!')


        print("for the " + str(k_th) + " model training")  # print(model)
        #optimizer = Adam(model.parameters(), lr=args.lr)
        best_valid_val_roc = float('-inf')
        model.train()
        '''
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10,
                                      verbose=True, threshold=1e-4, threshold_mode='rel',
                                      cooldown=0, min_lr=0, eps=1e-8)

        early_stopping = EarlyStopping(patience=10, verbose=True,
                                       path=dir_train_result + '/checkpoint_k_{}.pt'.format(str(k_th)))
        '''
        hidden_emb = None
        train_loss_record = []
        test_fpr = []
        test_tpr = []
        test_thresholds1 = []
        test_auroc = []

        test_recall = []
        test_precision = []
        test_thresholds2 = []
        test_auprc = []
        test_ap = []
        test_roc = []
        test_accuracy_for_record = []

        train_losses = []
        val_losses = []

        label_test = np.hstack([np.ones(len(test_edges)), np.zeros(len(test_edges_false))])


        adj_norm_train = adj_norm_train.to(DEVICE)
        adj_norm_val = adj_norm_val.to(DEVICE)
        features = features.to(DEVICE)
        adj_label_train=adj_label_train.to(DEVICE)

        weight_tensor=weight_tensor.to(DEVICE)
        adj_label_val=adj_label_val.to(DEVICE)

        weight_tensor_val= weight_tensor_val.to(DEVICE)
        model = model.to(DEVICE)

        ex_val = copy.deepcopy(val_edges)
        ex_val[:, [0, 1]] = ex_val[:, [1, 0]]
        ex_val_false = copy.deepcopy(val_edges_false)
        ex_val_false = np.array(ex_val_false)
        ex_val_false[:, [0, 1]] = ex_val_false[:, [1, 0]]
        merged_coor_to_retain = np.vstack((val_edges, ex_val, val_edges_false, ex_val_false))

        ex_test = copy.deepcopy(test_edges)
        ex_test[:, [0, 1]] = ex_test[:, [1, 0]]
        ex_test_false = copy.deepcopy(test_edges_false)
        ex_test_false = np.array(ex_test_false)
        ex_test_false[:, [0, 1]] = ex_test_false[:, [1, 0]]
        merged_coor_to_remove = np.vstack((test_edges, ex_test, test_edges_false, ex_test_false, merged_coor_to_retain))


        # train model
        for epoch in range(args.epochs):
            # 记录训练时间
            t_train = time.time()
            model.train()

            optimizer.zero_grad()
            A_pred_train, dis_real_train, dis_fake_train, mu_train, logvar_train = model(features, adj_norm_train)

            loss = varga_loss_function(preds=A_pred_train, labels=adj_label_train,
                                       mu=mu_train, logvar=logvar_train,
                                       dis_real=dis_real_train, dis_fake=dis_fake_train,
                                       norm=norm_train, weight=weight_tensor, sparsity_weight=0.1)
            loss.backward()
            optimizer.step()
            train_time = time.time() - t_train  # 训练时间

            # 记录验证时间
            t_val = time.time()
            model.eval()
            A_pred_val, dis_real_val, dis_fake_val, mu_val, logvar_val = model(features, adj_norm_val)

            val_loss = varga_loss_function(preds=A_pred_val, labels=adj_label_val,
                                           mu=mu_val, logvar=logvar_val,
                                           dis_real=dis_real_val, dis_fake=dis_fake_val,
                                           norm=norm_val, weight=weight_tensor_val, sparsity_weight=0.1)
            val_time = time.time() - t_val  # 验证时间

            # 记录性能指标
            fpr, tpr, precision, recall, auroc, auprc, roc_score, ap_score, preds_curr = get_roc_score(
                A_pred_train.cpu().data.numpy(), adj_orig, test_edges, test_edges_false)

            print("Epoch:", '%04d' % (epoch + 1),
                  "train_loss=", "{:.5f}".format(loss.item()),
                  "val_loss=", "{:.5f}".format(val_loss.item()),
                  "val_roc=", "{:.5f}".format(roc_score),
                  "val_prc=", "{:.5f}".format(auprc),
                  "train_time=", "{:.5f}".format(train_time),
                  "val_time=", "{:.5f}".format(val_time))

            if ((epoch + 1) % 10 == 0):
                '''
                if((epoch + 1) >240):
                    np.savetxt(dir_train_result + '/preds_curr_' + str(epoch + 1) + '.txt', preds_curr)
                '''
                thresh_ = np.median(preds_curr)
                preds_curr_binary = (preds_curr >= thresh_) + 0
                temp = preds_curr_binary + label_test
                test_accuracy_curr = (sum(temp == 2) + sum(temp == 0)) / len(temp)
                test_accuracy_for_record.append(test_accuracy_curr)
                #roc_test, ap_test = get_scores(adj_orig, test_edges, test_edges_false, A_pred)
                print("\n##   Epoch:", '%04d' % (epoch + 1),"test_accuracy=", "{:.5f}\n".format(test_accuracy_curr),
                      "test_roc=", "{:.5f}".format(roc_score),"test_prc=", "{:.5f}".format(auprc), "test_ap=", "{:.5f}".format(ap_score))
            '''
            if ((epoch + 1) >290):

                test_fpr.append(fpr)
                test_tpr.append(tpr)
                test_recall.append(recall)
                test_precision.append(precision)

                A_pred[A_pred >= thresh_] = 1
                A_pred[A_pred < thresh_] = 0
                A_pred = A_pred.astype(int)
                np.savetxt(dir_train_result + '/adj_matrix_' + str(epoch + 1) + '.txt', A_pred, delimiter=',')
            '''

        print("Optimization Finished for the " + str(k_th) + "th model!")
        #torch.save(model.state_dict(), dir_train_result + '/checkpoint_k_{}.pt'.format(str(k_th)))
        roc_and_ap = list(zip(test_roc, test_auprc, test_ap))
        np.savetxt(dir_train_result + '/test_rco_ap' + str(k_th) + '.txt', roc_and_ap, fmt='%.6f %.6f %.6f ', delimiter='\t', header='AUROC\tAUPRC\tAP')
        np.savetxt(dir_train_result + '/test_accuracy'+ str(k_th) + '.txt', test_accuracy_for_record, fmt='%.6f',
                   delimiter='\t', header='test_accuracy')

        # 创建 DataFrame
        losses_df = pd.DataFrame({
            'Epoch': range(1, args.epochs + 1),
            'Train Loss': train_losses,
            'Validation Loss': val_losses
        })

        # 保存到 CSV 文件
        losses_file_path = os.path.join(dir_train_result, 'losses.csv')
        losses_df.to_csv(losses_file_path, index=False)


        file_path = os.path.join(dir_train_result, 'evaluation_metrics.pkl')
        with open(file_path, 'wb') as file:
            pickle.dump((test_fpr, test_tpr, test_recall, test_precision), file)
        #np.savetxt(dir_train_result + '/adj_matrix_' + str(k_th) + '.txt', A_pred, delimiter=',')
'''
        model.load_state_dict(torch.load(dir_train_result + '/checkpoint_k_{}.pt'.format(str(k_th))))
        A_pred, dis_real, dis_fake, mu, logvar = model(features, adj_norm_train)
        A_pred = A_pred.data.numpy()
        preds_curr_for_edges_all = get_preds_all(A_pred, adj_orig, edges_all,edges_all_false)
        ranked_edges = get_ranked_edges(A_pred)
        
        np.savetxt(
            dir_train_result + '/adj_matrix_ranked_edges' + str(k_th) + '.txt',
            ranked_edges, fmt='%d %d %.6f', delimiter='\t')
        edges_all = np.vstack((edges_all, edges_all_false))
        edges_all_with_score = np.vstack(
            (edges_all[:, 0], edges_all[:, 1], preds_curr_for_edges_all)).T
        np.savetxt(
            dir_train_result + '/adj_matrix_predicted_weighted' + str(k_th) + '.txt',
            edges_all_with_score, fmt='%d %d %.6f', delimiter='\t')
'''







