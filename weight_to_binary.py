'''
import argparse
import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import coo_matrix
from train import prepareData, preprocess_graph, sparse_to_tuple, mask_test_edges, get_roc_score
parser = argparse.ArgumentParser()
parser.add_argument("-sample", "--sample_type", type=str,default='hHep', help="sample models used")
args = parser.parse_args()
import torch
from sklearn.decomposition import PCA
from model import NHGATModelGAN

sample_type = args.sample_type
print('processing type ' + sample_type)
k_fold_num = 10
replicates_num = 20

base_dir = os.getcwd()
data_path = base_dir + '/data/BEELINE-data/inputs/scRNA-Seq/hHep/'
data_file = data_path + 'ExpressionData.csv'
label_file = data_path + 'filtered_refNetwork.csv'
tf = pd.read_csv(base_dir + "/data/BEELINE-data/inputs/scRNA-Seq/{}/ExpressionData.csv".format(sample_type), header=None).iloc[:, 0]
len_ = tf.shape[0]

for m in range(1, replicates_num+1):
    np.random.seed(m)


    for k_th in range(1, k_fold_num+1):
        features, adj, var_names, labels = prepareData(data_file, label_file, 0)
        dir_train_result = data_path + "/output/result_r_{}_k_{}".format(str(m), str(k_fold_num))

        pca = PCA(n_components=300)
        pca.fit(features)
        features = pd.DataFrame(pca.transform(features))  # 降维后的结果

        explained_variance_ratio = pca.explained_variance_ratio_  # 降维后的各主成分的方差值占总方差值的比例，即方差贡献率
        explained_variance = pca.explained_variance_  # 降维后的各主成分的方差值

        features = sparse_to_tuple(sp.coo_matrix(features.values))
        # features = sparse_to_tuple(features.tocoo())
        num_features = features[2][1]
        features_nonzero = features[1].shape[0]

        # Store original adjacency matrix (without diagonal entries) for later

        adj_orig = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
        adj_orig.eliminate_zeros()
        print("processing {} cell for k_fold {}".format(sample_type, str(k_th)))
        adj_train, train_edges, adj_val, val_edges, val_edges_false, test_edges, test_edges_false, edges_all, edges_all_false = mask_test_edges(
            adj, k_th)

        adj = adj_train
        adj_norm_train = preprocess_graph(adj)
        adj_label_train = adj_train + sp.eye(adj_train.shape[0])
        adj_label_train = torch.FloatTensor(adj_label_train.toarray())
        pos_weight_train = torch.Tensor([float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()])
        norm_train = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

        adj_norm_train = torch.sparse.FloatTensor(torch.LongTensor(adj_norm_train[0].T),
                                                  torch.FloatTensor(adj_norm_train[1]),
                                                  torch.Size(adj_norm_train[2]))
        features = torch.sparse.FloatTensor(torch.LongTensor(features[0].T).long(),
                                            torch.FloatTensor(features[1]),
                                            torch.Size(features[2])).to_dense()
        model = NHGATModelGAN(num_features, hidden_dim1=args.hidden_dim1, hidden_dim2=args.hidden_dim2,
                              hidden_dim3=args.hidden_dim3, num_heads=args.num_heads, dropout=args.dropout,
                              alpha=args.alpha, vae_bool=args.vae_bool)

        # define optimizer
        model.load_state_dict(torch.load(dir_train_result + '/checkpoint_k_{}.pt'.format(str(k_th))))
        A_pred, dis_real, dis_fake, mu, logvar = model(features, adj_norm_train)
        A_pred = A_pred.data.numpy()





        auroc, auprc, roc_score, ap_score, preds_curr = get_roc_score(A_pred, adj_orig, test_edges, test_edges_false)
        thresh = np.median(preds_curr)


        matrix_binary = (A_pred > thresh) + 0
        matrix_binary = pd.DataFrame(matrix_binary)

        result_tmp = result_tmp + matrix_binary

        final_tmp = pd.concat([tf.T, matrix_binary], axis=0)
        tf_new = pd.concat([pd.DataFrame(['gene']), tf], axis=0)
        final_matrix = pd.concat([tf_new, final_tmp], axis=1)

        final_matrix.to_csv(dir_train_result +
                            "/adj_matrix_predicted_binary_k_" + str(k_th) + ".txt",
                            float_format='%d', sep='\t', index=False, header=None)
        print("     {} KFold = {} :  replicate index = {}, k fold index = {} prediction determined".format(
            sample_type,str(k_fold_num), str(m), str(k_th)))

    final_result_tmp = pd.concat([tf.T,  pd.DataFrame(result_tmp)], axis=0)
    final_result = pd.concat([tf_new,final_result_tmp], axis=1)

    final_result.to_csv(dir_train_result +
                        "/adj_matrix_predicted_binary_sum.txt",
                        float_format='%d', sep='\t', index=False, header=None)

    result_tmp_2 = (result_tmp >= int(k_fold_num*0.6) ) + 0
    final_result_tmp_2 = pd.concat([tf.T, pd.DataFrame(result_tmp_2)], axis=0)
    final_result_2 = pd.concat([tf_new, final_result_tmp_2], axis=1)
    final_result_2.to_csv(dir_train_result +
                        "/adj_matrix_predicted_binary_final.txt",
                        float_format='%d', sep='\t', index=False, header=None)
    print("{} KFold = {} : replicate index = {} FINAL prediction determined".format(
            sample_type,str(k_fold_num), str(i)))

'''
