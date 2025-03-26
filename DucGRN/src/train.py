import torch
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from itertools import product, permutations, combinations, combinations_with_replacement
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import json
import scipy.sparse as sp
import numpy as np
import pickle
import os
import time
import pandas as pd
# from input_data import load_data
from preprocessing import *
import define_optimizer
import args
import copy
import torch.nn as nn
from src.models.model import NHGATModelGAN
from src.utils.data_preprocess import prepareData
from src.loss import varga_loss_function
import argparse
import warnings
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore", message="'dropout_adj' is deprecated")
base_dir = os.getcwd()
data_path = base_dir + '/data/BEELINE-data/inputs/scRNA-Seq/mHSC-E/'
data_file = data_path + 'ExpressionData.csv'
label_file = data_path + 'TF_reNetwork.csv'

k_fold_num = 10
replicate_num = 10


for m in range(1, replicate_num + 1):
    np.random.seed(m)
    for k_th in range(1, 10):

        features, adj, var_names, labels = prepareData(data_file, label_file, 0)
        dir_train_result = data_path + "/GAN/result_r_{}_k_{}".format(str(m), str(k_fold_num))
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
        print("processing {} cell for k_fold {}".format(sample_type, str(k_th)))
        adj_train, train_edges, adj_val, val_edges, val_edges_false, test_edges, test_edges_false, edges_all, edges_all_false = mask_test_edges(
            adj, 1)
        num_edges = np.size(train_edges, 0)
        edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))
        # edge_index_train = torch.randint(0, num_nodes, (2, num_edges))  # 边索引
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

        # adj_label = adj_train + sp.eye(adj_train.shape[0])
        # adj_label = sparse_to_tuple(adj_label)

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
        # model = GCNModelVAE(num_features, hidden_dim1=args.hidden_dim1, hidden_dim2=args.hidden_dim2, dropout=args.dropout,
        #                    vae_bool=args.vae_bool)

        # model = VGAE(num_features, hidden_dim1=args.hidden_dim1, hidden_dim2=args.hidden_dim2, hidden_dim3=args.hidden_dim3, num_heads=args.num_heads, dropout=args.dropout, alpha=args.alpha)
        model = NHGATModelGAN(num_features, hidden_dim1=args.hidden_dim1, hidden_dim2=args.hidden_dim2,
                              hidden_dim3=args.hidden_dim3, num_heads=args.num_heads, dropout=args.dropout,
                              alpha=args.alpha, vae_bool=args.vae_bool)
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
        # optimizer = Adam(model.parameters(), lr=args.lr)
        best_valid_val_roc = float('-inf')
        model.train()

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
        adj_label_train = adj_label_train.to(DEVICE)

        weight_tensor = weight_tensor.to(DEVICE)
        adj_label_val = adj_label_val.to(DEVICE)

        weight_tensor_val = weight_tensor_val.to(DEVICE)
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
            train_time = time.time() - t_train


            t_val = time.time()
            model.eval()
            A_pred_val, dis_real_val, dis_fake_val, mu_val, logvar_val = model(features, adj_norm_val)

            val_loss = varga_loss_function(preds=A_pred_val, labels=adj_label_val,
                                           mu=mu_val, logvar=logvar_val,
                                           dis_real=dis_real_val, dis_fake=dis_fake_val,
                                           norm=norm_val, weight=weight_tensor_val, sparsity_weight=0.1)
            val_time = time.time() - t_val

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

                thresh_ = np.median(preds_curr)
                preds_curr_binary = (preds_curr >= thresh_) + 0
                temp = preds_curr_binary + label_test
                test_accuracy_curr = (sum(temp == 2) + sum(temp == 0)) / len(temp)
                test_accuracy_for_record.append(test_accuracy_curr)
                # roc_test, ap_test = get_scores(adj_orig, test_edges, test_edges_false, A_pred)
                print("\n##   Epoch:", '%04d' % (epoch + 1), "test_accuracy=", "{:.5f}\n".format(test_accuracy_curr),
                      "test_roc=", "{:.5f}".format(roc_score), "test_prc=", "{:.5f}".format(auprc), "test_ap=",
                      "{:.5f}".format(ap_score))

        print("Optimization Finished for the " + str(k_th) + "th model!")
        # torch.save(model.state_dict(), dir_train_result + '/checkpoint_k_{}.pt'.format(str(k_th)))
        roc_and_ap = list(zip(test_roc, test_auprc, test_ap))
        np.savetxt(dir_train_result + '/test_rco_ap' + str(k_th) + '.txt', roc_and_ap, fmt='%.6f %.6f %.6f ',
                   delimiter='\t', header='AUROC\tAUPRC\tAP')
        np.savetxt(dir_train_result + '/test_accuracy' + str(k_th) + '.txt', test_accuracy_for_record, fmt='%.6f',
                   delimiter='\t', header='test_accuracy')

        # 创建 DataFrame
        losses_df = pd.DataFrame({
            'Epoch': range(1, args.epochs + 1),
            'Train Loss': train_losses,
            'Validation Loss': val_losses
        })

        losses_file_path = os.path.join(dir_train_result, 'losses.csv')
        losses_df.to_csv(losses_file_path, index=False)

        file_path = os.path.join(dir_train_result, 'evaluation_metrics.pkl')
        with open(file_path, 'wb') as file:
            pickle.dump((test_fpr, test_tpr, test_recall, test_precision), file)
        np.savetxt(dir_train_result + '/adj_matrix_' + str(k_th) + '.txt', A_pred, delimiter=',')

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








