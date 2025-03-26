import torch

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