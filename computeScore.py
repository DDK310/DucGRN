from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from itertools import product, permutations, combinations, combinations_with_replacement
import torch
import pandas as pd
import numpy as np
import os
import torch.nn.functional as F
input_path = os.getcwd()
label_path = input_path + '/data/BEELINE-data/inputs/scRNA-Seq/mDC/filtered_refNetwork.csv'
trueEdgesDF = pd.read_csv(label_path, header=0, sep = ',', encoding='utf-8')
output_path = input_path +'/output'
A_pred = torch.load(input_path + '/data/BEELINE-data/inputs/scRNA-Seq/mDC/A_pred.pt')
A_pred = torch.from_numpy(A_pred)
A_pred = (A_pred > 0.9).float()
# 找到值为 1 的索引
indices = torch.nonzero(A_pred == 1)

# 将索引保存为 DataFrame
predEdgeDF = pd.DataFrame(indices.numpy(), columns=['Gene1', 'Gene2'])

possibleEdges = list(combinations(np.unique(trueEdgesDF.loc[:, ['Gene1', 'Gene2']]), r=2))
possibleEdges = [tuple(map(str, p)) for p in possibleEdges]
TrueEdgeDict = {'|'.join(p): 0 for p in possibleEdges}
PredEdgeDict = {'|'.join(p): 0 for p in possibleEdges}

# Compute TrueEdgeDict Dictionary
# 1 if edge is present in the ground-truth
# 0 if edge is not present in the ground-truth

for key in TrueEdgeDict.keys():
    if len(trueEdgesDF.loc[((trueEdgesDF['Gene1'].astype(str) == key.split('|')[0]) &
                            (trueEdgesDF['Gene2'].astype(str) == key.split('|')[1])) |
                           ((trueEdgesDF['Gene2'].astype(str) == key.split('|')[0]) &
                            (trueEdgesDF['Gene1'].astype(str) == key.split('|')[1]))]) > 0:
        TrueEdgeDict[key] = 1

    # Compute PredEdgeDict Dictionary
    # from predEdgeDF

for key in PredEdgeDict.keys():
    if len(predEdgeDF.loc[((predEdgeDF['Gene1'].astype(str) == key.split('|')[0]) &
                            (predEdgeDF['Gene2'].astype(str) == key.split('|')[1])) |
                           ((predEdgeDF['Gene2'].astype(str) == key.split('|')[0]) &
                            (predEdgeDF['Gene1'].astype(str) == key.split('|')[1]))]) > 0:
        PredEdgeDict[key] = 1

# Combine into one dataframe
# to pass it to sklearn
outDF = pd.DataFrame([TrueEdgeDict, PredEdgeDict]).T
outDF.columns = ['TrueEdges', 'PredEdges']

fpr, tpr, thresholds = roc_curve(y_true=outDF['TrueEdges'],
                                 y_score=outDF['PredEdges'], pos_label=1)

prec, recall, thresholds = precision_recall_curve(y_true=outDF['TrueEdges'],
                                                  probas_pred=outDF['PredEdges'], pos_label=1)

print("fpr:",fpr,"tpr:",tpr,"thresholds:",thresholds)
print("prec:",prec,"recall:",recall,"thresholds:",thresholds)
with open(output_path + "human_hESC_roc_curve_values.txt", "w") as file:
    file.write("fpr: " + str(fpr) + "\n")
    file.write("tpr: " + str(tpr) + "\n")
    file.write("thresholds: " + str(thresholds) + "\n")
    file.write("prec: " + str(prec) + "\n")
    file.write("recall: " + str(recall) + "\n")