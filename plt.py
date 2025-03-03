import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import matplotlib
matplotlib.use('Agg')


# 假设你有保存的真实标签和模型预测的概率数据
labels_all = [0, 1, 0, 1, 0, 1]  # 示例的真实标签数据
preds_all = [0.1, 0.3, 0.2, 0.8, 0.6, 0.9]  # 示例的模型预测的概率数据

# 计算 ROC 曲线上的 FPR 和 TPR
fpr, tpr, _ = roc_curve(labels_all, preds_all)

# 计算 ROC 曲线下的面积
roc_auc = auc(fpr, tpr)

# 打印 AUC 值
print("AUC:", roc_auc)

# 绘制 ROC 曲线
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
plt.savefig('roc_curve.png')