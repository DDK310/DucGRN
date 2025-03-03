import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# 创建示例数据
data = {
    'Method': ['DeepRIG', 'DeepSEM', '3DCEMA', 'CNNC', 'SCRIBE', 'SCODE', 'PPCOR', 'PIDC', 'LEAP', 'GRNBOOST2'],
    'AUROC_LI': [0.85, 0.76, 0.80, 0.78, 0.82, 0.73, 0.81, 0.79, 0.83, 0.75],
    'AUROC_LL': [0.87, 0.77, 0.81, 0.79, 0.84, 0.74, 0.83, 0.80, 0.85, 0.76],
    'AUPRC_LI': [0.65, 0.55, 0.60, 0.58, 0.62, 0.53, 0.61, 0.59, 0.63, 0.54],
    'AUPRC_LL': [0.67, 0.57, 0.62, 0.60, 0.64, 0.55, 0.63, 0.61, 0.65, 0.56],
    'Early_Precision_Ratio_LI': [2.6, 1.3, 2.6, 1.3, 2.6, 1.7, 3.0, 3.0, 2.6, 2.6],
    'Early_Precision_Ratio_LL': [2.5, 7.6, 6.0, 4.3, 6.0, 4.1, 2.4, 3.0, 6.9, 5.9],
    'AUPRC_Ratio_LI': [4.4, 1.8, 3.5, 1.2, 3.0, 2.7, 3.7, 2.8, 4.4, 4.5],
    'AUPRC_Ratio_LL': [4.5, 3.0, 12.5, 1.3, 1.9, 2.7, 1.6, 1.3, 2.3, 1.9],
}

df = pd.DataFrame(data)

# 设置图表风格
sns.set(style="whitegrid")

# 创建一个大的绘图区域
fig, ax = plt.subplots(figsize=(20, 12))

# 定义方框和圆圈的颜色
cmap = plt.cm.YlOrRd
norm = plt.Normalize(df.iloc[:, 1:].values.min(), df.iloc[:, 1:].values.max())

# 绘制自定义的热图
methods = df['Method']
metrics = df.columns[1:]

# 创建网格位置
x, y = np.meshgrid(np.arange(len(metrics)), np.arange(len(methods)))

# 展开网格位置
x = x.flatten()
y = y.flatten()

# 获取数据值并标准化大小
values = df[metrics].values.flatten()
sizes = (values - np.min(values)) / (np.max(values) - np.min(values)) * 3000  # 调整大小比例，使图形更小
colors = values

# 绘制方框和圆圈
for (i, j, val, size) in zip(x, y, colors, sizes):
    if 'AUROC' in metrics[i]:
        ax.add_patch(plt.Rectangle((i, j), 4, 4, fill=True, color=cmap(norm(val)), edgecolor='black'))
        ax.scatter(i + 0.5, j + 0.5, s=size, color='blue', edgecolor='black', alpha=0.5)
    else:
        ax.scatter(i + 0.5, j + 0.5, s=size, color=cmap(norm(val)), edgecolor='black', alpha=0.5)

# 设置标签和间隔
ax.set_xticks(np.arange(len(metrics)) + 0.5)
ax.set_xticklabels(metrics, rotation=45, ha='right', fontsize=12)  # 调整字体大小
ax.set_yticks(np.arange(len(methods)) + 0.5)
ax.set_yticklabels(methods, fontsize=12)  # 调整字体大小
ax.set_xlim(0, len(metrics))
ax.set_ylim(0, len(methods))

# 添加标题和标签
ax.set_title('Performance Comparison of Methods', fontsize=16)
ax.set_xlabel('Metrics', fontsize=14)
ax.set_ylabel('Methods', fontsize=14)

plt.tight_layout()

# 保存图表为图片文件
plt.savefig('performance_comparison_custom.png')

# 显示图表
plt.show()
