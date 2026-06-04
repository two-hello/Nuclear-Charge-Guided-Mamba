import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm

# 直接使用SimHei字体（如果可用），否则使用默认字体
try:
    simhei_path = '/root/autodl-tmp/ht/Graph-Mamba-main/fonts/simhei.ttf'
    fm.fontManager.addfont(simhei_path)
    font_prop = fm.FontProperties(fname=simhei_path)
    font_name = font_prop.get_name()
    plt.rcParams['font.family'] = font_name
    plt.rcParams['axes.unicode_minus'] = False
    print(f"Successfully set SimHei font: {font_name}")
except Exception as e:
    print(f"Failed to set SimHei font: {e}, using default English font")
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

# 数据准备
class_datasets = ['BBBP', 'BACE', 'HIV', 'CLINTOX', 'TOX21', 'TOXCAST', 'SIDER']
regression_datasets = ['LIPO', 'FREESOLV', 'ESOL']

# 分类任务的ROC-AUC值（原始范围0-1）
atomic_number = [0.9582, 0.9704, 0.8608, 0.961, 0.871, 0.7732, 0.696]
atom_degree = [0.9464, 0.9546, 0.8366, 0.9496, 0.8572, 0.7726, 0.6522]
random_order = [0.9434, 0.9454, 0.8374, 0.9376, 0.858, 0.7708, 0.6574]

reg_atomic_number = [0.6688, 0.7794, 0.8454]
reg_atom_degree = [0.6944, 0.9984, 0.9666]
reg_random_order = [0.7152, 1.2406, 1.0738]

# 蓝色向浅绿色过渡的颜色方案
colors = [
    '#1E6EC7',  # 深蓝色
    '#3AAEA9',  # 蓝绿色过渡
    '#A4D8C1'  # 浅绿色
]

# 设置柱状图位置和宽度
x = np.arange(len(class_datasets))
x_reg = np.arange(len(regression_datasets))
width = 0.25

# 创建分类数据集图表
fig1, ax1 = plt.subplots(figsize=(16, 8))

# 绘制分类数据集柱状图 - 使用英文图例
bars1 = ax1.bar(x - width, atomic_number, width, label='Nuclear Charge',
                color=colors[0], edgecolor='black', linewidth=2.0, zorder=3)
bars2 = ax1.bar(x, atom_degree, width, label='Degree',
                color=colors[1], edgecolor='black', linewidth=0.5, zorder=2)
bars3 = ax1.bar(x + width, random_order, width, label='Random',
                color=colors[2], edgecolor='black', linewidth=0.5, zorder=2)

ax1.set_xlabel('Datasets', fontsize=16, fontweight='bold')
ax1.set_ylabel('ROC-AUC', fontsize=16, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(class_datasets, fontsize=14)
ax1.tick_params(axis='y', labelsize=14)

# 设置y轴范围为0.60到1.00，刻度间隔0.05
ax1.set_ylim(0.60, 1.00)
ax1.set_yticks(np.arange(0.60, 1.01, 0.05))

# 添加图例
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3,
           fontsize=16, frameon=True, framealpha=0.9, edgecolor='black')

# 美化背景
ax1.set_facecolor('#F8F9FA')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_color('#DDDDDD')
ax1.spines['bottom'].set_color('#DDDDDD')

# 添加轻微的网格线以提高可读性
ax1.grid(axis='y', linestyle='--', alpha=0.3, linewidth=0.5)

# 调整布局
plt.tight_layout()

# 保存分类图表
plt.savefig('classification_sorting_methods.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('classification_sorting_methods.pdf', bbox_inches='tight', facecolor='white')

# 显示分类图表
plt.show()

# 创建回归数据集图表
fig2, ax2 = plt.subplots(figsize=(14, 8))

# 绘制回归数据集柱状图 - 使用英文图例
bars4 = ax2.bar(x_reg - width, reg_atomic_number, width, label='Nuclear Charge',
                color=colors[0], edgecolor='black', linewidth=2.0, zorder=3)
bars5 = ax2.bar(x_reg, reg_atom_degree, width, label='Degree',
                color=colors[1], edgecolor='black', linewidth=0.5, zorder=2)
bars6 = ax2.bar(x_reg + width, reg_random_order, width, label='Random',
                color=colors[2], edgecolor='black', linewidth=0.5, zorder=2)

ax2.set_xlabel('Datasets', fontsize=16, fontweight='bold')
ax2.set_ylabel('RMSE', fontsize=16, fontweight='bold')
ax2.set_xticks(x_reg)
ax2.set_xticklabels(regression_datasets, fontsize=14)
ax2.tick_params(axis='y', labelsize=14)

# 设置y轴范围（回归不变）
ax2.set_ylim(0.6, 1.3)

# 添加图例
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3,
           fontsize=16, frameon=True, framealpha=0.9, edgecolor='black')

# 美化背景
ax2.set_facecolor('#F8F9FA')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_color('#DDDDDD')
ax2.spines['bottom'].set_color('#DDDDDD')

# 添加轻微的网格线以提高可读性
ax2.grid(axis='y', linestyle='--', alpha=0.3, linewidth=0.5)

# 调整布局
plt.tight_layout()

# 保存回归图表
plt.savefig('regression_sorting_methods.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('regression_sorting_methods.pdf', bbox_inches='tight', facecolor='white')

# 显示回归图表
plt.show()

print("Classification plot saved as 'classification_sorting_methods.png' and 'classification_sorting_methods.pdf'")
print("Regression plot saved as 'regression_sorting_methods.png' and 'regression_sorting_methods.pdf'")