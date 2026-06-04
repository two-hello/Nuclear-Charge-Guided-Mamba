import matplotlib.pyplot as plt
import numpy as np

# 设置字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 数据准备
class_datasets = ['BBBP', 'BACE', 'HIV', 'MUV', 'TOX21', 'TOXCAST', 'SIDER']
regression_datasets = ['LIPO', 'FREESOLV', 'ESOL']

atomic_number = [0.9622, 0.953, 0.8516, 0.8626, 0.8722, 0.7724, 0.6904]
atom_degree = [0.9464, 0.9546, 0.8484, 0.8168, 0.8572, 0.7682, 0.6522]
random_order = [0.9434, 0.9454, 0.8328, 0.8034, 0.858, 0.7684, 0.6422]

reg_atomic_number = [0.6582, 0.8202, 0.8824]
reg_atom_degree = [0.6944, 0.9984, 0.9666]
reg_random_order = [0.7152, 1.2406, 1.0738]

# 三组颜色
colors = [
    '#1E6EC7',
    '#3AAEA9',
    '#A4D8C1'
]

# 图大小（适合 LaTeX 一行放两个）
figsize = (10, 6)

width = 0.25
x = np.arange(len(class_datasets))
x_reg = np.arange(len(regression_datasets))

# =============================
#   分类图
# =============================
fig1, ax1 = plt.subplots(figsize=figsize)

bars1 = ax1.bar(x - width, atomic_number, width, label='Nuclear Charge',
                color=colors[0], edgecolor='black', linewidth=1.5)
bars2 = ax1.bar(x, atom_degree, width, label='Degree',
                color=colors[1], edgecolor='black', linewidth=1.2)
bars3 = ax1.bar(x + width, random_order, width, label='Random',
                color=colors[2], edgecolor='black', linewidth=1.2)

ax1.set_xlabel('Datasets', fontsize=18, fontweight='bold')
ax1.set_ylabel('ROC-AUC', fontsize=18, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(class_datasets, fontsize=14)
ax1.tick_params(axis='y', labelsize=14)

ax1.set_ylim(0.6, 1.0)

# 图例更大
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=3,
           fontsize=16, frameon=True, framealpha=0.95, edgecolor='black')

# 背景美化
ax1.set_facecolor('#F8F9FA')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.grid(axis='y', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig('classification_sorting_methods.png', dpi=400, bbox_inches='tight', facecolor='white')
plt.savefig('classification_sorting_methods.pdf', bbox_inches='tight', facecolor='white')
plt.show()


# =============================
#   回归图
# =============================
fig2, ax2 = plt.subplots(figsize=figsize)

bars4 = ax2.bar(x_reg - width, reg_atomic_number, width, label='Nuclear Charge',
                color=colors[0], edgecolor='black', linewidth=1.5)
bars5 = ax2.bar(x_reg, reg_atom_degree, width, label='Degree',
                color=colors[1], edgecolor='black', linewidth=1.2)
bars6 = ax2.bar(x_reg + width, reg_random_order, width, label='Random',
                color=colors[2], edgecolor='black', linewidth=1.2)

ax2.set_xlabel('Datasets', fontsize=18, fontweight='bold')
ax2.set_ylabel('RMSE', fontsize=18, fontweight='bold')
ax2.set_xticks(x_reg)
ax2.set_xticklabels(regression_datasets, fontsize=14)
ax2.tick_params(axis='y', labelsize=14)

ax2.set_ylim(0.6, 1.3)

ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=3,
           fontsize=16, frameon=True, framealpha=0.95, edgecolor='black')

ax2.set_facecolor('#F8F9FA')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.grid(axis='y', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig('regression_sorting_methods.png', dpi=400, bbox_inches='tight', facecolor='white')
plt.savefig('regression_sorting_methods.pdf', bbox_inches='tight', facecolor='white')
plt.show()
