import matplotlib.pyplot as plt
import numpy as np

# ======================================================
# Nature Communications Style
# ======================================================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    'font.size': 7.5,
    'axes.labelsize': 8,
    'axes.titlesize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'figure.dpi': 300,
    'savefig.dpi': 300,
})

# ======================================================
# Methods
# ======================================================
methods = [
    'Concat\n+MLP',
    'Concat\n+KAN',
    'Attn\n+MLP',
    'KDM\n(Ours)'
]

# ======================================================
# Original Results
# ======================================================

# Classification datasets:
# BACE BBBP HIV Tox21 SIDER ClinTox ToxCast

cls_concat_mlp = [95.03, 95.73, 82.50, 85.87, 65.47, 91.87, 75.67]
cls_concat_kan = [95.77, 94.73, 82.37, 86.10, 66.53, 93.67, 76.73]
cls_attn_mlp   = [96.17, 94.83, 83.03, 85.77, 64.80, 89.07, 76.30]
cls_kdm        = [97.00, 95.80, 86.10, 87.10, 69.60, 96.10, 77.30]

# Regression datasets:
# FreeSolv Lipophilicity ESOL

reg_concat_mlp = [1.646, 0.716, 1.226]
reg_concat_kan = [1.011, 0.714, 0.886]
reg_attn_mlp   = [2.347, 0.689, 3.250]
reg_kdm        = [0.779, 0.668, 0.845]

# ======================================================
# Average Performance
# ======================================================

cls_avg = [
    np.mean(cls_concat_mlp),
    np.mean(cls_concat_kan),
    np.mean(cls_attn_mlp),
    np.mean(cls_kdm),
]

reg_avg = [
    np.mean(reg_concat_mlp),
    np.mean(reg_concat_kan),
    np.mean(reg_attn_mlp),
    np.mean(reg_kdm),
]

# ======================================================
# Figure
# ======================================================
fig, axes = plt.subplots(
    1, 2,
    figsize=(5.6, 2.6)
)

# ======================================================
# Colors (NC-style muted palette)
# ======================================================
colors = [
    '#B8B8B8',
    '#8FA8C9',
    '#6F6F6F',
    '#2F5D9B'
]

# ======================================================
# (a) Classification
# ======================================================
ax = axes[0]

x = np.arange(len(methods))

bars = ax.bar(
    x,
    cls_avg,
    width=0.62,
    color=colors,
    edgecolor='black',
    linewidth=0.35
)

ax.set_xticks(x)
ax.set_xticklabels(methods)

ax.set_ylabel('Average ROC-AUC (%) ↑')
ax.set_title('(a) Classification benchmarks')

# 自动设置y轴范围（更有层次感）
ax.set_ylim(80, 90)

# 数值标注
for bar, val in zip(bars, cls_avg):
    ax.text(
        bar.get_x() + bar.get_width()/2,
        val + 0.15,
        f'{val:.2f}',
        ha='center',
        va='bottom',
        fontsize=6.5
    )

# NC-style
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.grid(
    axis='y',
    linestyle='--',
    linewidth=0.4,
    alpha=0.6
)

# ======================================================
# (b) Regression
# ======================================================
ax = axes[1]

bars = ax.bar(
    x,
    reg_avg,
    width=0.62,
    color=colors,
    edgecolor='black',
    linewidth=0.35
)

ax.set_xticks(x)
ax.set_xticklabels(methods)

ax.set_ylabel('Average RMSE ↓')
ax.set_title('(b) Regression benchmarks')

# lower is better
ax.set_ylim(0.5, 2.2)

# 数值标注
for bar, val in zip(bars, reg_avg):
    ax.text(
        bar.get_x() + bar.get_width()/2,
        val + 0.04,
        f'{val:.3f}',
        ha='center',
        va='bottom',
        fontsize=6.5
    )

# NC-style
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.grid(
    axis='y',
    linestyle='--',
    linewidth=0.4,
    alpha=0.6
)

# ======================================================
# Tight layout
# ======================================================
plt.tight_layout()

# ======================================================
# Save
# ======================================================
plt.savefig(
    'fusion_ablation_summary_nc_style.pdf',
    bbox_inches='tight',
    pad_inches=0.02
)

plt.savefig(
    'fusion_ablation_summary_nc_style.png',
    bbox_inches='tight',
    pad_inches=0.02
)

plt.show()