
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# Nature Communications Style Configuration
# =========================================================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'Liberation Sans'],
    'font.size': 7.5,
    'axes.titlesize': 9,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.size': 2.5,
    'ytick.major.size': 2.5,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'lines.linewidth': 1.8,
})

# =========================================================
# Fusion Evolution Stages
# =========================================================
fusion_stages = [
    'Add',
    'Mean',
    'Concat\n+MLP',
    'Attn\n+MLP',
    'KSAF\n(Ours)'
]

x = np.arange(len(fusion_stages))

# =========================================================
# Example Averaged Results
# Replace with your real averaged values
# =========================================================

# Classification (higher is better)
cls_scores = [
    0.843,
    0.838,
    0.851,
    0.862,
    0.884
]

# Regression (lower is better)
reg_scores = [
    1.009,
    0.998,
    0.934,
    0.891,
    0.764
]

# =========================================================
# Colors
# =========================================================
cls_color = '#3B6FB6'
reg_color = '#D65F5F'

# =========================================================
# Create Figure
# =========================================================
fig, (ax1, ax2) = plt.subplots(
    1, 2,
    figsize=(7.1, 2.8)
)

fig.subplots_adjust(
    wspace=0.28,
    left=0.08,
    right=0.98,
    bottom=0.22,
    top=0.88
)

# =========================================================
# (a) Classification Evolution
# =========================================================
ax1.plot(
    x,
    cls_scores,
    marker='o',
    markersize=5.5,
    color=cls_color
)

# Highlight final point
ax1.scatter(
    x[-1],
    cls_scores[-1],
    s=55,
    color='#C23B22',
    zorder=5
)

# Progressive arrows
for i in range(len(x)-1):
    ax1.annotate(
        '',
        xy=(x[i+1], cls_scores[i+1]),
        xytext=(x[i], cls_scores[i]),
        arrowprops=dict(
            arrowstyle='->',
            lw=1.0,
            color='#808080',
            alpha=0.7
        )
    )

ax1.set_xticks(x)
ax1.set_xticklabels(fusion_stages)

ax1.set_ylabel('Average ROC-AUC')
ax1.set_title('a', loc='left', fontweight='bold')

ax1.grid(
    axis='y',
    linestyle='--',
    linewidth=0.5,
    alpha=0.5
)

ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

ax1.set_ylim(0.82, 0.90)

# =========================================================
# (b) Regression Evolution
# =========================================================
ax2.plot(
    x,
    reg_scores,
    marker='o',
    markersize=5.5,
    color=reg_color
)

# Highlight final point
ax2.scatter(
    x[-1],
    reg_scores[-1],
    s=55,
    color='#C23B22',
    zorder=5
)

# Progressive arrows
for i in range(len(x)-1):
    ax2.annotate(
        '',
        xy=(x[i+1], reg_scores[i+1]),
        xytext=(x[i], reg_scores[i]),
        arrowprops=dict(
            arrowstyle='->',
            lw=1.0,
            color='#808080',
            alpha=0.7
        )
    )

ax2.set_xticks(x)
ax2.set_xticklabels(fusion_stages)

ax2.set_ylabel('Average RMSE')
ax2.set_title('b', loc='left', fontweight='bold')

ax2.grid(
    axis='y',
    linestyle='--',
    linewidth=0.5,
    alpha=0.5
)

ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Lower is better
ax2.invert_yaxis()

# =========================================================
# Shared Figure Caption Style
# =========================================================
fig.suptitle(
    'Progressive Performance Evolution of Local-Global Fusion Architectures',
    fontsize=9,
    y=0.98
)

# =========================================================
# Save
# =========================================================
plt.savefig(
    'Fusion_Evolution_NC_Style.pdf',
    format='pdf',
    bbox_inches='tight',
    pad_inches=0.02
)

plt.savefig(
    'Fusion_Evolution_NC_Style.png',
    format='png',
    bbox_inches='tight',
    pad_inches=0.02
)

plt.show()

