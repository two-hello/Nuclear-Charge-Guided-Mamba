import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# =========================================================
# Atomic Semantic Perturbation Analysis
# Nature Communications Style Visualization
# =========================================================

# -----------------------------
# 1. Experimental Results
# -----------------------------
datasets = [
    'BACE', 'BBBP', 'HIV', 'Tox21', 'ToxCast',
    'SIDER', 'ClinTox', 'ESOL', 'LIPO', 'FreeSolv'
]

nuclear_charge = [13.85, 28.17, 35.82, 24.50, 18.23,
                  4.08, 10.78, 50.45, 102.44, 82.29]

degree_order = [6.63, 14.08, 12.42, 15.33, 11.61,
                2.78, 1.98, 19.34, 80.60, 64.86]

random_order = [-8.27, 3.42, 33.75, 11.18, 10.71,
                3.12, -5.10, 18.20, 68.17, 63.71]

# -----------------------------
# 2. Plot Configuration
# -----------------------------
x = np.arange(len(datasets))
width = 0.24

sns.set_theme(style="whitegrid")

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'axes.linewidth': 1.0,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300
})

fig, ax = plt.subplots(figsize=(13, 6.5))

# -----------------------------
# 3. Highlight Polarity-Sensitive Datasets
# -----------------------------
# Highlight ESOL / LIPO / FreeSolv region
ax.axvspan(6.5, 9.5, alpha=0.08, color='#D4A373')

# -----------------------------
# 4. Scientific Color Palette
# -----------------------------
color_nc = '#355C7D'      # deep muted blue
color_degree = '#C06C84' # muted rose
color_random = '#6C757D' # neutral gray

# -----------------------------
# 5. Draw Bars
# -----------------------------
bars1 = ax.bar(
    x - width,
    nuclear_charge,
    width,
    label='Nuclear-charge-guided',
    color=color_nc,
    edgecolor='black',
    linewidth=0.6
)

bars2 = ax.bar(
    x,
    degree_order,
    width,
    label='Degree-based',
    color=color_degree,
    edgecolor='black',
    linewidth=0.6
)

bars3 = ax.bar(
    x + width,
    random_order,
    width,
    label='Random',
    color=color_random,
    edgecolor='black',
    linewidth=0.6
)

# -----------------------------
# 6. Baseline Line
# -----------------------------
ax.axhline(
    0,
    color='black',
    linewidth=0.9
)

# -----------------------------
# 7. Axis Labels and Title
# -----------------------------
ax.set_ylabel(
    'Relative Performance Degradation (%)',
    fontweight='bold'
)

ax.set_xlabel(
    'Molecular Property Prediction Datasets',
    fontweight='bold',
    labelpad=10
)

ax.set_title(
    'Sensitivity to Atomic-Semantic Perturbation under Different Node Ordering Strategies',
    fontweight='bold',
    pad=16
)

# -----------------------------
# 8. Tick Labels
# -----------------------------
ax.set_xticks(x)
ax.set_xticklabels(
    datasets,
    rotation=0,
    fontweight='semibold'
)

# -----------------------------
# 9. Annotate ESOL / FreeSolv
# -----------------------------
ax.text(
    7,
    108,
    'Polarity-sensitive tasks',
    ha='center',
    fontsize=10,
    fontstyle='italic'
)

# -----------------------------
# 10. Grid
# -----------------------------
ax.grid(
    axis='y',
    linestyle='--',
    alpha=0.4
)

ax.grid(
    axis='x',
    visible=False
)

# -----------------------------
# 11. Legend
# -----------------------------
legend = ax.legend(
    frameon=True,
    fancybox=True,
    shadow=False,
    loc='upper left'
)

legend.get_frame().set_alpha(0.95)

# -----------------------------
# 12. Remove Top/Right Border
# -----------------------------
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# -----------------------------
# 13. Tight Layout
# -----------------------------
plt.tight_layout()

# -----------------------------
# 14. Save Figure
# -----------------------------
plt.savefig(
    'atomic_semantic_perturbation_analysis.png',
    dpi=600,
    bbox_inches='tight'
)

plt.savefig(
    'atomic_semantic_perturbation_analysis.pdf',
    bbox_inches='tight'
)

# -----------------------------
# 15. Show Figure
# -----------------------------
plt.show()