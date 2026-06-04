import matplotlib.pyplot as plt
import numpy as np

# =======================
# Nature style
# =======================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    'font.size': 7.5,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 6.8,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'figure.dpi': 300,
})

# =======================
# Data
# =======================
x = np.array([5, 10, 20])

smiles_mamba = [0.2583, 0.2258, 0.2080]
gps128 = [0.1413, 0.1311, 0.1188]
gps200 = [0.1354, 0.1275, 0.1160]
ours128 = [0.1329, 0.1268, 0.1165]

# =======================
# Broken axis figure
# =======================
fig, (ax_top, ax_bottom) = plt.subplots(
    2, 1,
    sharex=True,
    figsize=(3.5, 3.2),
    gridspec_kw={'height_ratios': [1, 2]}
)

# =======================
# Plot all curves
# =======================
for ax in [ax_top, ax_bottom]:

    ax.plot(
        x, smiles_mamba,
        marker='o',
        linewidth=1.2,
        markersize=4,
        label='SMILES-Mamba (2.7M)'
    )

    ax.plot(
        x, gps128,
        marker='s',
        linewidth=1.2,
        markersize=4,
        label='GPS-128 (0.67M)'
    )

    ax.plot(
        x, gps200,
        marker='^',
        linewidth=1.2,
        markersize=4,
        label='GPS-200 (1.62M)'
    )

    ax.plot(
        x, ours128,
        marker='D',
        linewidth=1.4,
        markersize=4.2,
        label='KAN-NC-Mamba (1.44M)'
    )

# =======================
# Axis ranges
# =======================

# Top axis: only SMILES-Mamba
ax_top.set_ylim(0.19, 0.27)

# Bottom axis: focus region
ax_bottom.set_ylim(0.112, 0.145)

# =======================
# Styling
# =======================
for ax in [ax_top, ax_bottom]:

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.grid(
        True,
        linestyle='--',
        linewidth=0.4,
        alpha=0.5
    )

# Hide connecting spine
ax_top.spines['bottom'].set_visible(False)
ax_bottom.spines['top'].set_visible(False)

ax_top.tick_params(labeltop=False)
ax_bottom.xaxis.tick_bottom()

# =======================
# Diagonal break marks
# =======================
d = .008

kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False, linewidth=0.6)
ax_top.plot((-d, +d), (-d, +d), **kwargs)
ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)

kwargs.update(transform=ax_bottom.transAxes)
ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)
ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

# =======================
# Labels
# =======================
ax_bottom.set_xlabel('Training subset ratio (%)')
ax_bottom.set_ylabel('Validation MAE ↓')

ax_bottom.set_xticks([5, 10, 20])

# =======================
# Legend
# =======================
legend = ax_top.legend(
    frameon=True,
    loc='upper right',
    borderpad=0.3
)

legend.get_frame().set_linewidth(0.5)

# =======================
# Layout
# =======================
plt.tight_layout()

# =======================
# Save
# =======================
plt.savefig(
    'pcqm_scaling_broken_axis.pdf',
    bbox_inches='tight',
    pad_inches=0.02
)

plt.savefig(
    'pcqm_scaling_broken_axis.png',
    bbox_inches='tight',
    pad_inches=0.02
)

plt.show()