import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# ==================== Nature Communications Style ====================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'Liberation Sans'],
    'font.size': 7.5,
    'axes.titlesize': 8.5,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 6.8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.linewidth': 0.5,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size': 2.5,
    'ytick.major.size': 2.5,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'lines.linewidth': 0.8,
    'legend.frameon': True,
    'legend.edgecolor': '#d9d9d9',
    'legend.framealpha': 0.9,
})

# ==================== Parameter Mapping ====================
param_dict = {
    'MLP-0.5x': 66048,
    'MLP-1x': 123648,
    'KAN': 131584,
    'MLP-2x': 267648
}

base_param = param_dict['MLP-0.5x']
base_area = 55


def bubble_area(p):
    return base_area * (p / base_param)


# ==================== Datasets ====================
cls_tasks = ['BACE', 'BBBP', 'HIV', 'Tox21', 'SIDER', 'ToxCast', 'ClinTox']
reg_tasks = ['FreeSolv', 'LIPO', 'ESOL']

# ==================== Results ====================
mlp05_c = [0.958667, 0.955, 0.820, 0.853, 0.641333, 0.762333, 0.949333]
mlp1_c = [0.957667, 0.955333, 0.827, 0.856333, 0.654333, 0.763667, 0.939333]
mlp2_c = [0.954, 0.948333, 0.832, 0.857333, 0.660, 0.756667, 0.947333]
kan_c = [0.970, 0.958, 0.861, 0.871, 0.696, 0.773, 0.961]

mlp05_r = [1.024, 0.692333, 0.961667]
mlp1_r = [0.981333, 0.696333, 2.436333]
mlp2_r = [0.965, 0.722, 2.426333]
kan_r = [0.779, 0.668, 0.845]

# ==================== Layout ====================
offsets = {
    'MLP-0.5x': -0.21,
    'MLP-1x': -0.07,
    'KAN': 0.07,
    'MLP-2x': 0.21
}

# ==================== Colors ====================
model_colors = {
    'MLP-0.5x': '#8FB9C9',
    'MLP-1x': '#4C84A6',
    'MLP-2x': '#1F4E6B',
    'KAN': '#C44E52'
}


# ==================== Draw Function ====================
def draw_panel(ax, tasks, models_data, ylabel, title_str, show_legend=False):
    x_pos = np.arange(len(tasks))

    for model_name, data in models_data.items():
        ax.scatter(
            x_pos + offsets[model_name],
            data['mean'],
            s=data['area'],
            color=model_colors[model_name],
            edgecolors='white',
            linewidth=0.45,
            alpha=0.92,
            zorder=3
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(tasks, rotation=20, ha='right')

    ax.set_ylabel(ylabel, labelpad=3)
    ax.set_title(title_str, loc='left', fontweight='bold', pad=6)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.grid(
        axis='y',
        linestyle='--',
        linewidth=0.4,
        color='#e6e6e6',
        zorder=1
    )

    if show_legend:

        handles = []

        for model_name in ['MLP-0.5x', 'MLP-1x', 'MLP-2x', 'KAN']:
            ms = np.sqrt(bubble_area(param_dict[model_name])) * 0.55

            handles.append(
                Line2D(
                    [0], [0],
                    marker='o',
                    color='w',
                    markerfacecolor=model_colors[model_name],
                    markeredgecolor='white',
                    markeredgewidth=0.4,
                    markersize=ms,
                    label=model_name
                )
            )

        ax.legend(
            handles=handles,
            loc='lower right',
            frameon=True,
            borderpad=0.4,
            labelspacing=0.5,
            handletextpad=0.5
        )


# ==================== Assemble ====================
cls_models = {
    'MLP-0.5x': {'mean': mlp05_c, 'area': bubble_area(param_dict['MLP-0.5x'])},
    'MLP-1x': {'mean': mlp1_c, 'area': bubble_area(param_dict['MLP-1x'])},
    'MLP-2x': {'mean': mlp2_c, 'area': bubble_area(param_dict['MLP-2x'])},
    'KAN': {'mean': kan_c, 'area': bubble_area(param_dict['KAN'])},
}

reg_models = {
    'MLP-0.5x': {'mean': mlp05_r, 'area': bubble_area(param_dict['MLP-0.5x'])},
    'MLP-1x': {'mean': mlp1_r, 'area': bubble_area(param_dict['MLP-1x'])},
    'MLP-2x': {'mean': mlp2_r, 'area': bubble_area(param_dict['MLP-2x'])},
    'KAN': {'mean': kan_r, 'area': bubble_area(param_dict['KAN'])},
}

# ==================== Plot ====================
fig, (ax1, ax2) = plt.subplots(
    1, 2,
    figsize=(7.1, 2.9)
)

fig.subplots_adjust(
    wspace=0.28,
    bottom=0.20,
    top=0.88,
    left=0.08,
    right=0.98
)

draw_panel(
    ax1,
    cls_tasks,
    cls_models,
    'ROC-AUC ↑',
    '(a) Classification Tasks',
    show_legend=True
)

ax1.set_ylim(0.60, 1.01)

draw_panel(
    ax2,
    reg_tasks,
    reg_models,
    'RMSE ↓',
    '(b) Regression Tasks',
    show_legend=False
)

ax2.set_ylim(0.40, 2.70)

# ==================== Save ====================
plt.savefig(
    'KAN_vs_MLP_ParameterEfficiency.pdf',
    format='pdf',
    bbox_inches='tight',
    pad_inches=0.02
)

plt.savefig(
    'KAN_vs_MLP_ParameterEfficiency.png',
    format='png',
    bbox_inches='tight',
    pad_inches=0.02
)

plt.show()
