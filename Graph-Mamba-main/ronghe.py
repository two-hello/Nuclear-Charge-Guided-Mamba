import numpy as np
import matplotlib.pyplot as plt
from math import pi

# 数据准备
categories_classification = ['BBBP', 'BACE', 'HIV', 'ClinTox', 'TOX21', 'TOXCAST', 'SIDER']
categories_regression = ['LIPO', 'FREESOLV', 'ESOL']

original_classification = [0.9582, 0.9704, 0.8608, 0.961, 0.871, 0.7732, 0.696]
add_classification = [0.9488, 0.8976, 0.8392, 0.9508, 0.8372, 0.7668, 0.6626]
concat_classification = [0.949, 0.9046, 0.8398, 0.9448, 0.8682, 0.7692, 0.6306]
mean_classification = [0.9386, 0.8816, 0.8378, 0.9462, 0.8368, 0.7672, 0.6614]

original_regression = [0.6688, 0.7794, 0.8454]
add_regression = [0.6724, 1.2422, 1.1114]
concat_regression = [0.687, 1.1278, 0.9284]
mean_regression = [0.6702, 1.2348, 1.1104]

# 使用更专业的配色方案 - 来自Tableau 10调色板
colors = ['#4E79A7', '#F28E2B', '#59A14F', '#E15759']

# 创建高级多边形网格雷达图函数
def create_advanced_polygon_radar_chart(categories, values_list, labels, title, fig, position, rlim_min=0.6,
                                        rlim_max=1.0):
    # 计算角度 - 根据数据集数量自动调整
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]  # 闭合图形

    # 初始化极坐标子图
    ax = fig.add_subplot(1, 2, position, polar=True)

    # 设置第一个标签在顶部
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # 根据数据集数量调整标签大小
    if N == 3:  # 三角形
        label_size = 12
    elif N == 7:  # 七边形
        label_size = 10
    else:
        label_size = 9

    plt.xticks(angles[:-1], categories, color='#333333', size=label_size, fontweight='bold')

    # 设置y轴范围
    plt.ylim(rlim_min, rlim_max)

    # 完全关闭默认网格和边框
    ax.grid(False)
    ax.spines['polar'].set_visible(False)

    # 移除径向标签
    ax.set_yticklabels([])

    # 创建多边形网格
    radial_ticks = np.linspace(rlim_min, rlim_max, 5)

    # 绘制同心多边形
    for i, tick in enumerate(radial_ticks):
        polygon_radii = [tick] * (N + 1)
        line_style = '-' if i == len(radial_ticks) - 1 else '--'
        line_width = 1.5 if i == len(radial_ticks) - 1 else 1.0
        line_color = '#666666' if i == len(radial_ticks) - 1 else '#999999'
        ax.plot(angles, polygon_radii, line_style, color=line_color, alpha=0.8, linewidth=line_width)

    # 添加径向标签
    for tick in radial_ticks:
        ax.text(0, tick, f'{tick:.1f}', ha='center', va='center', fontsize=10,
                color='#444444', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8, edgecolor='none'))

    # 绘制从中心到顶点的线
    for angle in angles[:-1]:
        ax.plot([angle, angle], [0, rlim_max], '--', color='#999999', alpha=0.6, linewidth=1.0)

    # 绘制每个方法的数据
    for i, (values, label) in enumerate(zip(values_list, labels)):
        values += values[:1]  # 闭合图形

        # 使用不同的线条样式增加区分度
        line_styles = ['-', '--', '-.', ':']

        ax.plot(angles, values, linewidth=3.0, linestyle=line_styles[i],
                label=label, color=colors[i], marker='o', markersize=7,
                markeredgecolor='white', markeredgewidth=1.5)
        ax.fill(angles, values, alpha=0.15, color=colors[i])

    # 添加标题 - 移除了形状名称
    plt.title(f'{title}', size=15, color='#222222', y=1.1, fontweight='bold')

    # 添加图例 - 更加专业的样式
    legend = ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.0) if N == 7 else (1.35, 0.9),
                       fontsize=11, frameon=True, fancybox=False, shadow=False,
                       framealpha=0.95, edgecolor='#DDDDDD', facecolor='#F8F8F8',
                       ncol=1, labelspacing=1.0, handlelength=2.0)

    # 设置图例边框样式
    legend.get_frame().set_linewidth(1.0)
    legend.get_frame().set_facecolor('#F8F8F8')


# 创建图表
fig = plt.figure(figsize=(16, 8))

# 设置图表背景色
fig.patch.set_facecolor('white')

# 分类任务雷达图 - 七边形
classification_values = [original_classification, add_classification,
                         concat_classification, mean_classification]
create_advanced_polygon_radar_chart(categories_classification, classification_values,
                                    ['Original', 'Add', 'Concat', 'Mean'],
                                    '(a) Classification Tasks (ROC-AUC)', fig, 1, 0.6, 1.0)

# 回归任务雷达图 - 三角形
regression_values = [original_regression, add_regression,
                     concat_regression, mean_regression]
create_advanced_polygon_radar_chart(categories_regression, regression_values,
                                    ['Original', 'Add', 'Concat', 'Mean'],
                                    '(b) Regression Tasks (RMSE)', fig, 2, 0.6, 2.5)

# 调整布局
plt.tight_layout()

# 保存图片为高分辨率PNG和PDF文件
plt.savefig('fusion_methods_advanced_polygon_radar_chart.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig('fusion_methods_advanced_polygon_radar_chart.pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none')

# 显示图表
plt.show()

print(
    "高级多边形网格雷达图已保存为 'fusion_methods_advanced_polygon_radar_chart.png' 和 'fusion_methods_advanced_polygon_radar_chart.pdf'")