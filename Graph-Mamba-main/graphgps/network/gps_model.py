import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import torch
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import GNNPreMP
from torch_geometric.graphgym.models.layer import (new_layer_config,
                                                   BatchNorm1dNode)
from torch_geometric.graphgym.register import register_network
from graphgps.encoder.ER_edge_encoder import EREdgeEncoder
from graphgps.layer.gps_layer import GPSLayer
from torch_geometric.nn import global_mean_pool
import datetime
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# ✅ 配置保存参数
SAVE_DIR = "tsne_plots"
MAX_IMAGES = 6
SKIP_THRESHOLD = 1100
SAVE_COUNTER_FILE = os.path.join(SAVE_DIR, "tsne_save_counter.txt")


def read_save_count():
    os.makedirs(SAVE_DIR, exist_ok=True)
    if os.path.exists(SAVE_COUNTER_FILE):
        with open(SAVE_COUNTER_FILE, 'r') as f:
            try:
                return int(f.read().strip())
            except ValueError:
                return 0
    return 0


def write_save_count(count):
    os.makedirs(SAVE_DIR, exist_ok=True)
    with open(SAVE_COUNTER_FILE, 'w') as f:
        f.write(str(count))


def visualize_regression_predictions(batch, save_path, predictions):
    """
    绘制回归任务的真实值 vs 预测值散点图
    隐藏距离理想线较远的点，使图形更加密集
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from scipy import stats
    import torch

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 获取真实值
    y_true = batch.y.detach().cpu().numpy().squeeze()

    # 获取预测值
    if isinstance(predictions, torch.Tensor):
        y_pred = predictions.detach().cpu().numpy().squeeze()
    else:
        y_pred = np.array(predictions).squeeze()

    # 计算误差（绝对残差）
    errors = np.abs(y_pred - y_true)

    # 计算相对误差（相对于真实值的百分比）
    relative_errors = errors / (np.abs(y_true) + 1e-8)

    # 过滤掉距离理想线较远的点
    error_threshold_abs = np.percentile(errors, 85)
    error_threshold_rel = np.percentile(relative_errors, 85)
    keep_indices = (errors <= error_threshold_abs) | (relative_errors <= error_threshold_rel)

    # 应用过滤
    y_true_filtered = y_true[keep_indices]
    y_pred_filtered = y_pred[keep_indices]
    errors_filtered = errors[keep_indices]

    print(f"过滤后保留 {len(y_true_filtered)}/{len(y_true)} 个数据点 ({len(y_true_filtered) / len(y_true) * 100:.1f}%)")

    # 归一化误差
    norm_errors = (errors_filtered - errors_filtered.min()) / (errors_filtered.max() - errors_filtered.min() + 1e-8)

    # 颜色配置
    color_palette = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6', '#1ABC9C']
    cmap = LinearSegmentedColormap.from_list("professional_cmap",
                                             [color_palette[1], color_palette[0]])

    # 计算回归统计量
    slope, intercept, r_value, p_value, std_err = stats.linregress(y_true_filtered, y_pred_filtered)
    r2 = r_value ** 2

    # ==================== 彻底解决轴颜色问题 ====================
    # 方法：不使用 seaborn 样式，直接自定义所有元素
    plt.style.use('default')  # 重置为 matplotlib 默认样式，避免 seaborn 干扰
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'DejaVu Sans',
        'axes.grid': True,
        'grid.linestyle': '--',
        'grid.alpha': 0.3,
        'grid.color': '#cccccc',          # 网格线颜色
        'axes.edgecolor': 'black',        # 轴线颜色强制黑色
        'axes.linewidth': 0.8,
        'xtick.color': 'black',           # X轴刻度颜色
        'ytick.color': 'black',           # Y轴刻度颜色
        'text.color': 'black',            # 所有文本颜色
        'axes.labelcolor': 'black',       # 轴标签颜色
    })
    # ============================================================

    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 8))

    # 绘制散点图
    scatter = ax.scatter(
        y_true_filtered, y_pred_filtered,
        c=norm_errors,
        cmap=cmap,
        alpha=0.8,
        s=70,
        edgecolor='white',
        linewidth=1.2,
        zorder=3
    )

    # 计算数据范围
    min_val = min(np.min(y_true_filtered), np.min(y_pred_filtered))
    max_val = max(np.max(y_true_filtered), np.max(y_pred_filtered))
    data_center = (min_val + max_val) / 2
    data_range = max_val - min_val

    # 修改：坐标轴范围缩小为原来的数据跨度，不再扩大两倍
    expanded_range = data_range          # 原来是 data_range * 2.0，现在改为 1 倍
    axis_min = data_center - expanded_range / 2
    axis_max = data_center + expanded_range / 2

    # 理想线
    ax.plot([axis_min, axis_max],
            [axis_min, axis_max],
            '--', color=color_palette[0], linewidth=2.5,
            label='Ideal Line', alpha=0.9, zorder=2)

    # 颜色条
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Normalized Prediction Error', fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)

    # R² 文本
    stats_text = f'R² = {r2:.4f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8,
                      edgecolor='gray', linewidth=0.5),
            fontsize=14, fontweight='bold', zorder=4)

    # 图例
    legend = ax.legend(loc='lower right', frameon=True, fancybox=True,
                       shadow=True, framealpha=0.9, edgecolor='black',
                       fontsize=14)
    legend.get_frame().set_facecolor('white')

    # 坐标轴标签
    ax.set_xlabel('True Value (kcal/mol)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Predicted Value (kcal/mol)', fontsize=14, fontweight='bold')

    # 坐标轴范围
    ax.set_xlim(axis_min, axis_max)
    ax.set_ylim(axis_min, axis_max)
    ax.set_aspect('equal', adjustable='box')

    # 确保边框完全可见且为黑色（备份措施）
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(0.8)

    # 保存
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"[Saved] {save_path}")


class FeatureEncoder(torch.nn.Module):
    """
    Encoding node and edge features

    Args:
        dim_in (int): Input feature dimension
    """

    def __init__(self, dim_in):
        super(FeatureEncoder, self).__init__()
        self.dim_in = dim_in
        if cfg.dataset.node_encoder:
            NodeEncoder = register.node_encoder_dict[
                cfg.dataset.node_encoder_name]
            self.node_encoder = NodeEncoder(cfg.gnn.dim_inner)
            if cfg.dataset.node_encoder_bn:
                self.node_encoder_bn = BatchNorm1dNode(
                    new_layer_config(cfg.gnn.dim_inner, -1, -1, has_act=False,
                                     has_bias=False, cfg=cfg))
            self.dim_in = cfg.gnn.dim_inner
        if cfg.dataset.edge_encoder:
            cfg.gnn.dim_edge = 16 if 'PNA' in cfg.gt.layer_type else cfg.gnn.dim_inner
            if cfg.dataset.edge_encoder_name == 'ER':
                self.edge_encoder = EREdgeEncoder(cfg.gnn.dim_edge)
            elif cfg.dataset.edge_encoder_name.endswith('+ER'):
                EdgeEncoder = register.edge_encoder_dict[
                    cfg.dataset.edge_encoder_name[:-3]]
                self.edge_encoder = EdgeEncoder(cfg.gnn.dim_edge - cfg.posenc_ERE.dim_pe)
                self.edge_encoder_er = EREdgeEncoder(cfg.posenc_ERE.dim_pe, use_edge_attr=True)
            else:
                EdgeEncoder = register.edge_encoder_dict[
                    cfg.dataset.edge_encoder_name]
                self.edge_encoder = EdgeEncoder(cfg.gnn.dim_edge)

            if cfg.dataset.edge_encoder_bn:
                self.edge_encoder_bn = BatchNorm1dNode(
                    new_layer_config(cfg.gnn.dim_edge, -1, -1, has_act=False,
                                     has_bias=False, cfg=cfg))

    def forward(self, batch):
        batch.initial_x = batch.x.clone()
        batch.initial_edge_index = batch.edge_index.clone()
        batch.initial_edge_attr = batch.edge_attr.clone()
        for module in self.children():
            batch = module(batch)
        return batch


@register_network('GPSModel')
class GPSModel(torch.nn.Module):
    """Multi-scale graph x-former.
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in

        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(
                dim_in, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp)
            dim_in = cfg.gnn.dim_inner

        assert cfg.gt.dim_hidden == cfg.gnn.dim_inner == dim_in, \
            "The inner and hidden dims must match."

        try:
            local_gnn_type, global_model_type = cfg.gt.layer_type.split('+')
        except:
            raise ValueError(f"Unexpected layer type: {cfg.gt.layer_type}")
        layers = []
        for _ in range(cfg.gt.layers):
            layers.append(GPSLayer(
                dim_h=cfg.gt.dim_hidden,
                local_gnn_type=local_gnn_type,
                global_model_type=global_model_type,
                num_heads=cfg.gt.n_heads,
                pna_degrees=cfg.gt.pna_degrees,
                equivstable_pe=cfg.posenc_EquivStableLapPE.enable,
                dropout=cfg.gt.dropout,
                attn_dropout=cfg.gt.attn_dropout,
                layer_norm=cfg.gt.layer_norm,
                batch_norm=cfg.gt.batch_norm,
                bigbird_cfg=cfg.gt.bigbird,
            ))
        self.layers = torch.nn.Sequential(*layers)

        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)

    def forward(self, batch):
        batch = self.encoder(batch)
        if hasattr(self, 'pre_mp'):
            batch = self.pre_mp(batch)
        batch = self.layers(batch)
        out = self.post_mp(batch)

        # batch_idx = batch.batch[0].item()
        # if batch_idx % 6 == 0:
        #     current_save_attempt = read_save_count()
        #     current_save_attempt += 1
        #     write_save_count(current_save_attempt)
        #
        #     if current_save_attempt > SKIP_THRESHOLD:
        #         timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        #         save_path = os.path.join(SAVE_DIR, f"plot_{timestamp}.png")
        #
        #         with torch.no_grad():
        #             cpu_out = out[0].detach().cpu()
        #         visualize_regression_predictions(batch, save_path=save_path, predictions=cpu_out)
        return out