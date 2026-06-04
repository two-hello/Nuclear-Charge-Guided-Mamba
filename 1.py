import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from rdkit import Chem
from rdkit.Chem import Draw
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from IPython.display import Image, display

# ✅ 彻底解决所有警告
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")
warnings.filterwarnings("ignore", category=UserWarning, module="lightning_utilities.core.imports")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.init")
warnings.filterwarnings("ignore", category=UserWarning, module="torch_geometric")

from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_scipy_sparse_matrix
from ogb.utils import smiles2graph
from torch_geometric.graphgym.config import set_cfg, cfg
from torch_geometric.graphgym.model_builder import create_model
import graphgps  # 确保加载自定义注册器

# ====================== 配置参数 ======================
CONFIG_FILE = "/data/ht/Graph-Mamba-main/configs/Mamba/ogbg-molbbbp-EX.yaml"
MODEL_WEIGHTS = "/data/ht/Graph-Mamba-main/results/ogbg-molbbbp-EX/0/ckpt/59.ckpt"
TARGET_SMILES = "CC(=O)NC1=CC=C(C=C1)O"  # 扑热息痛
SAVE_PATH = "molecule_heatmap.png"

OGB_MOLBBBP_NODE_FEAT_DIM = 9
OGB_MOLBBBP_NUM_TASKS = 1


# ======================================================================

def load_your_gps_model(config_file, model_weights):
    # 1. 初始化并加载 yaml 配置
    set_cfg(cfg)
    cfg.merge_from_file(config_file)

    # 2. 核心修复：显式注入 RWSE 的核心步数和维度
    cfg.dataset.node_encoder_name = "Atom+RWSE"
    cfg.posenc_RWSE.enable = True
    cfg.posenc_RWSE.dim_pe = 16

    # 手动补齐 1 到 16 的步数列表，让框架认出输入是 16 维
    cfg.posenc_RWSE.kernel.times = list(range(1, 17))
    cfg.posenc_RWSE.model = "Linear"
    cfg.posenc_RWSE.raw_norm_type = "BatchNorm"

    # 3. 修正数据集基础维度（根据你的模型实际输出 shape=[1, 1] 设为 1）
    cfg.share.dim_in = 9
    cfg.share.dim_out = 1

    # 4. 创建空模型结构
    model = create_model(to_device=False)

    # 5. 后续正常加载权重
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_weights, map_location=device)

    if 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'], strict=True)
    else:
        model.load_state_dict(checkpoint, strict=True)

    model = model.to(device)
    model.eval()

    return model, device


def compute_rwse_dynamically(data, walk_length=16):
    """动态为单个 PyG Data 对象计算随机游走结构编码 (RWSE)"""
    edge_index = data.edge_index
    num_nodes = data.num_nodes

    # 1. 转换为 scipy 稀疏矩阵
    adj = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes)

    # 2. 计算度矩阵的逆 D^-1
    deg = np.array(adj.sum(axis=1)).flatten()
    deg_inv = np.zeros_like(deg, dtype=np.float32)
    deg_inv[deg > 0] = 1.0 / deg[deg > 0]
    deg_inv_mat = sp.diags(deg_inv)

    # 3. 计算随机游走转移矩阵 P = D^-1 * A
    P = deg_inv_mat.dot(adj)

    # 4. 迭代计算 1 到 walk_length 步的自环概率（矩阵对角线值）
    rwse_list = []
    P_power = sp.eye(num_nodes)
    for _ in range(walk_length):
        P_power = P_power.dot(P)
        rwse_list.append(P_power.diagonal())

    # 5. 拼接为 [num_nodes, walk_length] 的特征矩阵
    rwse = np.stack(rwse_list, axis=1)

    # 绑定为 RWSENodeEncoder 强制要求的属性名
    data.pestat_RWSE = torch.from_numpy(rwse).float()
    return data


def smiles_to_ogb_data(smiles, device):
    """将SMILES转换为OGB标准格式，并补齐随机游走特征与状态标签"""
    graph = smiles2graph(smiles)
    data = Data(
        x=torch.tensor(graph["node_feat"], dtype=torch.long),
        edge_index=torch.tensor(graph["edge_index"], dtype=torch.long),
        edge_attr=torch.tensor(graph["edge_feat"], dtype=torch.long),
        smiles=smiles
    )

    # 计算并注入 16 维的位置编码
    data = compute_rwse_dynamically(data, walk_length=16)

    # 强行注入 split 属性，骗过 GPS 层的训练校验
    data.split = 'test'

    batch = Batch.from_data_list([data]).to(device)
    batch.split = 'test'

    return batch, data


def gps_model_grad_cam(model, batch, target_class=0):
    """动态适配 GraphGPS 内部 Tensor 层的 Grad-CAM (修复反向传播数据污染)"""
    # 强制将输入特征转回 LongTensor，防止被上一次反向传播污染
    if batch.x.dtype != torch.long:
        batch.x = batch.x.long()

    features = []
    gradients = []

    # 1. 动态探测最后一层内部的纯 Tensor 组件（如 nn.Linear）
    target_layer = None
    for name, layer in model.layers[-1].named_modules():
        if isinstance(layer, torch.nn.Linear):
            target_layer = layer

    if target_layer is None:
        submodules = list(model.layers[-1].children())
        target_layer = submodules[-1] if submodules else model.layers[-1]

    print(f"🎯 选定的 Grad-CAM 目标特征捕获层: {target_layer.__class__.__name__}")

    # 2. 定义复用钩子
    def forward_hook(module, input, output):
        if isinstance(output, tuple):
            features.append(output[0].detach())
        elif hasattr(output, 'x'):
            features.append(output.x.detach())
        else:
            features.append(output.detach())

    def backward_hook(module, grad_input, grad_output):
        grad = grad_output[0]
        if grad is not None:
            if hasattr(grad, 'x'):
                gradients.append(grad.x.detach())
            else:
                gradients.append(grad.detach())

    # 注册钩子
    handle_forward = target_layer.register_forward_hook(forward_hook)
    handle_backward = target_layer.register_full_backward_hook(backward_hook)

    # 3. 开启梯度模式进行前向传播
    # 显式允许梯度流，同时在输入前确保数据类型绝对正确
    with torch.set_grad_enabled(True):
        if batch.x.dtype != torch.long:
            batch.x = batch.x.long()

        output = model(batch)
        if isinstance(output, tuple):
            output = output[0]

        target = output[:, target_class]

        model.zero_grad()
        target.backward(retain_graph=False)  # 释放计算图

    # 及时解绑钩子
    handle_forward.remove()
    handle_backward.remove()

    if not features or not gradients:
        raise RuntimeError("❌ 钩子未能成功捕获特征或梯度，请检查网络层结构。")

    # 4. 计算 Grad-CAM 权重
    feature = features[0]
    grad = gradients[0]

    weights = torch.mean(grad, dim=0)
    atom_importance = torch.sum(feature * weights, dim=1).cpu().numpy()

    # 限制在正向贡献并进行零底归一化
    atom_importance = np.maximum(atom_importance, 0)

    # Min-Max 归一化
    denom = atom_importance.max() - atom_importance.min()
    if denom > 1e-8:
        atom_importance = (atom_importance - atom_importance.min()) / denom
    else:
        atom_importance = np.ones_like(atom_importance)

    return atom_importance


def draw_molecule_heatmap(smiles, atom_importance, save_path=None, size=(600, 600), cmap="jet"):
    """绘制真正带二维热力晕染效果的分子结构图，并标注原子索引与得分"""
    from rdkit.Chem import AllChem
    from rdkit.Chem.Draw import SimilarityMaps  # 🌟 引入 RDKit 专业高亮热力图映射模块
    import matplotlib

    mol = Chem.MolFromSmiles(smiles)
    AllChem.Compute2DCoords(mol)

    # 1. 为原子配置标签：显示 "索引:得分" (例如 "O11:1.00")
    for i, score in enumerate(atom_importance):
        atom = mol.GetAtomWithIdx(i)
        # 仅对重要性较高的原子或所有原子加标签，这里全部加上方便比对
        atom.SetProp('atomNote', f"#{i}({score:.2f})")

    # 2. 获取 matplotlib 对应的颜色映射（如 'jet' 或 'Reds'）
    # SimilarityMaps 需要一个 weights 列表，以及对应的 matplotlib cmap 对象
    my_cmap = matplotlib.colormaps[cmap] if hasattr(matplotlib, 'colormaps') else plt.get_cmap(cmap)

    # 3. 构造用于绘图的 Cairo 属性画布
    drawer = Draw.MolDraw2DCairo(size[0], size[1])
    drawer.drawOptions().useBWAtomPalette()  # 使用黑白原子调色板，防止原本元素颜色干扰热力图
    drawer.drawOptions().clearBackground = True  # 保持背景干净
    drawer.drawOptions().atomLabelFontSize = 14  # 调大原子标签
    drawer.drawOptions().annotationFontScale = 0.8  # 调大旁边的 Note (得分) 字体

    # 4. 🌟 核心：调用 RDKit 绘图引擎绘制连续的分子热力图映射
    # 这会在原子周围生成平滑渐变的彩色圆形晕染
    SimilarityMaps.GetSimilarityMapFromWeights(
        mol,
        weights=[float(w) for w in atom_importance],
        colorMap=my_cmap,
        draw2d=drawer,
        contourLines=0  # 设为0代表平滑的颜色渐变热力图；如果想要等高线圈可以设为 3 或 5
    )

    drawer.FinishDrawing()
    img_data = drawer.GetDrawingText()

    # 5. 保存并展示生成的二维热力图
    if save_path:
        with open(save_path, "wb") as f:
            f.write(img_data)
        print(f"✅ 二维晕染热力图已成功保存至: {os.path.abspath(save_path)}")

    display(Image(img_data))

    # 6. 绘制配套 Colorbar 颜色条
    norm = Normalize(vmin=0, vmax=1)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    fig, ax = plt.subplots(figsize=(6, 0.4))
    fig.subplots_adjust(bottom=0.5)
    cb = plt.colorbar(sm, cax=ax, orientation="horizontal")
    cb.set_label("Atom Importance Score (Grad-CAM)", fontsize=10)
    if save_path:
        plt.savefig(save_path.replace(".png", "_colorbar.png"), bbox_inches="tight", dpi=300)
    plt.close(fig)

    return atom_importance


def explain_smiles(smiles, model, device, save_path=None):
    """端到端分析单个分子并打印原子重要性排名"""
    print(f"\n{'=' * 60}")
    print(f"🔍 正在解析目标分子: {smiles}")

    batch, data = smiles_to_ogb_data(smiles, device)
    num_atoms = data.x.shape[0]
    print(f"📊 结构信息: 包含 {num_atoms} 个重原子")

    # 1. 干净地拿一次预测结果
    with torch.no_grad():
        output = model(batch)
        if isinstance(output, tuple):
            output = output[0]
        pred_score = torch.sigmoid(output)[0, 0].item()

    print(f"🔮 模型预测BBBP穿透概率: {pred_score:.4f}")
    print(f"📢 预测结论: {'✅ 该分子可以穿透血脑屏障' if pred_score > 0.5 else '❌ 该分子无法穿透血脑屏障'}")

    # 2. 重新初始化一个干净的干净的 batch 喂给 Grad-CAM，防止反向传播污染
    cam_batch, _ = smiles_to_ogb_data(smiles, device)
    atom_importance = gps_model_grad_cam(model, cam_batch, target_class=0)

    # 3. 绘图
    draw_molecule_heatmap(smiles, atom_importance, save_path=save_path)

    # 4. 打印排名
    mol = Chem.MolFromSmiles(smiles)
    print("\n📊 原子重要性贡献度排名 (Top 5):")
    sorted_indices = np.argsort(atom_importance)[::-1]
    for i in range(min(10, len(sorted_indices))):
        idx = int(sorted_indices[i])
        atom = mol.GetAtomWithIdx(idx)
        print(f"   【排名 {i + 1}】 原子索引 {idx:02d} ({atom.GetSymbol()}): {atom_importance[idx]:.4f}")

    print(f"{'=' * 60}\n")
    return pred_score, atom_importance

if __name__ == "__main__":
    if not os.path.exists(CONFIG_FILE) or not os.path.exists(MODEL_WEIGHTS):
        print(f"❌ 错误：请确认配置文件或权重路径是否正确！")
        sys.exit(1)

    # 1. 初始化并装载模型
    model, device = load_your_gps_model(CONFIG_FILE, MODEL_WEIGHTS)

    # 2. 运行单分子可视化
    pred_score, atom_importance = explain_smiles(
        TARGET_SMILES,
        model,
        device,
        save_path=SAVE_PATH
    )