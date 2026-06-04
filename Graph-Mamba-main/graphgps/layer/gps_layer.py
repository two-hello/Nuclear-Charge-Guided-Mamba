# -*- coding: utf-8 -*-
import matplotlib  # 必须导入主模块
import matplotlib.pyplot as plt  # 导入pyplot子模块
import seaborn as sns
import os
import time
import glob # 导入 glob 模块用于文件列表操作
import numpy as np
import torch
from sklearn.manifold import TSNE
from torch_geometric.nn import global_mean_pool

torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pygnn
from performer_pytorch import SelfAttention
from torch_geometric.data import Batch
from torch_geometric.nn import Linear as Linear_pyg
from torch_geometric.utils import to_dense_batch

from graphgps.layer.gatedgcn_layer import GatedGCNLayer
from graphgps.layer.gine_conv_layer import GINEConvESLapPE
from graphgps.layer.bigbird_layer import SingleBigBirdLayer
from mamba_ssm import Mamba

from torch_geometric.utils import degree, sort_edge_index
from typing import List

import numpy as np
import torch
from torch import Tensor
from typing import Optional

from torch_geometric.nn import GATConv

from torch_geometric.utils import to_dense_batch
import networkx as nx



from rdkit import Chem
import re


# ========== 新增导入 ==========
from datetime import datetime
import umap.umap_ as umap
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec

# ========== 优化后的基础 KAN 模块 ==========
# ========== 修正后的基础 KAN 模块（参数量减少75%+） ==========
class SimpleKAN(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, k: int = 8):
        super().__init__()
        self.in_dim = in_dim
        self.k = k

        self.u_weights = nn.Parameter(torch.Tensor(in_dim, k))
        self.u_biases = nn.Parameter(torch.Tensor(in_dim, k))
        self.v_weights = nn.Parameter(torch.Tensor(in_dim, k))
        self.v_biases = nn.Parameter(torch.Tensor(in_dim, 1))
        self.w_funcs = nn.Linear(in_dim, out_dim)

        # 可学习的Swish参数
        self.beta = nn.Parameter(torch.ones(1))

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.u_weights, a=5 ** 0.5)
        nn.init.kaiming_uniform_(self.v_weights, a=5 ** 0.5)
        nn.init.constant_(self.u_biases, 0)
        nn.init.constant_(self.v_biases, 0)

    def forward(self, x):
        x_reshaped = x.unsqueeze(-1)  # [B, in_dim, 1]

        # 使用Swish激活代替ReLU
        u_out = x_reshaped * self.u_weights.unsqueeze(0) + self.u_biases.unsqueeze(0)
        u_out = u_out * torch.sigmoid(self.beta * u_out)  # Swish激活

        v_out = torch.sum(u_out * self.v_weights.unsqueeze(0), dim=2) + self.v_biases.squeeze(-1)

        return self.w_funcs(v_out)


# ========== 同步修正KANLayer ==========
class KANLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, k: int = 8, dropout: float = 0.1):
        super().__init__()
        self.kan = SimpleKAN(in_dim, out_dim, k=k)  # 传入基函数数量k
        self.dropout = nn.Dropout(dropout)

        if in_dim != out_dim:
            self.res_proj = nn.Linear(in_dim, out_dim)
        else:
            self.res_proj = nn.Identity()

    def forward(self, x):
        out = self.dropout(self.kan(x))
        return out + self.res_proj(x)
# ========== U-KAN 网络 ==========
class UKAN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, depth=2):
        """
        U-KAN: 编码器-瓶颈-解码器结构
        参数:
            in_dim      : 输入特征维度
            hidden_dim  : 隐藏层维度
            out_dim     : 输出维度
            depth       : 下采样/上采样的层数
        """
        super().__init__()

        # 编码器
        self.encoder = nn.ModuleList()
        dims = [in_dim] + [hidden_dim] * depth
        for i in range(depth):
            self.encoder.append(KANLayer(dims[i], hidden_dim, dims[i+1]))

        # 瓶颈
        self.bottleneck = KANLayer(hidden_dim, hidden_dim, hidden_dim)

        # 解码器
        self.decoder = nn.ModuleList()
        for i in range(depth-1, -1, -1):
            self.decoder.append(KANLayer(hidden_dim + dims[i], hidden_dim, dims[i]))

        # 最终输出
        self.final = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        # 编码
        skips = []
        out = x
        for enc in self.encoder:
            out = enc(out)
            skips.append(out)

        # 瓶颈
        out = self.bottleneck(out)

        # 解码 + skip connection
        for dec, skip in zip(self.decoder, reversed(skips)):
            out = torch.cat([out, skip], dim=-1)  # U-Net 拼接
            out = dec(out)

        return self.final(out)


def graph_to_smiles(batch, batch_idx=0):
    """从 batch 中第 batch_idx 个分子图反推 SMILES；失败返回 None"""
    mask = batch.batch == batch_idx
    node_idx = mask.nonzero(as_tuple=True)[0]          # 全局节点索引
    z = batch.initial_x[node_idx, 0].long().cpu().numpy()
    z = z[(z > 0) & (z <= 118)]
    if len(z) == 0:
        return None

    # 只保留该分子的边
    e_mask = mask[batch.initial_edge_index[0]] & mask[batch.initial_edge_index[1]]
    edge_index = batch.initial_edge_index[:, e_mask].cpu().numpy()

    # 用 NetworkX 建图，节点编号天然 0..n-1
    G = nx.Graph()
    for i, atom_z in enumerate(z):
        G.add_node(i, atom_num=int(atom_z))

    for u, v in edge_index.T:
        if u == v:                       # 忽略自环
            continue
        # 把全局索引映射到 0..n-1
        local_u = int((node_idx == u).nonzero(as_tuple=True)[0])
        local_v = int((node_idx == v).nonzero(as_tuple=True)[0])
        G.add_edge(local_u, local_v)

    # 用 RDKit 从 NetworkX 转 SMILES
    try:
        mol = nx_to_rdkit(G)
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol, isomericSmiles=False)
    except Exception:
        return None


# ---------- 辅助：NetworkX → RDKit ----------
def nx_to_rdkit(G):
    mol = Chem.RWMol()
    # 按节点顺序添加原子
    for n in sorted(G.nodes):
        mol.AddAtom(Chem.Atom(G.nodes[n]['atom_num']))
    # 添加边，默认单键
    for u, v in G.edges:
        mol.AddBond(int(u), int(v), Chem.rdchem.BondType.SINGLE)
    return mol
# GraphAttentionModel for calculating node attention using GATConv
class GraphAttentionModel(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GraphAttentionModel, self).__init__()
        # 定义两个 GAT 层
        self.gat1 = GATConv(in_channels, out_channels, heads=1, concat=False)
        self.gat2 = GATConv(out_channels, out_channels, heads=1, concat=False)

    def compute_attention_weights(self, h, edge_index, num_nodes: Optional[int] = None, dtype: Optional[torch.dtype] = None):
        """
        计算节点的注意力权重，返回一个一维张量，其中每个元素是对应节点的注意力权重。
        """
        # 使用 GATConv 层计算节点的注意力权重
        h_attn = self.gat1(h, edge_index)  # 第一层 GAT
        h_attn = F.relu(h_attn)
        h_attn = self.gat2(h_attn, edge_index)  # 第二层 GAT

        # 将输出转化为一维向量，表示每个节点的注意力权重
        # 取每个节点的注意力权重（通过 pooling 或其他方式）
        attention_weights = h_attn.mean(dim=1)  # 可以通过取每个节点的特征均值来获得节点的注意力权重

        # 使输出维度一致，返回一个与节点数量匹配的一维张量
        if num_nodes is not None:
            # 如果有指定节点数量，确保返回的张量大小为 num_nodes
            out = torch.zeros((num_nodes,), dtype=dtype, device=h.device)
            out[:attention_weights.size(0)] = attention_weights  # 只填充有效的节点数
        else:
            out = attention_weights  # 如果没有指定节点数，则直接返回注意力权重

        return out




def permute_nodes_within_identity(identities):
    unique_identities, inverse_indices = torch.unique(identities, return_inverse=True)
    node_indices = torch.arange(len(identities), device=identities.device)

    masks = identities.unsqueeze(0) == unique_identities.unsqueeze(1)

    # Generate random indices within each identity group using torch.randint
    permuted_indices = torch.cat([
        node_indices[mask][torch.randperm(mask.sum(), device=identities.device)] for mask in masks
    ])
    return permuted_indices


def sort_rand_gpu(pop_size, num_samples, neighbours):
    # Randomly generate indices and select num_samples in neighbours
    idx_select = torch.argsort(torch.rand(pop_size, device=neighbours.device))[:num_samples]
    neighbours = neighbours[idx_select]
    return neighbours


def augment_seq(edge_index, batch, num_k=-1):
    unique_batches = torch.unique(batch)
    # Initialize list to store permuted indices
    permuted_indices = []
    mask = []

    for batch_index in unique_batches:
        # Extract indices for the current batch
        indices_in_batch = (batch == batch_index).nonzero().squeeze()
        for k in indices_in_batch:
            neighbours = edge_index[1][edge_index[0] == k]
            if num_k > 0 and len(neighbours) > num_k:
                neighbours = sort_rand_gpu(len(neighbours), num_k, neighbours)
            permuted_indices.append(neighbours)
            mask.append(torch.zeros(neighbours.shape, dtype=torch.bool, device=batch.device))
            permuted_indices.append(torch.tensor([k], device=batch.device))
            mask.append(torch.tensor([1], dtype=torch.bool, device=batch.device))
    permuted_indices = torch.cat(permuted_indices)
    mask = torch.cat(mask)
    return permuted_indices.to(device=batch.device), mask.to(device=batch.device)


def lexsort(
        keys: List[Tensor],
        dim: int = -1,
        descending: bool = False,
) -> Tensor:
    r"""Performs an indirect stable sort using a sequence of keys.

    Given multiple sorting keys, returns an array of integer indices that
    describe their sort order.
    The last key in the sequence is used for the primary sort order, the
    second-to-last key for the secondary sort order, and so on.

    Args:
        keys ([torch.Tensor]): The :math:`k` different columns to be sorted.
            The last key is the primary sort key.
        dim (int, optional): The dimension to sort along. (default: :obj:`-1`)
        descending (bool, optional): Controls the sorting order (ascending or
            descending). (default: :obj:`False`)
    """
    assert len(keys) >= 1

    out = keys[0].argsort(dim=dim, descending=descending, stable=True)
    for k in keys[1:]:
        index = k.gather(dim, out)
        index = index.argsort(dim=dim, descending=descending, stable=True)
        out = out.gather(dim, index)
    return out


def permute_within_batch(batch):
    # Enumerate over unique batch indices
    unique_batches = torch.unique(batch)

    # Initialize list to store permuted indices
    permuted_indices = []

    for batch_index in unique_batches:
        # Extract indices for the current batch - 确保始终返回1D张量
        indices_in_batch = (batch == batch_index).nonzero(as_tuple=True)[0]

        # 获取元素数量（安全处理0D/1D张量）
        n = indices_in_batch.numel()

        # 只有至少2个元素时才需要随机排列
        if n > 1:
            permuted_indices_in_batch = indices_in_batch[torch.randperm(n)]
        else:
            permuted_indices_in_batch = indices_in_batch  # 单个元素不需要排列

        # Append permuted indices to the list
        permuted_indices.append(permuted_indices_in_batch)

    # Concatenate permuted indices into a single tensor
    permuted_indices = torch.cat(permuted_indices)

    return permuted_indices


class GPSLayer(nn.Module):
    """Local MPNN + full graph attention x-former layer.
    """

    def __init__(self, dim_h,
                 local_gnn_type, global_model_type, num_heads,
                 pna_degrees=None, equivstable_pe=False, dropout=0.1,
                 attn_dropout=0.1, layer_norm=True, batch_norm=False,
                 bigbird_cfg=None):
        super().__init__()

        self.dim_h = dim_h
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.equivstable_pe = equivstable_pe
        self.NUM_BUCKETS = 3

        # 可学习的特征权重：[自由基电子数权重, 原子序数权重]
        self.feat_weights = nn.Parameter(torch.ones(2))  # 初始权重1:1

        # ========== 保留你原来的local_model和self_attn定义 ==========
        # Local message-passing model.
        if local_gnn_type == 'None':
            self.local_model = None
        elif local_gnn_type == 'GENConv':
            self.local_model = pygnn.GENConv(dim_h, dim_h)
        elif local_gnn_type == 'GINE':
            gin_nn = nn.Sequential(Linear_pyg(dim_h, dim_h),
                                   nn.ReLU(),
                                   Linear_pyg(dim_h, dim_h))
            if self.equivstable_pe:  # Use specialised GINE layer for EquivStableLapPE.
                self.local_model = GINEConvESLapPE(gin_nn)
            else:
                self.local_model = pygnn.GINEConv(gin_nn)
        elif local_gnn_type == 'GAT':
            self.local_model = pygnn.GATConv(in_channels=dim_h,
                                             out_channels=dim_h // num_heads,
                                             heads=num_heads,
                                             edge_dim=dim_h)
        elif local_gnn_type == 'PNA':
            # Defaults from the paper.
            aggregators = ['mean', 'max', 'sum']
            scalers = ['identity']
            deg = torch.from_numpy(np.array(pna_degrees))
            self.local_model = pygnn.PNAConv(dim_h, dim_h,
                                             aggregators=aggregators,
                                             scalers=scalers,
                                             deg=deg,
                                             edge_dim=16,
                                             towers=1,
                                             pre_layers=1,
                                             post_layers=1,
                                             divide_input=False)
        elif local_gnn_type == 'CustomGatedGCN':
            self.local_model = GatedGCNLayer(dim_h, dim_h,
                                             dropout=dropout,
                                             residual=True,
                                             equivstable_pe=equivstable_pe)
        else:
            raise ValueError(f"Unsupported local GNN model: {local_gnn_type}")
        self.local_gnn_type = local_gnn_type

        # Global attention transformer-style model.
        if global_model_type == 'None':
            self.self_attn = None
        elif global_model_type == 'Transformer':
            self.self_attn = torch.nn.MultiheadAttention(
                dim_h, num_heads, dropout=self.attn_dropout, batch_first=True)
        elif global_model_type == 'Performer':
            self.self_attn = SelfAttention(
                dim=dim_h, heads=num_heads,
                dropout=self.attn_dropout, causal=False)
        elif global_model_type == "BigBird":
            bigbird_cfg.dim_hidden = dim_h
            bigbird_cfg.n_heads = num_heads
            bigbird_cfg.dropout = dropout
            self.self_attn = SingleBigBirdLayer(bigbird_cfg)
        elif 'Mamba' in global_model_type:
            if global_model_type.split('_')[-1] == '2':
                self.self_attn = Mamba(d_model=dim_h, d_state=8, d_conv=4, expand=2)
            elif global_model_type.split('_')[-1] == '4':
                self.self_attn = Mamba(d_model=dim_h, d_state=4, d_conv=4, expand=4)
            elif global_model_type.split('_')[-1] == 'Multi':
                self.self_attn = nn.ModuleList([
                    Mamba(d_model=dim_h, d_state=16, d_conv=4, expand=1)
                    for _ in range(4)
                ])
            elif global_model_type.split('_')[-1] == 'SmallConv':
                self.self_attn = Mamba(d_model=dim_h, d_state=16, d_conv=2, expand=1)
            elif global_model_type.split('_')[-1] == 'SmallState':
                self.self_attn = Mamba(d_model=dim_h, d_state=8, d_conv=4, expand=1)
            else:
                self.self_attn = Mamba(d_model=dim_h, d_state=16, d_conv=4, expand=1)
        else:
            raise ValueError(f"Unsupported global x-former model: {global_model_type}")
        self.global_model_type = global_model_type

        if self.layer_norm and self.batch_norm:
            raise ValueError("Cannot apply two types of normalization together")

        # Normalization for MPNN and Self-Attention representations.
        if self.layer_norm:
            self.norm1_local = pygnn.norm.GraphNorm(dim_h)
            self.norm1_attn = pygnn.norm.GraphNorm(dim_h)
        if self.batch_norm:
            self.norm1_local = nn.BatchNorm1d(dim_h)
            self.norm1_attn = nn.BatchNorm1d(dim_h)
        self.dropout_local = nn.Dropout(dropout)
        self.dropout_attn = nn.Dropout(dropout)

        # ========== 完全按照架构图修正融合层定义 ==========
        # 1. 多头注意力层：输入2*dim_h，batch_first=True
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=2 * dim_h,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True  # ✅ 关键修复
        )

        # 2. ResKAN Layer：对应架构图中的ResKAN Layer
        self.reskan_layer = KANLayer(2 * dim_h, 2 * dim_h, k=8, dropout=dropout)

        # 3. KAN FFN：对应架构图中的KAN FFN，将2*dim_h降为dim_h
        self.kan_ffn = nn.Sequential(
            KANLayer(2 * dim_h, 2 * dim_h, k=3, dropout=dropout),
            nn.Linear(2 * dim_h, dim_h)
        )

        # 4. 归一化层
        if self.layer_norm:
            self.norm_attn_fusion = pygnn.norm.GraphNorm(2 * dim_h)
            self.norm_reskan = pygnn.norm.GraphNorm(2 * dim_h)
            self.norm_final = pygnn.norm.GraphNorm(dim_h)
        if self.batch_norm:
            self.norm_attn_fusion = nn.BatchNorm1d(2 * dim_h)
            self.norm_reskan = nn.BatchNorm1d(2 * dim_h)
            self.norm_final = nn.BatchNorm1d(dim_h)

        # 5. Dropout层
        self.dropout_fusion = nn.Dropout(dropout)
        self.dropout_reskan = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)

        # ========== 删除所有冗余定义 ==========
        # 删除：self.multihead_attention_local, self.multihead_attention_attn
        # 删除：self.ff_kan1, self.ff_kan2, self.ff_dropout1, self.ff_dropout2
        # 删除：self.norm2

    def forward(self, batch):
        h = batch.x
        h_original = h  # ✅ 保存H_original，用于最终残差连接
        h_in2 = batch.initial_x

        # ✅ 添加：确保所有输入数据都在GPU上
        device = next(self.parameters()).device
        if not h.is_cuda:
            h = h.to(device)
            batch.x = batch.x.to(device)
            batch.initial_x = batch.initial_x.to(device) if hasattr(batch, 'initial_x') else None
            if hasattr(batch, 'edge_index') and batch.edge_index is not None:
                batch.edge_index = batch.edge_index.to(device)
            if hasattr(batch, 'edge_attr') and batch.edge_attr is not None:
                batch.edge_attr = batch.edge_attr.to(device)
            if hasattr(batch, 'batch') and batch.batch is not None:
                batch.batch = batch.batch.to(device)

        # ========== 保留你原来的局部GNN处理代码 ==========
        h_local = None
        if self.local_model is not None:
            self.local_model: pygnn.conv.MessagePassing  # Typing hint.
            if self.local_gnn_type == 'CustomGatedGCN':
                es_data = None
                if self.equivstable_pe:
                    es_data = batch.pe_EquivStableLapPE
                local_out = self.local_model(Batch(batch=batch,
                                                   x=h,
                                                   edge_index=batch.edge_index,
                                                   edge_attr=batch.edge_attr,
                                                   pe_EquivStableLapPE=es_data))
                h_local = local_out.x
                batch.edge_attr = local_out.edge_attr
            else:
                if self.equivstable_pe:
                    h_local = self.local_model(h, batch.edge_index, batch.edge_attr,
                                               batch.pe_EquivStableLapPE)
                else:
                    h_local = self.local_model(h, batch.edge_index, batch.edge_attr)
                h_local = self.dropout_local(h_local)
                h_local = h + h_local  # Residual connection.

            if self.layer_norm:
                h_local = self.norm1_local(h_local, batch.batch)
            if self.batch_norm:
                h_local = self.norm1_local(h_local)

        # ========== 保留你原来的全局Mamba处理代码 ==========
        h_attn = None
        if self.self_attn is not None:
            if self.global_model_type in ['Transformer', 'Performer', 'BigBird', 'Mamba']:
                h_dense, mask = to_dense_batch(h, batch.batch)
            if self.global_model_type == 'Transformer':
                h_attn = self._sa_block(h_dense, None, ~mask)[mask]
            elif self.global_model_type == 'Performer':
                h_attn = self.self_attn(h_dense, mask=mask)[mask]
            elif self.global_model_type == 'BigBird':
                h_attn = self.self_attn(h_dense, attention_mask=mask)
            elif self.global_model_type == 'Mamba':
                h_attn = self.self_attn(h_dense)[mask]
            elif self.global_model_type == 'Mamba_Permute':
                h_ind_perm = permute_within_batch(batch.batch)
                h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
                h_ind_perm_reverse = torch.argsort(h_ind_perm)
                h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]
            elif 'Mamba_Hybrid_Degree_Noise' == self.global_model_type:
                if batch.split == 'train':
                    deg = degree(batch.edge_index[0], batch.x.shape[0]).to(torch.float)
                    deg_noise = torch.rand_like(deg).to(deg.device)
                    h_ind_perm = lexsort([deg + deg_noise, batch.batch])
                    h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
                    h_ind_perm_reverse = torch.argsort(h_ind_perm)
                    h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]
                else:
                    mamba_arr = []
                    deg = degree(batch.edge_index[0], batch.x.shape[0]).to(torch.float)
                    for i in range(5):
                        deg_noise = torch.rand_like(deg).to(deg.device)
                        h_ind_perm = lexsort([deg + deg_noise, batch.batch])
                        h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
                        h_ind_perm_reverse = torch.argsort(h_ind_perm)
                        h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]
                        mamba_arr.append(h_attn)
                    h_attn = sum(mamba_arr) / 5
            elif self.global_model_type == 'Mamba_Test':
                if batch.split == 'train':
                    gat_model = GraphAttentionModel(self.dim_h, self.dim_h // self.num_heads).to(h.device)
                    attention_weights = gat_model.compute_attention_weights(h, batch.edge_index)
                    attention_weights_noise = torch.rand_like(attention_weights).to(attention_weights.device)
                    perturbed_attention_weights = attention_weights + attention_weights_noise
                    if perturbed_attention_weights.dim() > 1:
                        perturbed_attention_weights = perturbed_attention_weights.view(-1)
                    h_ind_perm = lexsort([perturbed_attention_weights, batch.batch])
                    h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
                    h_ind_perm_reverse = torch.argsort(h_ind_perm)
                    h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]
                else:
                    mamba_arr = []
                    gat_model = GraphAttentionModel(self.dim_h, self.dim_h // self.num_heads).to(h.device)
                    attention_weights = gat_model.compute_attention_weights(h, batch.edge_index)
                    for i in range(5):
                        attention_weights_noise = torch.rand_like(attention_weights).to(attention_weights.device)
                        perturbed_attention_weights = attention_weights + attention_weights_noise
                        if perturbed_attention_weights.dim() > 1:
                            perturbed_attention_weights = perturbed_attention_weights.view(-1)
                        h_ind_perm = lexsort([perturbed_attention_weights, batch.batch])
                        h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
                        h_ind_perm_reverse = torch.argsort(h_ind_perm)
                        h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]
                        mamba_arr.append(h_attn)
                    h_attn = sum(mamba_arr) / 5
            elif 'Mamba_Hybrid_Importance' == self.global_model_type:
                if batch.split == 'train':
                    feat_weights = F.softmax(self.feat_weights, dim=0)
                    importance = torch.matmul(h[:, :9], feat_weights)
                    importance_noise = torch.rand_like(importance)
                    sorted_scores = importance + importance_noise
                    h_ind_perm = lexsort([sorted_scores, batch.batch])
                    h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
                    h_ind_perm_reverse = torch.argsort(h_ind_perm)
                    h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]
                else:
                    mamba_arr = []
                    feat_weights = F.softmax(self.feat_weights, dim=0)
                    base_importance = torch.matmul(h[:, :9], feat_weights)
                    for _ in range(5):
                        importance_noise = torch.randn_like(base_importance) * 0.1
                        sorted_scores = base_importance + importance_noise
                        h_ind_perm = lexsort([-sorted_scores, batch.batch])
                        h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
                        h_ind_perm_reverse = torch.argsort(h_ind_perm)
                        mamba_arr.append(self.self_attn(h_dense)[mask][h_ind_perm_reverse])
                    h_attn = sum(mamba_arr) / 5
            elif 'Mamba_Hybrid_H_Num' == self.global_model_type:
                if batch.split == 'train':
                    h_num = h[:, 4].to(torch.float)
                    h_num_noise = torch.rand_like(h_num)
                    sorted_scores = h_num + h_num_noise
                    h_ind_perm = lexsort([-sorted_scores, batch.batch])
                    h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
                    h_ind_perm_reverse = torch.argsort(h_ind_perm)
                    h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]
                else:
                    mamba_arr = []
                    base_h_num = h[:, 4].to(torch.float)
                    for _ in range(5):
                        h_num_noise = torch.rand_like(base_h_num)
                        sorted_scores = base_h_num + h_num_noise
                        h_ind_perm = lexsort([-sorted_scores, batch.batch])
                        h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
                        h_ind_perm_reverse = torch.argsort(h_ind_perm)
                        mamba_arr.append(self.self_attn(h_dense)[mask][h_ind_perm_reverse])
                    h_attn = sum(mamba_arr) / 5
            elif 'Mamba_Hybrid_Radical' == self.global_model_type:
                if batch.split == 'train':
                    radical = h[:, 5].to(torch.float)
                    radical_noise = torch.rand_like(radical)
                    sorted_scores = radical + radical_noise
                    h_ind_perm = lexsort([sorted_scores, batch.batch])
                    h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
                    h_ind_perm_reverse = torch.argsort(h_ind_perm)
                    h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]
                else:
                    mamba_arr = []
                    base_radical = h[:, 5].to(torch.float)
                    for _ in range(5):
                        radical_noise = torch.rand_like(base_radical)
                        sorted_scores = base_radical + radical_noise
                        h_ind_perm = lexsort([sorted_scores, batch.batch])
                        h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
                        h_ind_perm_reverse = torch.argsort(h_ind_perm)
                        mamba_arr.append(self.self_attn(h_dense)[mask][h_ind_perm_reverse])
                    h_attn = sum(mamba_arr) / 5
            elif 'Mamba_yuanzi' == self.global_model_type:
                if batch.split == 'train':
                    radical = h_in2[:, 0].to(torch.float)
                    radical_noise = torch.rand_like(radical)
                    sorted_scores = radical + radical_noise
                    h_ind_perm = lexsort([sorted_scores, batch.batch])
                    h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
                    h_ind_perm_reverse = torch.argsort(h_ind_perm)
                    h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]
                else:
                    mamba_arr = []
                    base_radical = h_in2[:, 0].to(torch.float)
                    for _ in range(5):
                        radical_noise = torch.rand_like(base_radical)
                        sorted_scores = base_radical + radical_noise
                        h_ind_perm = lexsort([sorted_scores, batch.batch])
                        h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
                        h_ind_perm_reverse = torch.argsort(h_ind_perm)
                        mamba_arr.append(self.self_attn(h_dense)[mask][h_ind_perm_reverse])
                    h_attn = sum(mamba_arr) / 5
            elif 'Mamba_Hybrid_FormalCharge' == self.global_model_type:
                if batch.split == 'train':
                    formal_charge = h_in2[:, 3].to(torch.float).abs()
                    charge_noise = torch.rand_like(formal_charge)
                    sorted_scores = formal_charge + charge_noise
                    h_ind_perm = lexsort([sorted_scores, batch.batch])
                    h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
                    h_ind_perm_reverse = torch.argsort(h_ind_perm)
                    h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]
                else:
                    mamba_arr = []
                    base_charge = h_in2[:, 3].to(torch.float).abs()
                    for _ in range(5):
                        charge_noise = torch.rand_like(base_charge)
                        sorted_scores = base_charge + charge_noise
                        h_ind_perm = lexsort([sorted_scores, batch.batch])
                        h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
                        h_ind_perm_reverse = torch.argsort(h_ind_perm)
                        mamba_arr.append(self.self_attn(h_dense)[mask][h_ind_perm_reverse])
                    h_attn = sum(mamba_arr) / 5
            elif 'Mamba_Hybrid_Radical_Atomic' == self.global_model_type:
                if batch.split == 'train':
                    radical = h[:, 5].to(torch.float)
                    atomic_num = h[:, 0].to(torch.float)
                    weights = F.softmax(self.feat_weights, dim=0)
                    combined_score = weights[0] * radical + weights[1] * atomic_num
                    noise = torch.rand_like(combined_score)
                    sorted_scores = combined_score + noise
                    h_ind_perm = lexsort([sorted_scores, batch.batch])
                    h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
                    h_ind_perm_reverse = torch.argsort(h_ind_perm)
                    h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]
                else:
                    mamba_arr = []
                    radical = h[:, 5].to(torch.float)
                    atomic_num = h[:, 0].to(torch.float)
                    weights = F.softmax(self.feat_weights, dim=0)
                    base_score = weights[0] * radical + weights[1] * atomic_num
                    for _ in range(5):
                        noise = torch.rand_like(base_score)
                        sorted_scores = base_score + noise
                        h_ind_perm = lexsort([sorted_scores, batch.batch])
                        h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
                        h_ind_perm_reverse = torch.argsort(h_ind_perm)
                        mamba_arr.append(self.self_attn(h_dense)[mask][h_ind_perm_reverse])
                    h_attn = sum(mamba_arr) / 5
            else:
                raise RuntimeError(f"Unexpected {self.global_model_type}")

            h_attn = self.dropout_attn(h_attn)
            if self.layer_norm:
                h_attn = self.norm1_attn(h_attn, batch.batch)
            if self.batch_norm:
                h_attn = self.norm1_attn(h_attn)

            # ========== 完全按照架构图实现的融合流程 ==========
            # 步骤1：特征拼接 H_local ⊕ H_global
            combined_features = torch.cat([h_local, h_attn], dim=-1)  # [N, 2*dim_h]

            # 步骤2：按分子分组，避免跨分子交互（关键修复）
            h_dense_fusion, mask_fusion = to_dense_batch(combined_features, batch.batch)  # [B, max_n, 2*dim_h]

            # 步骤3：多头自注意力融合
            h_attn_fused, _ = self.multihead_attention(
                h_dense_fusion, h_dense_fusion, h_dense_fusion,
                key_padding_mask=~mask_fusion
            )  # [B, max_n, 2*dim_h]

            # 步骤4：恢复稀疏格式 + 注意力残差
            h_attn_fused = h_attn_fused[mask_fusion]  # [N, 2*dim_h]
            h_attn_fused = self.dropout_fusion(h_attn_fused)
            h_attn_fused = h_attn_fused + combined_features  # 注意力残差

            # 步骤5：归一化
            if self.layer_norm:
                h_attn_fused = self.norm_attn_fusion(h_attn_fused, batch.batch)
            if self.batch_norm:
                h_attn_fused = self.norm_attn_fusion(h_attn_fused)

            # 步骤6：ResKAN Layer处理
            h_reskan = self.reskan_layer(h_attn_fused)  # [N, 2*dim_h]
            h_reskan = self.dropout_reskan(h_reskan)
            h_reskan = h_reskan + h_attn_fused  # KAN残差

            # 步骤7：归一化
            if self.layer_norm:
                h_reskan = self.norm_reskan(h_reskan, batch.batch)
            if self.batch_norm:
                h_reskan = self.norm_reskan(h_reskan)

            # 步骤8：KAN FFN降维
            h_ffn = self.kan_ffn(h_reskan)  # [N, dim_h]
            h_ffn = self.dropout_ffn(h_ffn)

            # 步骤9：最终残差连接 H_ffn + H_original
            h = h_ffn + h_original  # ✅ 完全匹配架构图

        # 步骤10：最终归一化
        if self.layer_norm:
            h = self.norm_final(h, batch.batch)
        if self.batch_norm:
            h = self.norm_final(h)

        batch.x = h
        return batch

    def _sa_block(self, x, attn_mask, key_padding_mask):
        """Self-attention block.
        """
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return x

    def _ff_block(self, x):
        x = self.ff_dropout1(self.activation(self.ff_kan1(x)))
        x = self.ff_dropout2(self.activation(self.ff_kan2(x)))
        return x


    def _ff_block1(self, x):
        x = self.ff_dropout1(self.activation(self.ff_kan1(x)))
        return x

    def extra_repr(self):
        s = f'summary: dim_h={self.dim_h}, ' \
            f'local_gnn_type={self.local_gnn_type}, ' \
            f'global_model_type={self.global_model_type}, ' \
            f'heads={self.num_heads}'
        return s

