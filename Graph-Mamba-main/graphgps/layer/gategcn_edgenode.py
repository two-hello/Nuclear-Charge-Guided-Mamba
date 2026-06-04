import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.transforms import LineGraph
from torch_geometric.utils import to_scipy_sparse_matrix, from_scipy_sparse_matrix
from torch_scatter import scatter
from torch_geometric.graphgym.register import register_layer
from torch_geometric.graphgym.models.layer import LayerConfig

# ---------- 1. 边⇄边的小模块 ----------
class EdgeConv(nn.Module):
    """对line-graph做一轮简单GCN，只更新边节点"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.zeros_(self.W.bias)

    def forward(self, x_edge, edge_index_lg):
        # 标准GCN：x = x + ReLU(W(x))
        row, col = edge_index_lg
        deg = pyg_nn.degree(row, num_nodes=x_edge.size(0), dtype=x_edge.dtype)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        out = scatter(x_edge[col] * norm.view(-1, 1), row, dim=0, reduce='add')
        return F.relu(self.W(out)) + x_edge          # 残差


# ---------- 2. 原GatedGCNLayer改成双更新 ----------
class GatedGCNEdgeNodeLayer(pyg_nn.conv.MessagePassing):
    def __init__(self, in_dim, out_dim, dropout, residual,
                 equivstable_pe=False, aggr="add"):
        super().__init__(aggr=aggr)
        # 原节点分支
        self.A = pyg_nn.Linear(in_dim, out_dim)
        self.B = pyg_nn.Linear(in_dim, out_dim)
        self.C = pyg_nn.Linear(in_dim, out_dim)
        self.D = pyg_nn.Linear(in_dim, out_dim)
        self.E = pyg_nn.Linear(in_dim, out_dim)

        # 边分支
        self.edge_conv = EdgeConv(in_dim)
        self.bn_edge_e = nn.BatchNorm1d(out_dim)
        self.bn_node_x = nn.BatchNorm1d(out_dim)
        self.dropout = dropout
        self.residual = residual
        self.EquivStablePE = equivstable_pe
        if self.EquivStablePE:
            self.mlp_r_ij = nn.Sequential(
                nn.Linear(1, out_dim), nn.ReLU(),
                nn.Linear(out_dim, 1), nn.Sigmoid())

    def forward(self, batch):
        x, e, edge_index = batch.x, batch.edge_attr, batch.edge_index
        if self.residual:
            x_in, e_in = x, e

        # ===== 1. 先更新边（边⇄边） =====
        # 构造line-graph：边→节点
        lg_edge_index, lg_edge_attr = self._make_line_graph(edge_index, e.size(0))
        e_new = self.edge_conv(e, lg_edge_index)          # [E, d]
        e_new = self.bn_edge_e(e_new)
        e_new = F.relu(e_new)
        e_new = F.dropout(e_new, self.dropout, self.training)
        e = e_in + e_new if self.residual else e_new

        # ===== 2. 原GatedGCN节点更新（用新边） =====
        Ax = self.A(x)
        Bx = self.B(x)
        Ce = self.C(e)
        Dx = self.D(x)
        Ex = self.E(x)

        pe_LapPE = batch.pe_EquivStableLapPE if self.EquivStablePE else None
        x, _ = self.propagate(edge_index,
                              Bx=Bx, Dx=Dx, Ex=Ex, Ce=Ce,
                              e=e, Ax=Ax, PE=pe_LapPE)

        x = self.bn_node_x(x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, self.training)
        x = x_in + x if self.residual else x

        batch.x = x
        batch.edge_attr = e
        return batch

    # ---------- 工具：把原图变line-graph ----------
    @staticmethod
    def _make_line_graph(edge_index, num_edges):
        # 用CPU稀疏矩阵最快
        adj = to_scipy_sparse_matrix(edge_index, num_nodes=num_edges)
        lg_adj = adj.T @ adj                   # 两条边相邻 ⇔ 共享节点
        lg_adj.setdiag(0)                      # 去掉自环
        lg_edge_index, _ = from_scipy_sparse_matrix(lg_adj)
        return lg_edge_index.to(edge_index.device), None

    # ---------- 以下与原GatedGCN完全一致 ----------
    def message(self, Dx_i, Ex_j, PE_i, PE_j, Ce):
        e_ij = Dx_i + Ex_j + Ce
        sigma_ij = torch.sigmoid(e_ij)
        if self.EquivStablePE:
            r_ij = ((PE_i - PE_j)**2).sum(dim=-1, keepdim=True)
            r_ij = self.mlp_r_ij(r_ij)
            sigma_ij = sigma_ij * r_ij
        self.e = e_ij
        return sigma_ij

    def aggregate(self, sigma_ij, index, Bx_j, Bx):
        dim_size = Bx.shape[0]
        sum_sig_x = sigma_ij * Bx_j
        numerator = scatter(sum_sig_x, index, 0, None, dim_size, reduce='sum')
        den_sig = scatter(sigma_ij, index, 0, None, dim_size, reduce='sum')
        return numerator / (den_sig + 1e-6)

    def update(self, aggr_out, Ax):
        x = Ax + aggr_out
        return x, self.e


# ---------- GraphGym 注册 ----------
@register_layer('gatedgcn_edgenode')
class GatedGCNEdgeNodeGraphGymLayer(nn.Module):
    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__()
        self.model = GatedGCNEdgeNodeLayer(
            in_dim=layer_config.dim_in,
            out_dim=layer_config.dim_out,
            dropout=0.,
            residual=False,
            **kwargs)

    def forward(self, batch):
        return self.model(batch)