# graphgym/models/layer/gatedgcn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.utils import (
    to_scipy_sparse_matrix,
    from_scipy_sparse_matrix,
    degree,
)
from torch_scatter import scatter
from torch_geometric.graphgym.register import register_layer
from torch_geometric.graphgym.models.layer import LayerConfig


# ========= 1. 边-边 GatedGCN（门控 + 残差） =========
class GatedGCNEdgeMini(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        d = hidden_dim
        # 门控：[e_k || e_m || h_i || h_j] -> gate
        self.W_g = nn.Linear(4 * d, d)
        self.W_m = nn.Linear(d, d)          # 邻居边映射
        self.W_o = nn.Linear(2 * d, d)      # 输出映射
        self.norm = nn.LayerNorm(d)
        self.reset_parameters()

    def reset_parameters(self):
        for m in [self.W_g, self.W_m, self.W_o]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, e, lg_edge_index, h_node, node_pair):
        """
        e            : [E, d]          当前边特征
        lg_edge_index: [2, E_lg]       line-graph 邻接表
        h_node       : [N, d]          最新节点特征
        node_pair    : [E, 2]          每条边对应的 (i,j) 节点编号
        """
        row, col = lg_edge_index                       # 中心边 <- 邻居边
        e_k, e_m = e[row], e[col]                      # [E_lg, d]

        # 1. 取出共享节点特征
        i_k, j_k = node_pair[row][:, 0], node_pair[row][:, 1]  # 中心边端点
        h_i, h_j = h_node[i_k], h_node[j_k]                    # [E_lg, d]

        # 2. 拼大向量并算门控
        z = torch.cat([e_k, e_m, h_i, h_j], dim=1)     # [E_lg, 4d]
        g = torch.sigmoid(self.W_g(z))                 # [E_lg, d]

        # 3. 加权聚合邻居边
        msg = g * self.W_m(e_m)                        # [E_lg, d]
        aggr = scatter(msg, row, dim=0, dim_size=e.size(0), reduce='sum')

        # 4. 残差更新
        out = self.norm(self.W_o(torch.cat([e, aggr], dim=1)))
        return e + F.relu(out)                         # 残差连接


# ========= 2. 原节点 GatedGCN（现在先用新边） =========
class GatedGCNLayer(pyg_nn.conv.MessagePassing):
    def __init__(self, in_dim, out_dim, dropout, residual,
                 equivstable_pe=False, aggr="add"):
        super().__init__(aggr=aggr)
        self.A = pyg_nn.Linear(in_dim, out_dim)
        self.B = pyg_nn.Linear(in_dim, out_dim)
        self.C = pyg_nn.Linear(in_dim, out_dim)
        self.D = pyg_nn.Linear(in_dim, out_dim)
        self.E = pyg_nn.Linear(in_dim, out_dim)

        self.edge_conv = GatedGCNEdgeMini(in_dim)   # ← 换成门控版
        self.bn_edge_e = nn.BatchNorm1d(out_dim)
        self.bn_node_x = nn.BatchNorm1d(out_dim)
        self.dropout = dropout
        self.residual = residual
        self.EquivStablePE = equivstable_pe
        if self.EquivStablePE:
            self.mlp_r_ij = nn.Sequential(
                nn.Linear(1, out_dim), nn.ReLU(),
                nn.Linear(out_dim, 1), nn.Sigmoid())

    # ---------- 前向：先卷边，再卷节点 ----------
    def forward(self, batch):
        x, e, edge_index = batch.x, batch.edge_attr, batch.edge_index
        if self.residual:
            x_in, e_in = x, e

        # 1. 构造 line-graph 并更新边（门控）
        lg_edge_index, _ = self._make_line_graph(edge_index, e.size(0))
        node_pair = edge_index.T                                  # [E, 2]
        e_new = self.edge_conv(e, lg_edge_index, x, node_pair)    # 边上门控
        e_new = self.bn_edge_e(e_new)
        e_new = F.relu(e_new)
        e_new = F.dropout(e_new, self.dropout, training=self.training)
        e = e_in + e_new if self.residual else e_new

        # 2. 原 GatedGCN 节点更新（用新边）
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
        x = F.dropout(x, self.dropout, training=self.training)
        x = x_in + x if self.residual else x

        batch.x = x
        batch.edge_attr = e
        return batch

    # ---------- 工具函数 ----------
    @staticmethod
    def _make_line_graph(edge_index, num_edges):
        adj = to_scipy_sparse_matrix(edge_index, num_nodes=num_edges)
        lg_adj = adj.T @ adj
        lg_adj.setdiag(0)
        lg_edge_index, _ = from_scipy_sparse_matrix(lg_adj)
        return lg_edge_index.to(edge_index.device), None

    # ---------- 以下与原库完全一致 ----------
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


# ========= 3. GraphGym 注册（名字保持原样） =========
@register_layer('gatedgcnconv')
class GatedGCNGraphGymLayer(nn.Module):
    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__()
        self.model = GatedGCNLayer(
            in_dim=layer_config.dim_in,
            out_dim=layer_config.dim_out,
            dropout=0.,
            residual=False,
            **kwargs)

    def forward(self, batch):
        return self.model(batch)