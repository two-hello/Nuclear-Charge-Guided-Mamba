# kan_layer.py
import torch
import torch.nn as nn
from torch_geometric.graphgym.register import register_layer
from torch_geometric.graphgym.config import cfg


class SimpleKAN(nn.Module):
    """你给出的 KAN 实现，仅把输入/输出维度通用化"""
    def __init__(self, input_dim: int, out_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.u_funcs = nn.ModuleList([nn.Linear(1, hidden_dim) for _ in range(input_dim)])
        self.v_funcs = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(input_dim)])
        self.w_funcs = nn.Linear(input_dim, out_dim)

    def forward(self, x):
        # x: (B, input_dim)
        u_outs = [torch.relu(u(x[:, i:i+1])) for i, u in enumerate(self.u_funcs)]
        v_outs = [v(u) for v, u in zip(self.v_funcs, u_outs)]
        stacked = torch.cat(v_outs, dim=1)   # (B, input_dim)
        return self.w_funcs(stacked)         # (B, out_dim)


class SimpleKANLayer(nn.Module):
    """
    GraphGym 要求的层接口：
    输入:  (batch, in_channels)
    输出:  (batch, out_channels)
    """
    def __init__(self, in_channels: int, out_channels: int, **kwargs):
        super().__init__()
        # 把隐藏维度暴露到 cfg 里，方便 YAML 调参
        hidden = cfg.kan.hidden_dim if hasattr(cfg, 'kan') else 64
        self.kan = SimpleKAN(in_channels, out_channels, hidden)

    def forward(self, x):
        return self.kan(x)


# 注册给 GraphGym
register_layer('kan', SimpleKANLayer)