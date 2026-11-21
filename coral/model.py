import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, GlobalAttention, JumpingKnowledge


class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


def mlp(d_in, d_out, hidden, layers, drop=0.0):
    mods = []
    dims = [d_in] + [hidden] * (layers - 1) + [d_out]
    for i in range(len(dims) - 1):
        mods.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            mods.append(Mish())
            if drop > 0:
                mods.append(nn.Dropout(drop))
    return nn.Sequential(*mods)


class GNNEncoder(nn.Module):
    def __init__(self, node_in, edge_in, global_in, hidden, layers, dropout):
        super().__init__()
        self.proj_node = nn.Linear(node_in, hidden)
        self.convs = nn.ModuleList()
        self.edge_nns = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(layers):
            self.edge_nns.append(mlp(edge_in, hidden, hidden, 2))
            self.convs.append(GINEConv(mlp(hidden, hidden, hidden, 2)))
            self.norms.append(nn.LayerNorm(hidden))

        self.jk = JumpingKnowledge(mode="cat", channels=hidden, num_layers=layers)
        self.jk_proj = nn.Linear(hidden * layers, hidden)
        self.pool = GlobalAttention(gate_nn=nn.Linear(hidden, 1))
        self.proj_global = mlp(global_in, hidden, hidden, 2, drop=dropout)
        self.fusion = nn.Linear(hidden * 2, hidden)
        self.dropout = nn.Dropout(dropout)
        self.act = Mish()

    def forward(self, x, edge_index, edge_attr, batch, g_feat):
        h = self.proj_node(x)
        h_list = []
        for i, conv in enumerate(self.convs):
            if edge_attr is not None and edge_attr.numel() > 0:
                he = self.edge_nns[i](edge_attr)
            else:
                he = torch.zeros((edge_index.size(1), h.size(-1)), device=h.device)
            h = conv(h, edge_index, he) + h
            h = self.norms[i](h)
            h = self.act(h)
            h = self.dropout(h)
            h_list.append(h)
        h = self.jk_proj(self.jk(h_list))

        h_graph = self.pool(h, batch)

        if g_feat is not None:
            if g_feat.dim() == 1:
                g_feat = g_feat.unsqueeze(0)
            h_global = self.proj_global(g_feat)
        else:
            h_global = torch.zeros_like(h_graph)

        return self.fusion(torch.cat([h_graph, h_global], dim=-1))


class CORAL(nn.Module):
    def __init__(self, node_in, edge_in, global_in, hidden=128, layers=6, dropout=0.1):
        super().__init__()
        self.encoder = GNNEncoder(node_in, edge_in, global_in, hidden, layers, dropout)
        self.score_head = mlp(hidden, 1, hidden, 3, drop=dropout)
        self.config = {
            "node_in": node_in,
            "edge_in": edge_in,
            "global_in": global_in,
            "hidden": hidden,
            "layers": layers,
            "dropout": dropout,
        }

    def score(self, batch, device):
        dev = device if isinstance(device, torch.device) else torch.device(device)
        x = batch["x"].to(dev, non_blocking=True)
        edge_index = batch["edge_index"].to(dev, non_blocking=True)
        edge_attr = batch["edge_attr"].to(dev, non_blocking=True)
        g = batch["g"].to(dev, non_blocking=True) if "g" in batch else None
        if "batch" in batch:
            idx = batch["batch"].to(dev, non_blocking=True)
        else:
            idx = torch.zeros(x.size(0), dtype=torch.long, device=dev)
        return self.score_head(self.encoder(x, edge_index, edge_attr, idx, g))

    def forward(self, batch_a, batch_b, device):
        return self.score(batch_a, device) - self.score(batch_b, device)


def save_model(model, path, extra=None):
    payload = {"model": model.state_dict(), "config": model.config}
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_model(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = ckpt.get("config")
    state = ckpt.get("model", ckpt)
    if config is None:
        config = {
            "node_in": state["encoder.proj_node.weight"].shape[1],
            "edge_in": state["encoder.edge_nns.0.0.weight"].shape[1],
            "global_in": state["encoder.proj_global.0.weight"].shape[1],
            "hidden": state["encoder.proj_node.weight"].shape[0],
            "layers": sum(1 for k in state if k.startswith("encoder.convs.") and k.endswith(".eps")),
            "dropout": 0.1,
        }
    model = CORAL(**config).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model
