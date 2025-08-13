import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
from tqdm import tqdm

from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, TopKPooling
from torch_geometric.utils import to_dense_adj, subgraph

# =============== SinCos 位置编码（单图） ===============
def sinusoidal_pe(num_nodes: int, dim: int, device: torch.device) -> torch.Tensor:
    pe = torch.zeros(num_nodes, dim, device=device)
    position = torch.arange(0, num_nodes, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, device=device) * -(np.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

# =============== 指标 ===============
@torch.no_grad()
def calc_metrics(pred: torch.Tensor, target: torch.Tensor) -> Tuple[float, float, float, float]:
    pred_bin = (pred > 0.5).float()
    tp = (pred_bin * target).sum()
    fp = (pred_bin * (1 - target)).sum()
    fn = ((1 - pred_bin) * target).sum()
    precision = tp / (tp + fp + 1e-6)
    recall    = tp / (tp + fn + 1e-6)
    f1        = 2 * precision * recall / (precision + recall + 1e-6)
    correct = (pred_bin == target).sum()
    total   = target.numel()
    accuracy = correct / (total + 1e-6)
    return precision.item(), recall.item(), f1.item(), accuracy.item()

# =============== KL ===============
def kl_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

# ==========================
# Encoder
# ==========================
class GCNEncoderWithPool(nn.Module):
    def __init__(self, in_dim: int, pe_dim: int, hidden_dim: int, latent_dim: int, pool_ratio: float = 0.5):
        super().__init__()
        self.gc1 = GCNConv(in_dim + pe_dim, hidden_dim)
        self.gc_mu = GCNConv(hidden_dim, latent_dim)
        self.gc_logvar = GCNConv(hidden_dim, latent_dim)
        self.pool = TopKPooling(latent_dim, ratio=pool_ratio)

    def forward(self, x_raw, sincos_pe, edge_index, batch=None):
        if batch is None:
            batch = x_raw.new_zeros(x_raw.size(0), dtype=torch.long)
        x_in = torch.cat([x_raw, sincos_pe], dim=-1)
        h = F.relu(self.gc1(x_in, edge_index))
        mu = self.gc_mu(h, edge_index)
        logvar = self.gc_logvar(h, edge_index)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        z_pool, edge_index_pool, _, batch_pool, perm, _ = self.pool(z, edge_index, batch=batch)
        mu_pool = mu[perm]
        logvar_pool = logvar[perm]
        return z_pool, mu_pool, logvar_pool, edge_index_pool, batch_pool

# ==========================
# Decoder
# ==========================
class TransformerDecoderCross(nn.Module):
    def __init__(self, latent_dim: int, pe_dim: int = 8, num_heads: int = 4, num_layers: int = 2, feat_dim: int = 11):
        super().__init__()
        self.input_proj = nn.Linear(pe_dim + feat_dim, latent_dim)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=latent_dim, nhead=num_heads,
            dim_feedforward=latent_dim * 2, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)
        self.edge_mlp = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, mask_nodes, sincos_pe, h_pool):
        q_dec = torch.cat([mask_nodes, sincos_pe], dim=-1)
        q_proj = self.input_proj(q_dec).unsqueeze(0)
        kv = h_pool.unsqueeze(0)
        h_dec = self.decoder(q_proj, kv).squeeze(0)
        N, D = h_dec.shape
        hi = h_dec.unsqueeze(1).expand(N, N, D)
        hj = h_dec.unsqueeze(0).expand(N, N, D)
        pair = torch.cat([hi, hj], dim=-1).reshape(N * N, 2 * D)
        edge_prob = self.edge_mlp(pair).reshape(N, N)
        diag_mask = 1.0 - torch.eye(N, device=edge_prob.device)
        edge_prob = edge_prob * diag_mask
        return edge_prob

# ==========================
# GraphVAE
# ==========================
class GraphVAEPooled(nn.Module):
    def __init__(self, feat_dim, pe_dim, hidden_dim, latent_dim, pool_ratio=0.5):
        super().__init__()
        self.encoder = GCNEncoderWithPool(feat_dim, pe_dim, hidden_dim, latent_dim, pool_ratio)
        self.decoder = TransformerDecoderCross(latent_dim, pe_dim=pe_dim, feat_dim=feat_dim)

    def forward(self, x_raw, edge_index, sincos_pe, batch=None):
        h_pool, mu_pool, logvar_pool, _, _ = self.encoder(x_raw, sincos_pe, edge_index, batch=batch)
        mask_nodes = x_raw.new_zeros(x_raw.size(0), x_raw.size(1))
        edge_prob = self.decoder(mask_nodes, sincos_pe, h_pool)
        return edge_prob, mu_pool, logvar_pool

# ==========================
# Train
# ==========================
def train_model(epochs=200, batch_size=16, pe_dim=8, hidden_dim=64, latent_dim=64, pool_ratio=0.5, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    dataset = QM9(root="data/QM9")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = GraphVAEPooled(11, pe_dim, hidden_dim, latent_dim, pool_ratio).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        p_sum = r_sum = f1_sum = acc_sum = 0.0
        count_graphs = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}")
        for data in pbar:
            data = data.to(device)
            x_all, edge_index_all = data.x, data.edge_index
            batch_vec = getattr(data, "batch", None)
            num_graphs = int(data.num_graphs)
            loss_batch = 0.0
            p_b = r_b = f1_b = acc_b = 0.0

            for gid in range(num_graphs):
                node_mask = (batch_vec == gid) if batch_vec is not None else torch.ones(x_all.size(0), dtype=torch.bool, device=device)
                node_idx = node_mask.nonzero(as_tuple=False).view(-1)
                if node_idx.numel() == 0:
                    continue
                edge_index_g, _ = subgraph(node_idx, edge_index_all, relabel_nodes=True)
                x_g = x_all[node_idx]
                N = x_g.size(0)
                pe_g = sinusoidal_pe(N, pe_dim, device=device)
                A_pred, mu_p, logvar_p = model(x_g, edge_index_g, pe_g, batch=None)
                adj_gt = to_dense_adj(edge_index_g, max_num_nodes=N, batch=None, edge_attr=None)[0].to(device)
                A_pred = A_pred.clamp(1e-6, 1 - 1e-6)
                recon = F.binary_cross_entropy(A_pred, adj_gt)
                kl = kl_normal(mu_p, logvar_p)
                loss_g = recon + kl
                loss_batch += loss_g
                p, r, f1, acc = calc_metrics(A_pred.detach(), adj_gt.detach())
                p_b += p; r_b += r; f1_b += f1; acc_b += acc
                count_graphs += 1

            if count_graphs == 0:
                continue
            loss_batch = loss_batch / num_graphs
            optimizer.zero_grad()
            loss_batch.backward()
            optimizer.step()
            total_loss += loss_batch.item()
            p_sum += p_b / num_graphs
            r_sum += r_b / num_graphs
            f1_sum += f1_b / num_graphs
            acc_sum += acc_b / num_graphs
            pbar.set_postfix(loss=loss_batch.item(), f1=f1_b / num_graphs)

        denom = max(1, len(loader))
        print(f"[Epoch {epoch:03d}] loss={total_loss/denom:.4f} | "
              f"P={p_sum/denom:.4f} R={r_sum/denom:.4f} F1={f1_sum/denom:.4f} Acc={acc_sum/denom:.4f}")

if __name__ == "__main__":
    train_model()
