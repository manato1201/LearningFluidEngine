"""
model.py  —  FluidKit Neural Fluid モデル定義
粒子の現在位置・速度 → 次フレームの位置 を予測する
Graph Neural Network ベースのモデルです。

アーキテクチャ:
    - Particle Encoder: 各粒子の局所特徴を抽出 (MLP)
    - Interaction Network: 近傍粒子との相互作用をモデル化 (edge MLP)
    - Decoder: 次フレームの位置変化量を予測 (MLP)

依存:
    pip install torch torch-scatter torch-sparse  (推奨)
    ※ torch のみでも動く簡易版 (use_gnn=False) あり
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────
#  Utilities
# ──────────────────────────────────────────

def mlp(dims: list[int], act=nn.SiLU, last_act=False) -> nn.Sequential:
    """多層パーセプトロン."""
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i+1]))
        if i < len(dims) - 2 or last_act:
            layers.append(act())
    return nn.Sequential(*layers)


def knn_graph(pos: torch.Tensor, k: int = 16):
    """
    pos: (B, N, 3)
    Returns edges: list of (src, dst) tensors per batch  → simplified: (B, N, k) indices
    """
    B, N, _ = pos.shape
    # 距離行列
    diff = pos.unsqueeze(2) - pos.unsqueeze(1)     # (B,N,N,3)
    dist = (diff ** 2).sum(-1)                      # (B,N,N)
    dist.diagonal(dim1=1, dim2=2).fill_(float('inf'))
    _, idx = dist.topk(k, dim=-1, largest=False)   # (B,N,k)
    return idx


# ──────────────────────────────────────────
#  Encoder / Decoder MLPs
# ──────────────────────────────────────────

class ParticleEncoder(nn.Module):
    """各粒子の (pos, vel) → 潜在ベクトル."""
    def __init__(self, in_dim: int = 6, hidden: int = 128, latent: int = 64):
        super().__init__()
        self.net = mlp([in_dim, hidden, hidden, latent], last_act=True)

    def forward(self, x):
        # x: (B, N, in_dim)
        return self.net(x)          # (B, N, latent)


class EdgeMLP(nn.Module):
    """エッジ特徴（2粒子の相互作用）→ メッセージ."""
    def __init__(self, in_dim: int, hidden: int = 128, out_dim: int = 64):
        super().__init__()
        self.net = mlp([in_dim, hidden, hidden, out_dim], last_act=True)

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    """粒子特徴 → 位置変化量 (Δpos)."""
    def __init__(self, latent: int = 64, hidden: int = 128, out_dim: int = 3):
        super().__init__()
        self.net = mlp([latent, hidden, hidden, out_dim])

    def forward(self, x):
        return self.net(x)          # (B, N, 3)


# ──────────────────────────────────────────
#  Main Model
# ──────────────────────────────────────────

class NeuralFluidModel(nn.Module):
    """
    Interaction Network ベースの粒子シミュレーター。

    Input:  x  (B, N, 6)  — 各粒子の [pos(3) + vel(3)]
    Output: Δpos (B, N, 3) — 次フレームの位置変化量
            → 次フレーム位置 = pos + Δpos
    """

    def __init__(self,
                 in_dim:   int = 6,
                 latent:   int = 64,
                 hidden:   int = 128,
                 k_neighbors: int = 12,
                 n_mp_steps:  int = 2):
        super().__init__()
        self.k = k_neighbors
        self.n_mp = n_mp_steps

        self.encoder = ParticleEncoder(in_dim, hidden, latent)

        # メッセージパッシング
        # エッジ入力: [hi, hj, rel_pos(3), dist(1)] → メッセージ
        edge_in = latent * 2 + 3 + 1
        self.edge_mlps = nn.ModuleList([
            EdgeMLP(edge_in, hidden, latent) for _ in range(n_mp_steps)
        ])
        self.node_mlps = nn.ModuleList([
            mlp([latent * 2, hidden, latent], last_act=True) for _ in range(n_mp_steps)
        ])

        self.decoder = Decoder(latent, hidden, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, 6)
        Returns Δpos: (B, N, 3)
        """
        B, N, _ = x.shape
        pos = x[..., :3]                     # (B,N,3)

        # Encode
        h = self.encoder(x)                  # (B,N,latent)

        # KNN グラフ構築
        nbr_idx = knn_graph(pos, self.k)     # (B,N,k)

        # Message passing
        for step in range(self.n_mp):
            # エッジ特徴を組み立て
            hi = h.unsqueeze(2).expand(-1, -1, self.k, -1)          # (B,N,k,L)
            hj = h[torch.arange(B)[:,None,None], nbr_idx]            # (B,N,k,L)

            pi = pos.unsqueeze(2).expand(-1, -1, self.k, -1)         # (B,N,k,3)
            pj = pos[torch.arange(B)[:,None,None], nbr_idx]          # (B,N,k,3)
            rel = pj - pi                                              # (B,N,k,3)
            dist = rel.norm(dim=-1, keepdim=True)                     # (B,N,k,1)

            edge_feat = torch.cat([hi, hj, rel, dist], dim=-1)        # (B,N,k,2L+4)
            msg = self.edge_mlps[step](edge_feat)                      # (B,N,k,L)
            agg = msg.sum(dim=2)                                       # (B,N,L)

            # ノード更新
            h = self.node_mlps[step](torch.cat([h, agg], dim=-1))    # (B,N,L)

        # Decode
        delta = self.decoder(h)              # (B,N,3)
        return delta

    def predict_next(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,N,6) → 次フレーム pos (B,N,3)
        """
        delta = self.forward(x)
        return x[..., :3] + delta


# ──────────────────────────────────────────
#  Loss
# ──────────────────────────────────────────

class FluidLoss(nn.Module):
    """
    位置予測誤差 + オプション: 速度・密度正則化。
    """
    def __init__(self, pos_weight=1.0, smooth_weight=0.01):
        super().__init__()
        self.w_pos    = pos_weight
        self.w_smooth = smooth_weight

    def forward(self, pred_pos: torch.Tensor, true_pos: torch.Tensor):
        """
        pred_pos, true_pos: (B, N, 3)
        """
        # 位置 MSE
        loss_pos = F.mse_loss(pred_pos, true_pos)

        # 平滑化正則化（隣接粒子間の位置差分の分散を小さく）
        diff = pred_pos[:, 1:] - pred_pos[:, :-1]
        loss_smooth = diff.pow(2).mean()

        return self.w_pos * loss_pos + self.w_smooth * loss_smooth, {
            "pos": loss_pos.item(),
            "smooth": loss_smooth.item(),
        }


# ──────────────────────────────────────────
#  Model Summary
# ──────────────────────────────────────────

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = NeuralFluidModel(in_dim=6, latent=64, hidden=128, k_neighbors=12, n_mp_steps=2)
    total = count_params(model)
    print(f"NeuralFluidModel  パラメータ数: {total:,}")

    # ダミー推論テスト
    B, N = 2, 400
    x = torch.randn(B, N, 6)
    with torch.no_grad():
        delta = model(x)
    print(f"Input:  {x.shape}  →  Δpos: {delta.shape}")

    loss_fn = FluidLoss()
    pred = x[..., :3] + delta
    target = torch.randn(B, N, 3)
    loss, breakdown = loss_fn(pred, target)
    print(f"Loss: {loss.item():.4f}  {breakdown}")
