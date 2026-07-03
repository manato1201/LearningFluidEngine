"""
model_v2.py  —  FluidKit Neural Fluid モデル v2
=================================================
v1 からの改善点:
  - latent 64→128, hidden 128→256, mp_steps 2→3
  - 位置だけでなく速度変化量も同時予測（マルチタスク）
  - LayerNorm 追加（訓練安定化）
  - グローバル特徴（全粒子の平均状態）を各粒子に付与
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ──────────────────────────────────────────
#  ユーティリティ
# ──────────────────────────────────────────

def mlp(dims: list[int], act=nn.SiLU, last_act=False) -> nn.Sequential:
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i+1]))
        if i < len(dims) - 2:
            layers.append(act())
            layers.append(nn.LayerNorm(dims[i+1]))
        elif last_act:
            layers.append(act())
    return nn.Sequential(*layers)


def knn_graph(pos: torch.Tensor, k: int = 16):
    """O(N^2) フォールバック実装（近傍インデックス未提供時のみ使用）。

    学習/推論の通常経路では、近傍は cKDTree で事前計算し forward() に
    渡すため本関数は呼ばれない。前計算インデックスが無い場合（単体テスト等）
    のフォールバックとして残す。
    """
    B, N, _ = pos.shape
    diff = pos.unsqueeze(2) - pos.unsqueeze(1)    # (B,N,N,3)
    dist = (diff ** 2).sum(-1)                     # (B,N,N)
    dist.diagonal(dim1=1, dim2=2).fill_(float('inf'))
    _, idx = dist.topk(k, dim=-1, largest=False)
    return idx                                     # (B,N,k)


# ──────────────────────────────────────────
#  モデル構成要素
# ──────────────────────────────────────────

class ParticleEncoder(nn.Module):
    def __init__(self, in_dim=6, hidden=256, latent=128):
        super().__init__()
        self.net = mlp([in_dim, hidden, hidden, latent], last_act=True)

    def forward(self, x):
        return self.net(x)    # (B,N,L)


class GlobalContext(nn.Module):
    """全粒子の平均特徴 → グローバルコンテキスト（各粒子に付与）"""
    def __init__(self, latent=128, hidden=256):
        super().__init__()
        self.net = mlp([latent, hidden, latent], last_act=True)

    def forward(self, h: torch.Tensor):
        # h: (B,N,L)
        g = h.mean(dim=1, keepdim=True)    # (B,1,L)
        g = self.net(g)                     # (B,1,L)
        return g.expand_as(h)              # (B,N,L)


class EdgeMLP(nn.Module):
    def __init__(self, in_dim, hidden=256, out_dim=128):
        super().__init__()
        self.net = mlp([in_dim, hidden, hidden, out_dim], last_act=True)

    def forward(self, x):
        return self.net(x)


class NodeMLP(nn.Module):
    def __init__(self, in_dim, hidden=256, out_dim=128):
        super().__init__()
        self.net = mlp([in_dim, hidden, out_dim], last_act=True)

    def forward(self, x):
        return self.net(x)


# ──────────────────────────────────────────
#  メインモデル v2
# ──────────────────────────────────────────

class NeuralFluidV2(nn.Module):
    """
    Input:  x  (B, N, 6)   [pos(3) + vel(3)]
    Output: (Δpos, Δvel)  各 (B, N, 3)
            → 次フレーム pos = pos + Δpos
            → 次フレーム vel = vel + Δvel
    """

    def __init__(self,
                 in_dim: int   = 6,
                 latent: int   = 128,
                 hidden: int   = 256,
                 k_neighbors: int = 16,
                 n_mp_steps: int  = 3):
        super().__init__()
        self.k    = k_neighbors
        self.n_mp = n_mp_steps

        # エンコーダ
        self.encoder = ParticleEncoder(in_dim, hidden, latent)

        # グローバルコンテキスト
        self.global_ctx = GlobalContext(latent, hidden)

        # グローバル付与後の射影
        self.ctx_proj = mlp([latent * 2, hidden, latent], last_act=True)

        # メッセージパッシング: エッジ入力 = [hi, hj, rel_pos(3), dist(1)]
        edge_in = latent * 2 + 3 + 1
        self.edge_mlps = nn.ModuleList([
            EdgeMLP(edge_in, hidden, latent) for _ in range(n_mp_steps)
        ])
        self.node_mlps = nn.ModuleList([
            NodeMLP(latent * 2, hidden, latent) for _ in range(n_mp_steps)
        ])

        # デコーダ（位置・速度を同時予測）
        self.dec_pos = mlp([latent, hidden, 3])
        self.dec_vel = mlp([latent, hidden, 3])

    def forward(self, x: torch.Tensor, nbr: torch.Tensor = None):
        """
        x   : (B,N,6) [pos+vel]
        nbr : (B,N,k) 近傍インデックス（long tensor）。
              事前計算済みインデックス（cKDTree 由来）を渡すのが通常経路。
              None の場合のみ knn_graph() の O(N^2) フォールバックで計算する。
        """
        B, N, _ = x.shape
        pos = x[..., :3]   # (B,N,3)

        # エンコード
        h = self.encoder(x)                         # (B,N,L)

        # グローバルコンテキスト付与
        g = self.global_ctx(h)                      # (B,N,L)
        h = self.ctx_proj(torch.cat([h, g], dim=-1))# (B,N,L)

        # KNN グラフ（事前計算インデックスを利用。無い場合のみ O(N^2) フォールバック）
        if nbr is None:
            nbr = knn_graph(pos, self.k)            # (B,N,k)

        # メッセージパッシング
        for step in range(self.n_mp):
            hi = h.unsqueeze(2).expand(-1, -1, self.k, -1)          # (B,N,k,L)
            hj = h[torch.arange(B)[:,None,None], nbr]                # (B,N,k,L)

            pi = pos.unsqueeze(2).expand(-1, -1, self.k, -1)
            pj = pos[torch.arange(B)[:,None,None], nbr]
            rel  = pj - pi                                            # (B,N,k,3)
            dist = rel.norm(dim=-1, keepdim=True)                    # (B,N,k,1)

            e    = torch.cat([hi, hj, rel, dist], dim=-1)
            msg  = self.edge_mlps[step](e)                           # (B,N,k,L)
            agg  = msg.sum(dim=2)                                    # (B,N,L)
            h    = self.node_mlps[step](torch.cat([h, agg], dim=-1))# (B,N,L)

        # デコード
        d_pos = self.dec_pos(h)   # (B,N,3)
        d_vel = self.dec_vel(h)   # (B,N,3)
        return d_pos, d_vel

    def predict_next(self, x: torch.Tensor, nbr: torch.Tensor = None):
        """x: (B,N,6) → 次フレーム (B,N,6) [pos+vel]
        nbr: (B,N,k) 事前計算済み近傍インデックス（forward()参照）"""
        d_pos, d_vel = self.forward(x, nbr)
        next_pos = x[..., :3] + d_pos
        next_vel = x[..., 3:] + d_vel
        return torch.cat([next_pos, next_vel], dim=-1)


# ──────────────────────────────────────────
#  損失関数 v2（位置 + 速度 + 運動量保存）
# ──────────────────────────────────────────

class FluidLossV2(nn.Module):
    """
    v3修正版: 速度損失を廃止し位置損失のみに統一。
    pred と target_y はどちらも正規化位置空間（-1〜1）で比較するため
    スケール不一致が生じない。
    smoothness 項で隣接粒子の急激な位置変化を抑制。
    """
    def __init__(self, w_pos=1.0, w_smooth=0.05):
        super().__init__()
        self.w_pos    = w_pos
        self.w_smooth = w_smooth

    def forward(self, pred: torch.Tensor, target_x: torch.Tensor, target_y: torch.Tensor):
        """
        pred     : (B,N,6) 次フレーム予測 [pos+vel]  (vel は参考値として出力のみ)
        target_x : (B,N,6) 現在フレーム入力
        target_y : (B,N,3) 次フレーム正解 pos（正規化済み）
        """
        pred_pos = pred[..., :3]
        true_pos = target_y

        loss_pos    = F.mse_loss(pred_pos, true_pos)
        # 粒子順方向のスムースネス（粒子配列が急変しないように）
        loss_smooth = (pred_pos[:, 1:] - pred_pos[:, :-1]).pow(2).mean()

        total = self.w_pos * loss_pos + self.w_smooth * loss_smooth

        return total, {
            "pos":    loss_pos.item(),
            "smooth": loss_smooth.item(),
        }


# ──────────────────────────────────────────
#  確認
# ──────────────────────────────────────────

def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = NeuralFluidV2()
    n_p = count_params(model)
    print(f"NeuralFluidV2  パラメータ数: {n_p:,}")

    B, N = 2, 400
    x = torch.randn(B, N, 6)

    # 事前計算済み近傍インデックス（本来は cKDTree 由来）を模擬
    nbr = knn_graph(x[..., :3], model.k)
    d_pos, d_vel = model(x, nbr)
    print(f"Input {x.shape} → Δpos {d_pos.shape}, Δvel {d_vel.shape}")

    loss_fn = FluidLossV2()
    pred = model.predict_next(x, nbr)
    target_y = torch.randn(B, N, 3)
    loss, bd  = loss_fn(pred, x, target_y)
    print(f"Loss: {loss.item():.4f}  {bd}")
