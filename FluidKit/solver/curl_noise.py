"""
curl_noise.py — カールノイズによる「発散しない速度場」(擬似流体)

ノイズ (ポテンシャル) の回転 (curl) を速度場とすると、∇·(∇×A) = 0 が恒等的に成り立つため
圧力投影なしで非圧縮の流れが得られる。物理方程式 (ナビエ–ストークス) は解かない
**見た目用の近似**であり、境界での反射・圧力・粘性は扱わない。

ノイズは「ランダムな波数の正弦波の和」(Fourier 特徴) で作る。解析的に微分できるので、
curl も解析式で求まり、発散が厳密にゼロになる (有限差分で測った残差は離散化誤差のみ)。

  2D: ψ(x,y) の流れ関数 → u = ∂ψ/∂y, v = -∂ψ/∂x
  3D: ベクトルポテンシャル A = (A1,A2,A3) → u = ∇×A

参考: Bridson et al., "Curl-Noise for Procedural Fluid Flow", SIGGRAPH 2007 (考え方のみ)
"""

from __future__ import annotations

import numpy as np


class _FourierNoise:
    """スカラー場 f(x,t) = Σ a_k sin(k·x + φ_k + ω_k t) と、その勾配 (解析式)。"""

    def __init__(self, dim: int, rng: np.random.RandomState, octaves: int,
                 base_freq: float, lacunarity: float, persistence: float,
                 modes_per_octave: int, drift: float):
        ks, amps = [], []
        for o in range(octaves):
            freq = base_freq * lacunarity ** o          # 周期数 / 単位長
            for _ in range(modes_per_octave):
                direction = rng.normal(size=dim)
                direction /= np.linalg.norm(direction)
                ks.append(2.0 * np.pi * freq * direction)
                # 速度 ≈ |k|·a が 1 オクターブごとに persistence 倍になるよう a ∝ p^o / |k|
                amps.append(persistence ** o / (2.0 * np.pi * freq))
        self.k = np.asarray(ks)                              # (M, dim)
        self.a = np.asarray(amps) / np.sqrt(modes_per_octave)
        self.phi = rng.uniform(0.0, 2.0 * np.pi, size=len(ks))
        self.omega = rng.uniform(-drift, drift, size=len(ks))

    def grad(self, p: np.ndarray, t: float) -> np.ndarray:
        """∇f を (N, dim) で返す。"""
        phase = p @ self.k.T + self.phi + self.omega * t     # (N, M)
        return np.cos(phase) @ (self.k * self.a[:, None])


class CurlNoise2D:
    """2D カールノイズ。velocity(points[N,2], t) → (N,2)。"""

    def __init__(self, seed: int = 0, octaves: int = 3, base_freq: float = 0.5,
                 lacunarity: float = 2.0, persistence: float = 0.5,
                 modes_per_octave: int = 6, drift: float = 0.5, strength: float = 1.0):
        rng = np.random.RandomState(seed)
        self._psi = _FourierNoise(2, rng, octaves, base_freq, lacunarity,
                                  persistence, modes_per_octave, drift)
        self.strength = strength

    def velocity(self, p: np.ndarray, t: float = 0.0) -> np.ndarray:
        g = self._psi.grad(np.asarray(p, dtype=np.float64), t)   # (∂ψ/∂x, ∂ψ/∂y)
        return self.strength * np.stack([g[:, 1], -g[:, 0]], axis=1)


class CurlNoise3D:
    """3D カールノイズ (ベクトルポテンシャルの curl)。velocity(points[N,3], t) → (N,3)。"""

    def __init__(self, seed: int = 0, octaves: int = 3, base_freq: float = 0.5,
                 lacunarity: float = 2.0, persistence: float = 0.5,
                 modes_per_octave: int = 6, drift: float = 0.5, strength: float = 1.0):
        rng = np.random.RandomState(seed)
        self._A = [_FourierNoise(3, rng, octaves, base_freq, lacunarity,
                                 persistence, modes_per_octave, drift) for _ in range(3)]
        self.strength = strength

    def velocity(self, p: np.ndarray, t: float = 0.0) -> np.ndarray:
        p = np.asarray(p, dtype=np.float64)
        g1, g2, g3 = (a.grad(p, t) for a in self._A)             # 各 (N,3) = ∇A_i
        u = g3[:, 1] - g2[:, 2]
        v = g1[:, 2] - g3[:, 0]
        w = g2[:, 0] - g1[:, 1]
        return self.strength * np.stack([u, v, w], axis=1)


def divergence_residual(field, lo, hi, n: int = 24, h: float = 1e-4, t: float = 0.0):
    """
    領域 [lo, hi] 内のランダム点で中心差分の発散を測る。

    Returns: (max|div|, rms|∂u_i/∂x_j|) — 後者は「発散が小さい」の基準にする速度勾配の大きさ。
    中心差分の誤差は O(h²) なので、h を小さくしすぎると丸め誤差が支配的になる点に注意。
    """
    lo, hi = np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
    dim = len(lo)
    rng = np.random.RandomState(12345)
    pts = rng.uniform(lo, hi, size=(n * n, dim))
    div = np.zeros(len(pts))
    grad_sq = 0.0
    for j in range(dim):
        e = np.zeros(dim)
        e[j] = h
        dudx = (field.velocity(pts + e, t) - field.velocity(pts - e, t)) / (2.0 * h)
        div += dudx[:, j]
        grad_sq += float((dudx ** 2).sum())
    return float(np.abs(div).max()), float(np.sqrt(grad_sq / (len(pts) * dim * dim)))
