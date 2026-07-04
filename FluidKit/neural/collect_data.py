"""
collect_data.py  —  FluidKit Neural Fluid 学習データ収集
複数のパラメータ設定でシミュレーションを走らせて
NumPy 形式の学習データセットを生成します。

出力フォーマット:
    dataset/
    ├── X_train.npy     # shape (N, n_particles, 6)  [x,y,z, vx,vy,vz] at frame t
    ├── Y_train.npy     # shape (N, n_particles, 3)  [x,y,z] at frame t+1
    ├── NBR_train.npy   # shape (N, n_particles, k)  事前計算済み k近傍インデックス（cKDTree）
    ├── X_val.npy
    ├── Y_val.npy
    ├── NBR_val.npy
    └── meta.json

使い方:
    python collect_data.py                         # デフォルト設定
    python collect_data.py --simulations 30 --frames 100
    python collect_data.py --output ./my_dataset
"""

import sys, json, math, random, argparse, time
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree

# gen_sample.py の SimpleSPH を流用
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
from gen_sample import SimpleSPH, PRESETS

# 正規化/逆正規化の共通ロジック
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import normalize_pos, normalize_vel

# model_v2.py の NeuralFluidV2 デフォルト k と一致させる
K_NEIGHBORS = 16


def compute_knn(pos: np.ndarray, k: int = K_NEIGHBORS) -> np.ndarray:
    """cKDTree で k近傍インデックスを事前計算する。

    pos: (N,3) numpy 配列（GPU テンソル不可 — 事前に .cpu().numpy() する）
    戻り値: (N,k) int64 インデックス。自分自身（距離0）は必ず除外する。
    """
    tree = cKDTree(pos)
    _, idx = tree.query(pos, k=k + 1)   # 自分自身を含め k+1 個取得
    idx = idx[:, 1:]                     # 先頭（自分自身, 距離0）を除外
    return idx.astype(np.int64)


# ──────────────────────────────────────────
#  パラメータ空間の定義
# ──────────────────────────────────────────

PARAM_SPACE = {
    # (min, max) で一様ランダムサンプリング
    "gravity_y":   (-15.0, -5.0),
    "stiffness":   (40.0,  150.0),
    "viscosity":   (0.002, 0.05),
    "rest_density": (800.0, 1200.0),
}

BASE_PRESET = "water_drop"


def sample_preset(seed: int) -> dict:
    """ランダムなパラメータでプリセットを生成."""
    rng = random.Random(seed)
    p = dict(PRESETS[BASE_PRESET])  # base copy
    p["particle_count"] = 400       # 軽量化（学習用）
    p["gravity"] = (0.0,
                    rng.uniform(*PARAM_SPACE["gravity_y"]),
                    0.0)
    p["stiffness"]    = rng.uniform(*PARAM_SPACE["stiffness"])
    p["viscosity"]    = rng.uniform(*PARAM_SPACE["viscosity"])
    p["rest_density"] = rng.uniform(*PARAM_SPACE["rest_density"])
    return p, {
        "seed": seed,
        "gravity_y":    p["gravity"][1],
        "stiffness":    p["stiffness"],
        "viscosity":    p["viscosity"],
        "rest_density": p["rest_density"],
    }


# ──────────────────────────────────────────
#  1シミュレーション → (X, Y) ペア生成
# ──────────────────────────────────────────

def run_simulation(preset: dict, frames: int, dt: float, n_particles: int, k: int = K_NEIGHBORS):
    """
    Returns:
        X:   (frames-1, n_particles, 6)  [pos + vel]
        Y:   (frames-1, n_particles, 3)  [next pos]
        NBR: (frames-1, n_particles, k)  現在フレーム(t)の pos から計算した近傍インデックス
    """
    sph = SimpleSPH(preset, seed=preset.get("_seed", 0))

    snapshots_pos = []
    snapshots_vel = []

    for _ in range(frames):
        # positions
        p = np.array(sph.pos[:n_particles], dtype=np.float32)      # (N,3)
        v = np.array(sph.vel[:n_particles], dtype=np.float32)      # (N,3)
        snapshots_pos.append(p)
        snapshots_vel.append(v)
        sph.step(dt)

    X, Y, NBR = [], [], []
    for t in range(frames - 1):
        pv = np.concatenate([snapshots_pos[t], snapshots_vel[t]], axis=-1)  # (N,6)
        X.append(pv)
        Y.append(snapshots_pos[t + 1])                                       # (N,3)
        NBR.append(compute_knn(snapshots_pos[t], k))                         # (N,k)

    return (np.stack(X, axis=0),      # (T,N,6)
            np.stack(Y, axis=0),      # (T,N,3)
            np.stack(NBR, axis=0))    # (T,N,k)


# ──────────────────────────────────────────
#  Main
# ──────────────────────────────────────────

def collect(n_sims: int, frames: int, dt: float, val_ratio: float,
            output: Path, seed: int, k: int = K_NEIGHBORS):

    output.mkdir(parents=True, exist_ok=True)

    n_particles = PRESETS[BASE_PRESET]["particle_count"]
    # 軽量化
    n_particles = min(n_particles, 400)

    all_X, all_Y, all_NBR = [], [], []
    param_log = []

    t0 = time.time()
    for sim_i in range(n_sims):
        s = seed + sim_i
        preset, params = sample_preset(s)
        preset["_seed"] = s

        print(f"  [{sim_i+1:3d}/{n_sims}] seed={s}  "
              f"grav_y={params['gravity_y']:.2f}  "
              f"stiff={params['stiffness']:.1f}  "
              f"visc={params['viscosity']:.4f}", end=" ")

        X, Y, NBR = run_simulation(preset, frames, dt, n_particles, k)
        all_X.append(X)
        all_Y.append(Y)
        all_NBR.append(NBR)
        param_log.append(params)

        elapsed = time.time() - t0
        eta = elapsed / (sim_i + 1) * (n_sims - sim_i - 1)
        print(f"→ X{X.shape}  ETA {eta:.0f}s")

    # 結合
    X_all   = np.concatenate(all_X, axis=0)     # (total_frames, N, 6)
    Y_all   = np.concatenate(all_Y, axis=0)     # (total_frames, N, 3)
    NBR_all = np.concatenate(all_NBR, axis=0)   # (total_frames, N, k)
    print(f"\n[collect] total samples: {len(X_all)}")

    # 正規化（bounds で -1〜1 に）
    bounds = PRESETS[BASE_PRESET]["bounds"]
    pos_min = np.array([bounds[0], bounds[2], bounds[4]], dtype=np.float32)
    pos_max = np.array([bounds[1], bounds[3], bounds[5]], dtype=np.float32)

    # 速度の正規化: 全サンプルの速度成分の標準偏差でスケール
    vel_std = X_all[..., 3:].std(axis=(0, 1)).clip(min=1e-6)

    X_norm = X_all.copy()
    X_norm[..., :3] = normalize_pos(X_all[..., :3], pos_min, pos_max)
    X_norm[..., 3:] = normalize_vel(X_all[..., 3:], vel_std)
    Y_norm = normalize_pos(Y_all, pos_min, pos_max)

    # train / val split
    n_val = max(1, int(len(X_norm) * val_ratio))
    idx = np.random.default_rng(seed).permutation(len(X_norm))
    val_idx   = idx[:n_val]
    train_idx = idx[n_val:]

    np.save(output / "X_train.npy",   X_norm[train_idx])
    np.save(output / "Y_train.npy",   Y_norm[train_idx])
    np.save(output / "NBR_train.npy", NBR_all[train_idx])
    np.save(output / "X_val.npy",     X_norm[val_idx])
    np.save(output / "Y_val.npy",     Y_norm[val_idx])
    np.save(output / "NBR_val.npy",   NBR_all[val_idx])

    meta = {
        "n_simulations": n_sims,
        "frames_per_sim": frames - 1,
        "total_samples": int(len(X_norm)),
        "train_samples": int(len(train_idx)),
        "val_samples":   int(len(val_idx)),
        "n_particles": n_particles,
        "input_dim": 6,    # [x,y,z, vx,vy,vz]
        "output_dim": 3,   # [x,y,z]
        "k_neighbors": k,  # NBR_*.npy の近傍数（cKDTree 事前計算, model_v2.py の k と一致させる）
        "pos_min": pos_min.tolist(),
        "pos_max": pos_max.tolist(),
        "vel_std": vel_std.tolist(),
        "dt": dt,
        "base_preset": BASE_PRESET,
        "param_space": PARAM_SPACE,
        "simulations": param_log,
    }
    with open(output / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[collect] saved to {output}/")
    print(f"  X_train:   {X_norm[train_idx].shape}")
    print(f"  Y_train:   {Y_norm[train_idx].shape}")
    print(f"  NBR_train: {NBR_all[train_idx].shape}")
    print(f"  X_val:     {X_norm[val_idx].shape}")
    print(f"  Y_val:     {Y_norm[val_idx].shape}")
    print(f"  NBR_val:   {NBR_all[val_idx].shape}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--simulations", type=int,   default=20,  help="シミュレーション数")
    ap.add_argument("--frames",      type=int,   default=80,  help="1シムあたりのフレーム数")
    ap.add_argument("--dt",          type=float, default=0.016)
    ap.add_argument("--val-ratio",   type=float, default=0.15)
    ap.add_argument("--seed",        type=int,   default=0)
    ap.add_argument("--output",      default=str(Path(__file__).parent / "dataset"))
    ap.add_argument("--k",           type=int,   default=K_NEIGHBORS,
                    help="事前計算する近傍数（model_v2.py の k_neighbors と一致させること）")
    args = ap.parse_args()

    print(f"[FluidKit Neural] データ収集開始")
    print(f"  {args.simulations} シミュレーション × {args.frames} フレーム  k={args.k}")
    collect(args.simulations, args.frames, args.dt,
            args.val_ratio, Path(args.output), args.seed, args.k)


if __name__ == "__main__":
    main()
