"""
infer_v2.py  —  Neural Fluid v2 推論 → Three.js Viewer 用 JSON 出力

NeuralFluidV2 を使った自律ロールアウト推論。
vel_std を meta.json から読み込み、速度も正規化して入力する。

使い方:
    python infer_v2.py --checkpoint ./checkpoints_v2/best.pt
    python infer_v2.py --checkpoint ./checkpoints_v2/best.pt --frames 120 --preset water_drop
"""

import json, sys, argparse
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
from gen_sample import SimpleSPH, PRESETS

from model_v2 import NeuralFluidV2


# ──────────────────────────────────────────
#  初期状態生成（v2: 速度も正規化）
# ──────────────────────────────────────────

def get_initial_state(preset_name: str, n_particles: int,
                      pos_min, pos_max, vel_std, seed: int = 0):
    """
    Returns x: (1, N, 6) — pos と vel を両方正規化済み
    """
    preset = dict(PRESETS[preset_name])
    preset["particle_count"] = n_particles
    sph = SimpleSPH(preset, seed=seed)

    # 数フレーム進めて安定させる
    for _ in range(5):
        sph.step(0.016)

    pos = np.array(sph.pos[:n_particles], dtype=np.float32)   # (N,3)
    vel = np.array(sph.vel[:n_particles], dtype=np.float32)   # (N,3)

    pos_range = pos_max - pos_min
    pos_norm = (pos - pos_min) / pos_range * 2 - 1             # [-1,1]
    vel_norm = vel / (vel_std * 3.0)                           # ±3σ → ±1

    x = np.concatenate([pos_norm, vel_norm], axis=-1)          # (N,6)
    return torch.tensor(x, dtype=torch.float32).unsqueeze(0)   # (1,N,6)


# ──────────────────────────────────────────
#  推論ループ（v2: モデルが pos+vel を出力）
# ──────────────────────────────────────────

def run_inference(model, x0: torch.Tensor, frames: int, device):
    """
    x0: (1, N, 6) normalized [pos_norm | vel_norm]
    Returns: list of (N,3) normalized positions
    """
    model.eval()
    x = x0.to(device)
    results = []

    with torch.no_grad():
        for _ in range(frames):
            # v2 の predict_next は (B,N,6) [next_pos_norm | next_vel_norm] を返す
            x_next = model.predict_next(x)   # (1,N,6)
            results.append(x_next[..., :3].squeeze(0).cpu().numpy())  # (N,3)
            x = x_next

    return results


# ──────────────────────────────────────────
#  逆正規化
# ──────────────────────────────────────────

def denorm_pos(pos_norm: np.ndarray, pos_min, pos_max):
    pos_range = pos_max - pos_min
    return (pos_norm + 1) / 2 * pos_range + pos_min


# ──────────────────────────────────────────
#  Main
# ──────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(Path(__file__).parent / "checkpoints_v2/best.pt"))
    ap.add_argument("--dataset",    default=str(Path(__file__).parent / "dataset_v2"))
    ap.add_argument("--frames",     type=int, default=120)
    ap.add_argument("--preset",     default="water_drop")
    ap.add_argument("--seed",       type=int, default=99)
    ap.add_argument("--output",     default=None)
    args = ap.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"[ERROR] チェックポイントが見つかりません: {ckpt_path}")
        return

    # meta.json から vel_std と境界を取得
    meta_path = Path(args.dataset) / "meta.json"
    if not meta_path.exists():
        print(f"[ERROR] meta.json が見つかりません: {meta_path}")
        return
    with open(meta_path) as f:
        meta_json = json.load(f)

    vel_std = np.array(meta_json["vel_std"], dtype=np.float32)
    pos_min = np.array(meta_json["pos_min"], dtype=np.float32)
    pos_max = np.array(meta_json["pos_max"], dtype=np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[infer_v2] device={device}  preset={args.preset}  frames={args.frames}")
    print(f"  vel_std: {vel_std.tolist()}")

    # モデル復元
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg  = ckpt.get("config", {})
    model = NeuralFluidV2(
        latent      = cfg.get("latent", 128),
        hidden      = cfg.get("hidden", 256),
        k_neighbors = cfg.get("k",      16),
        n_mp_steps  = cfg.get("mp",     3),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  パラメータ数: {n_params:,}")

    # 初期状態（正規化済み）
    preset  = PRESETS[args.preset]
    n_particles = min(preset["particle_count"], meta_json.get("n_particles", 400))
    x0 = get_initial_state(args.preset, n_particles, pos_min, pos_max, vel_std, args.seed)

    # 推論
    norm_frames = run_inference(model, x0, args.frames, device)

    # 逆正規化 → flat JSON
    frame_data = []
    for nf in norm_frames:
        pos_real = denorm_pos(nf, pos_min, pos_max)       # (N,3)
        flat = [round(float(v), 4) for row in pos_real for v in row]
        frame_data.append(flat)

    bounds = preset["bounds"]
    out_meta = {
        "preset":        f"neural_v2_{args.preset}",
        "description":   f"Neural Fluid v2 (NeuralFluidV2) — {args.preset}",
        "frameCount":    args.frames,
        "particleCount": n_particles,
        "fps":           30,
        "checkpoint":    str(ckpt_path),
        "best_val":      ckpt.get("best_val", None),
        "bounds": {
            "min": [bounds[0], bounds[2], bounds[4]],
            "max": [bounds[1], bounds[3], bounds[5]],
        },
    }

    out = Path(args.output) if args.output else \
          Path(__file__).parent.parent / "viewer" / "data" / f"neural_v2_{args.preset}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"metadata": out_meta, "frames": frame_data}, f, separators=(",", ":"))

    size_kb = out.stat().st_size / 1024
    print(f"[infer_v2] 保存完了 → {out}  ({size_kb:.1f} KB)")
    print("  → Three.js Viewer で読み込んで確認してください！")


if __name__ == "__main__":
    main()
