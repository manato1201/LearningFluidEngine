"""
infer_to_json.py  —  Neural Fluid 推論 → Three.js Viewer 用 JSON 出力

訓練済みモデルで初期状態から N フレーム分を自律予測し、
viewer/data/neural_XXXX.json として保存します。

使い方:
    python infer_to_json.py --checkpoint ./checkpoints/best.pt
    python infer_to_json.py --checkpoint ./checkpoints/best.pt --frames 200 --preset water_drop
"""

import json, sys, argparse, random
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
from gen_sample import SimpleSPH, PRESETS

from model import NeuralFluidModel


# ──────────────────────────────────────────
#  初期状態生成（1フレーム分のシミュレーション結果）
# ──────────────────────────────────────────

def get_initial_state(preset_name: str, n_particles: int, seed: int = 0):
    """
    Returns x: (1, N, 6) — normalized
    """
    preset = dict(PRESETS[preset_name])
    preset["particle_count"] = n_particles
    sph = SimpleSPH(preset, seed=seed)
    # 数フレーム進めて安定させる
    for _ in range(5):
        sph.step(0.016)

    pos = np.array(sph.pos[:n_particles], dtype=np.float32)
    vel = np.array(sph.vel[:n_particles], dtype=np.float32)

    bounds = preset["bounds"]
    pos_min = np.array([bounds[0], bounds[2], bounds[4]], dtype=np.float32)
    pos_max = np.array([bounds[1], bounds[3], bounds[5]], dtype=np.float32)
    pos_range = pos_max - pos_min

    pos_norm = (pos - pos_min) / pos_range * 2 - 1
    x = np.concatenate([pos_norm, vel], axis=-1)         # (N,6)
    return torch.tensor(x, dtype=torch.float32).unsqueeze(0), pos_min, pos_max


# ──────────────────────────────────────────
#  推論ループ
# ──────────────────────────────────────────

def run_inference(model, x0: torch.Tensor, frames: int, device):
    """
    x0: (1, N, 6) normalized
    Returns: list of flat arrays [x0,y0,z0, x1,y1,z1, ...]
    """
    model.eval()
    x = x0.to(device)
    results = []
    with torch.no_grad():
        for _ in range(frames):
            next_pos = model.predict_next(x)           # (1,N,3)
            # 速度を差分で近似
            vel_approx = next_pos - x[..., :3]
            x = torch.cat([next_pos, vel_approx], dim=-1)
            results.append(next_pos.squeeze(0).cpu().numpy())   # (N,3)
    return results


# ──────────────────────────────────────────
#  Denormalize
# ──────────────────────────────────────────

def denorm(pos_norm: np.ndarray, pos_min, pos_max):
    pos_range = pos_max - pos_min
    return (pos_norm + 1) / 2 * pos_range + pos_min


# ──────────────────────────────────────────
#  Main
# ──────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--frames",     type=int, default=120)
    ap.add_argument("--preset",     default="water_drop")
    ap.add_argument("--seed",       type=int, default=99)
    ap.add_argument("--output",     default=None)
    ap.add_argument("--latent",     type=int, default=64)
    ap.add_argument("--hidden",     type=int, default=128)
    ap.add_argument("--k",          type=int, default=12)
    ap.add_argument("--mp-steps",   type=int, default=2)
    args = ap.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"[ERROR] チェックポイントが見つかりません: {ckpt_path}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[infer] device={device}  preset={args.preset}  frames={args.frames}")

    # モデル復元
    ckpt = torch.load(ckpt_path, map_location=device)
    model = NeuralFluidModel(in_dim=6, latent=args.latent,
                             hidden=args.hidden, k_neighbors=args.k,
                             n_mp_steps=args.mp_steps)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)

    # 初期状態
    preset = PRESETS[args.preset]
    n_particles = min(preset["particle_count"], 400)
    x0, pos_min, pos_max = get_initial_state(args.preset, n_particles, args.seed)

    # 推論
    norm_frames = run_inference(model, x0, args.frames, device)

    # 逆正規化 → flat JSON
    frame_data = []
    for nf in norm_frames:
        pos_real = denorm(nf, pos_min, pos_max)           # (N,3)
        flat = [round(float(v), 4) for row in pos_real for v in row]
        frame_data.append(flat)

    bounds = preset["bounds"]
    meta = {
        "preset": f"neural_{args.preset}",
        "description": f"Neural Fluid (NeuralFluidModel) — {args.preset}",
        "frameCount": args.frames,
        "particleCount": n_particles,
        "fps": 30,
        "checkpoint": str(ckpt_path),
        "bounds": {
            "min": [bounds[0], bounds[2], bounds[4]],
            "max": [bounds[1], bounds[3], bounds[5]],
        },
    }

    out = Path(args.output) if args.output else \
          Path(__file__).parent.parent / "viewer" / "data" / f"neural_{args.preset}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"metadata": meta, "frames": frame_data}, f, separators=(",", ":"))

    size_kb = out.stat().st_size / 1024
    print(f"[infer] saved → {out}  ({size_kb:.1f} KB)")
    print("  → Three.js Viewer で読み込めます！")


if __name__ == "__main__":
    main()
