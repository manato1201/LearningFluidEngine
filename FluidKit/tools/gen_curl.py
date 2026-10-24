"""
gen_curl.py  —  カールノイズによる軽量な擬似流体 (煙) の viewer 用 JSON 生成

solver/curl_noise.py の「発散しない速度場」で粒子を流すだけの近似です。
圧力・粘性・保存則は解きません (見た目用)。物理的に厳密な結果が必要なときは
gen_sample.py (SPH) など別の手法を使ってください。

使い方:
    python gen_curl.py                       # viewer/data/curl_smoke.json
    python gen_curl.py --frames 150 --particles 3000 --gzip
"""

import json, argparse, sys, gzip
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "solver"))
from curl_noise import CurlNoise3D

BOUNDS = (-1.5, 1.5, 0.0, 4.0, -1.5, 1.5)   # gen_sample の smoke プリセットと同じ
RISE = 0.6                                    # 浮力の代わりの一定の上昇速度


def simulate(frames: int, n: int, dt: float, seed: int, strength: float):
    rng = np.random.RandomState(seed)
    lo = np.array([BOUNDS[0], BOUNDS[2], BOUNDS[4]])
    hi = np.array([BOUNDS[1], BOUNDS[3], BOUNDS[5]])

    field = CurlNoise3D(seed=seed, base_freq=0.4, strength=strength)
    # 床付近の広がった発生源 (gen_sample の smoke_source と同じ範囲)
    pos = np.stack([
        rng.uniform(lo[0] * 0.6, hi[0] * 0.6, n),
        rng.uniform(lo[1], lo[1] + 0.4, n),
        rng.uniform(lo[2] * 0.6, hi[2] * 0.6, n),
    ], axis=1)

    def vel(p, t):
        v = field.velocity(p, t)
        v[:, 1] += RISE
        return v

    frames_out, speeds_out = [], []
    for f in range(frames):
        t = f * dt
        v1 = vel(pos, t)
        frames_out.append([round(float(x), 4) for x in pos.ravel()])
        speeds_out.append([round(float(s), 4) for s in np.linalg.norm(v1, axis=1)])
        # 中点法 (RK2)
        mid = pos + 0.5 * dt * v1
        pos = pos + dt * vel(mid, t + 0.5 * dt)
        # 上に抜けた粒子は発生源に戻す。側面は反射
        out = pos[:, 1] > hi[1]
        pos[out, 0] = rng.uniform(lo[0] * 0.6, hi[0] * 0.6, out.sum())
        pos[out, 1] = rng.uniform(lo[1], lo[1] + 0.4, out.sum())
        pos[out, 2] = rng.uniform(lo[2] * 0.6, hi[2] * 0.6, out.sum())
        pos = np.clip(pos, lo, hi)
    return frames_out, speeds_out


def main():
    ap = argparse.ArgumentParser(description="カールノイズ擬似流体の viewer 用 JSON 生成")
    ap.add_argument("--frames",    type=int,   default=120)
    ap.add_argument("--particles", type=int,   default=1500)
    ap.add_argument("--dt",        type=float, default=0.016)
    ap.add_argument("--seed",      type=int,   default=42)
    ap.add_argument("--strength",  type=float, default=1.0)
    ap.add_argument("--output",    default=None)
    ap.add_argument("--gzip",      action="store_true", help="<name>.json.gz として圧縮出力")
    args = ap.parse_args()

    out = Path(args.output) if args.output else \
          Path(__file__).parent.parent / "viewer" / "data" / "curl_smoke.json"

    frames, speeds = simulate(args.frames, args.particles, args.dt, args.seed, args.strength)
    meta = {
        "preset": "curl_smoke",
        "method": "CURL",
        "description": "カールノイズの擬似流体 (近似・保存則なし)",
        "frameCount": args.frames,
        "particleCount": args.particles,
        "fps": round(1.0 / args.dt),
        "dt": args.dt,
        "bounds": {"min": [BOUNDS[0], BOUNDS[2], BOUNDS[4]],
                   "max": [BOUNDS[1], BOUNDS[3], BOUNDS[5]]},
    }
    payload = {"metadata": meta, "frames": frames, "speeds": speeds}

    out.parent.mkdir(parents=True, exist_ok=True)
    if args.gzip:
        out = Path(str(out) + ".gz") if out.suffix == ".json" else out
        with gzip.open(out, "wt", encoding="utf-8") as f:
            json.dump(payload, f, separators=(",", ":"))
    else:
        with open(out, "w") as f:
            json.dump(payload, f, separators=(",", ":"))
    print(f"[FluidKit] saved -> {out}  ({out.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
