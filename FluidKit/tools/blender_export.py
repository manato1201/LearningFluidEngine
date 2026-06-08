"""
blender_export.py  —  FluidKit → Blender エクスポーター

XYZ パーティクルデータを Blender が高速に読める .npz 形式に変換します。
間引き・正規化・バウンディングボックス情報も付与。

使い方:
    python blender_export.py --input ./sph_output --output ../blender/fluid_data.npz
    python blender_export.py --input ./sph_output --output ../blender/fluid_data.npz --subsample 4
"""

import argparse, re, sys
import numpy as np
from pathlib import Path


# ──────────────────────────────────────────
#  XYZ parser（raw / 標準 両フォーマット）
# ──────────────────────────────────────────

def parse_xyz(path: Path) -> np.ndarray:
    lines = path.read_text(encoding="utf-8", errors="replace").strip().splitlines()
    if not lines:
        return np.zeros((0, 3), dtype=np.float32)

    data_lines = lines
    try:
        n = int(lines[0].strip())
        data_lines = lines[2: 2 + n]   # 標準 XYZ
    except ValueError:
        pass                            # raw XYZ（fluid-engine-dev 形式）

    pts = []
    for line in data_lines:
        p = line.split()
        if len(p) >= 3:
            try:
                pts.append([float(p[0]), float(p[1]), float(p[2])])
            except ValueError:
                pass
    return np.array(pts, dtype=np.float32) if pts else np.zeros((0, 3), dtype=np.float32)


def _sort_key(p: Path):
    nums = re.findall(r"\d+", p.stem)
    return int(nums[-1]) if nums else 0


def collect_files(input_path: Path):
    if input_path.is_file():
        return [input_path]
    exts = {".xyz", ".pos"}
    files = [f for f in input_path.iterdir() if f.suffix.lower() in exts]
    return sorted(files, key=_sort_key)


# ──────────────────────────────────────────
#  Main
# ──────────────────────────────────────────

def export(input_path: Path, output: Path, subsample: int, fps: float):
    files = collect_files(input_path)
    if not files:
        print(f"[ERROR] ファイルが見つかりません: {input_path}")
        sys.exit(1)

    print(f"[blender_export] {len(files)} フレームを読み込み中... (間引き: 1/{subsample})")

    all_frames = []
    n_particles_max = 0

    for i, f in enumerate(files):
        if i % 20 == 0:
            print(f"  {i+1}/{len(files)}  {f.name}", end="\r")
        pts = parse_xyz(f)
        if subsample > 1:
            pts = pts[::subsample]
        all_frames.append(pts)
        n_particles_max = max(n_particles_max, len(pts))

    print(f"\n  完了: {len(all_frames)} フレーム, 最大 {n_particles_max} 粒子")

    # 全フレームを統一サイズの配列にパディング（Blender 側で扱いやすくするため）
    n_frames = len(all_frames)
    data = np.zeros((n_frames, n_particles_max, 3), dtype=np.float32)
    counts = np.zeros(n_frames, dtype=np.int32)
    for i, pts in enumerate(all_frames):
        n = len(pts)
        data[i, :n] = pts
        counts[i] = n

    # バウンディングボックス（全フレーム合算）
    valid = data[data != 0].reshape(-1, 3) if data.any() else np.zeros((1, 3))
    bmin = data[:, :, :].reshape(-1, 3).min(axis=0)
    bmax = data[:, :, :].reshape(-1, 3).max(axis=0)

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        positions=data,          # (F, N, 3)  float32
        counts=counts,            # (F,)       int32  — 有効粒子数
        bmin=bmin,                # (3,)       float32
        bmax=bmax,                # (3,)       float32
        fps=np.float32(fps),
        subsample=np.int32(subsample),
    )

    size_mb = output.stat().st_size / 1024 / 1024
    print(f"[blender_export] 保存完了 → {output}  ({size_mb:.1f} MB)")
    print(f"  shape   : positions {data.shape}")
    print(f"  bounds  : min={bmin.round(3)}  max={bmax.round(3)}")
    print(f"  fps     : {fps}")


def main():
    ap = argparse.ArgumentParser(description="XYZ → Blender NPZ エクスポーター")
    ap.add_argument("--input",     required=True)
    ap.add_argument("--output",    default=str(Path(__file__).parent.parent / "blender" / "fluid_data.npz"))
    ap.add_argument("--subsample", type=int,   default=1,    help="粒子を 1/N に間引く（高速化用）")
    ap.add_argument("--fps",       type=float, default=30.0)
    args = ap.parse_args()
    export(Path(args.input), Path(args.output), args.subsample, args.fps)


if __name__ == "__main__":
    main()
