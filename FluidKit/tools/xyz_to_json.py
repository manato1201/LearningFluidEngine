"""
xyz_to_json.py  —  FluidKit XYZ→JSON コンバーター
fluid-engine-dev の sph_sim / hybrid_liquid_sim 等が出力する
*.xyz ファイルをまとめて frames.json に変換します。

使い方:
    # フォルダ内の全 xyz を変換（数字順）
    python xyz_to_json.py --input ./sph_sim_output --output ../viewer/data/sph.json

    # 単一ファイル確認
    python xyz_to_json.py --input ./frame_0000.xyz --output ../viewer/data/single.json

    # FPS 指定
    python xyz_to_json.py --input ./output --fps 30 --output ../viewer/data/out.json
"""

import json, re, sys, argparse
from pathlib import Path


# ──────────────────────────────────────────
#  XYZ parser
# ──────────────────────────────────────────

def parse_xyz(path: Path):
    """
    XYZ フォーマットを自動検出してパース。
    2種類対応:
      [標準]  1行目=粒子数, 2行目=コメント, 残り=X Y Z
      [raw]   全行が "X Y Z" のみ（fluid-engine-dev の出力形式）
    Returns: flat list [x0,y0,z0, x1,y1,z1, ...]
    """
    lines = path.read_text().strip().splitlines()
    if not lines:
        return []

    flat = []
    # 1行目が整数なら標準フォーマット
    try:
        n = int(lines[0].strip())
        data_lines = lines[2: 2 + n]
    except ValueError:
        # raw フォーマット（fluid-engine-dev）: 全行がX Y Z
        data_lines = lines

    for line in data_lines:
        parts = line.split()
        if len(parts) >= 3:
            try:
                flat.extend([round(float(parts[0]), 5),
                              round(float(parts[1]), 5),
                              round(float(parts[2]), 5)])
            except ValueError:
                pass
    return flat


def parse_pos(path: Path):
    """
    POS フォーマット（バイナリ）:
        float32 x4 per particle (x, y, z, w)  little-endian
    """
    import struct
    data = path.read_bytes()
    n = len(data) // 16         # 4 floats × 4 bytes = 16
    flat = []
    for i in range(n):
        x, y, z, _ = struct.unpack_from("<4f", data, i * 16)
        flat.extend([round(x, 5), round(y, 5), round(z, 5)])
    return flat


# ──────────────────────────────────────────
#  Collect files
# ──────────────────────────────────────────

def _sort_key(p: Path):
    """ファイル名中の数字を抽出してソート."""
    nums = re.findall(r"\d+", p.stem)
    return int(nums[-1]) if nums else 0


def collect_files(input_path: Path):
    if input_path.is_file():
        return [input_path]
    exts = {".xyz", ".pos"}
    files = [f for f in input_path.iterdir() if f.suffix.lower() in exts]
    return sorted(files, key=_sort_key)


# ──────────────────────────────────────────
#  Bounding box
# ──────────────────────────────────────────

def compute_bounds(all_frames):
    inf = float("inf")
    mn = [inf, inf, inf]
    mx = [-inf, -inf, -inf]
    for flat in all_frames:
        for i in range(0, len(flat), 3):
            mn[0] = min(mn[0], flat[i])
            mn[1] = min(mn[1], flat[i+1])
            mn[2] = min(mn[2], flat[i+2])
            mx[0] = max(mx[0], flat[i])
            mx[1] = max(mx[1], flat[i+1])
            mx[2] = max(mx[2], flat[i+2])
    return [round(v, 4) for v in mn], [round(v, 4) for v in mx]


# ──────────────────────────────────────────
#  Main
# ──────────────────────────────────────────

def convert(input_path: Path, output: Path, fps: float, name: str):
    files = collect_files(input_path)
    if not files:
        print(f"[ERROR] XYZ/POS ファイルが見つかりません: {input_path}")
        sys.exit(1)

    print(f"[FluidKit] {len(files)} フレームを変換中: {input_path}")

    all_frames = []
    for i, f in enumerate(files):
        if i % 20 == 0:
            print(f"  reading {f.name} ({i+1}/{len(files)})...", end="\r")
        if f.suffix.lower() == ".pos":
            flat = parse_pos(f)
        else:
            flat = parse_xyz(f)
        all_frames.append(flat)

    n_particles = len(all_frames[0]) // 3 if all_frames else 0
    bmin, bmax = compute_bounds(all_frames)

    meta = {
        "name": name,
        "source": str(input_path),
        "frameCount": len(all_frames),
        "particleCount": n_particles,
        "fps": fps,
        "bounds": {"min": bmin, "max": bmax},
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as fout:
        json.dump({"metadata": meta, "frames": all_frames}, fout,
                  separators=(",", ":"))

    size_kb = output.stat().st_size / 1024
    print(f"\n[FluidKit] saved → {output}  ({size_kb:.1f} KB,  {len(all_frames)} frames)")


def main():
    ap = argparse.ArgumentParser(description="XYZ/POS → frames.json コンバーター")
    ap.add_argument("--input",  required=True, help="XYZフォルダ or 単一ファイル")
    ap.add_argument("--output", required=True, help="出力 JSON パス")
    ap.add_argument("--fps",    type=float, default=60.0)
    ap.add_argument("--name",   default="fluid")
    args = ap.parse_args()
    convert(Path(args.input), Path(args.output), args.fps, args.name)


if __name__ == "__main__":
    main()
