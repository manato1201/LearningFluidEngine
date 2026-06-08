"""
render_to_gif.py  —  レンダリング済み PNG をアニメーション GIF / MP4 に変換

使い方:
    python render_to_gif.py              # ./render/ 内の PNG → fluid_anim.gif + .mp4
    python render_to_gif.py --fps 15
    python render_to_gif.py --format mp4
"""

import argparse, re, subprocess, sys
from pathlib import Path

try:
    from PIL import Image
    PIL_OK = True
except ImportError:
    PIL_OK = False


def _sort_key(p: Path):
    nums = re.findall(r"\d+", p.stem)
    return int(nums[-1]) if nums else 0


def to_gif(frames, output: Path, fps: int, size):
    if not PIL_OK:
        print("[WARN] Pillow が未インストール: pip install Pillow")
        return
    imgs = [Image.open(f).convert("RGBA") for f in frames]
    if size:
        imgs = [i.resize(size, Image.LANCZOS) for i in imgs]
    dur = int(1000 / fps)
    imgs[0].save(output, save_all=True, append_images=imgs[1:],
                 duration=dur, loop=0, optimize=True)
    print(f"[render_to_gif] GIF 保存 → {output}  ({output.stat().st_size//1024} KB)")


def to_mp4(frames_dir: Path, output: Path, fps: int):
    pattern = str(frames_dir / "frame_%*.png")
    # ffmpeg で PNG シーケンス → MP4
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-pattern_type", "glob",
        "-i", pattern,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "20",
        str(output),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"[render_to_gif] MP4 保存 → {output}")
    except FileNotFoundError:
        print("[WARN] ffmpeg が見つかりません。インストールしてください: https://ffmpeg.org/")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] ffmpeg 失敗: {e.stderr.decode()[:200]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  default=str(Path(__file__).parent / "render"))
    ap.add_argument("--output", default=None)
    ap.add_argument("--fps",    type=int, default=30)
    ap.add_argument("--format", choices=["gif","mp4","both"], default="both")
    ap.add_argument("--width",  type=int, default=None)
    args = ap.parse_args()

    render_dir = Path(args.input)
    frames = sorted(render_dir.glob("frame_*.png"), key=_sort_key)
    if not frames:
        print(f"[ERROR] PNG が見つかりません: {render_dir}")
        sys.exit(1)

    print(f"[render_to_gif] {len(frames)} フレーム読み込み")

    size = None
    if args.width:
        from PIL import Image
        w, h = Image.open(frames[0]).size
        size = (args.width, int(h * args.width / w))

    base = render_dir.parent
    if args.format in ("gif", "both"):
        out = Path(args.output) if args.output else base / "fluid_anim.gif"
        to_gif(frames, out, args.fps, size)
    if args.format in ("mp4", "both"):
        out = Path(args.output).with_suffix(".mp4") if args.output else base / "fluid_anim.mp4"
        to_mp4(render_dir, out, args.fps)


if __name__ == "__main__":
    main()
