"""
download_build.py — GitHub Releases から WASM ビルドを自動ダウンロード

使い方:
    python FluidKit/setup/download_build.py            # 最新リリースを取得
    python FluidKit/setup/download_build.py --tag v1.2.0  # 特定バージョン

ビルドフォルダが存在しない環境（Emscripten 未インストール）での代替手段。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

# ──────────────────────────────────────────────────────────
#  設定
# ──────────────────────────────────────────────────────────

REPO       = "manato1201/LearningFluidEngine"
API_BASE   = f"https://api.github.com/repos/{REPO}"
ASSET_NAME = "fluidkit-wasm-build.zip"

# このスクリプトから見た wasm フォルダの相対パス
WASM_DIR = Path(__file__).parent.parent / "wasm"


# ──────────────────────────────────────────────────────────
#  ヘルパー
# ──────────────────────────────────────────────────────────

def _fetch_json(url: str) -> dict | list:
    req = urllib.request.Request(
        url,
        headers={
            "Accept":     "application/vnd.github+json",
            "User-Agent": "FluidKit-setup/1.0",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def _download(url: str, dest: Path, label: str = "") -> None:
    """進捗表示付きダウンロード。"""
    req = urllib.request.Request(
        url,
        headers={
            "Accept":     "application/octet-stream",
            "User-Agent": "FluidKit-setup/1.0",
        },
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        chunk = 32 * 1024
        with dest.open("wb") as f:
            while True:
                data = resp.read(chunk)
                if not data:
                    break
                f.write(data)
                downloaded += len(data)
                if total:
                    pct = downloaded / total * 100
                    bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
                    print(f"\r  [{bar}] {pct:5.1f}%  {downloaded//1024}KB", end="", flush=True)
    print()


# ──────────────────────────────────────────────────────────
#  メイン処理
# ──────────────────────────────────────────────────────────

def check_already_built() -> bool:
    """ローカルに既に WASM ビルドが存在するか確認。"""
    return (WASM_DIR / "sph.wasm").exists() and (WASM_DIR / "sph.js").exists()


def get_release(tag: str | None) -> dict:
    """指定タグ（または latest）のリリース情報を取得。"""
    if tag:
        url = f"{API_BASE}/releases/tags/{tag}"
    else:
        url = f"{API_BASE}/releases/latest"

    try:
        return _fetch_json(url)  # type: ignore[return-value]
    except Exception as e:
        print(f"[ERROR] GitHub API 呼び出し失敗: {e}")
        print(f"        URL: {url}")
        print("        ネットワーク接続・リポジトリ名を確認してください。")
        sys.exit(1)


def find_asset(release: dict, name: str) -> dict:
    """リリースから指定名のアセットを探す。"""
    for asset in release.get("assets", []):
        if asset["name"] == name:
            return asset
    available = [a["name"] for a in release.get("assets", [])]
    print(f"[ERROR] アセット '{name}' が見つかりません。")
    print(f"        利用可能: {available}")
    sys.exit(1)


def install(zip_path: Path) -> None:
    """zip を解凍して wasm/ フォルダにコピー。"""
    WASM_DIR.mkdir(parents=True, exist_ok=True)
    tmp = zip_path.parent / "_extracted"
    tmp.mkdir(exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(tmp)

    # build/ フォルダの中身を wasm/ にコピー
    build_dir = tmp / "build"
    if not build_dir.exists():
        # フラットな zip の場合も対応
        build_dir = tmp

    copied = []
    for src in build_dir.iterdir():
        if src.suffix in (".wasm", ".js"):
            dst = WASM_DIR / src.name
            shutil.copy2(src, dst)
            copied.append(dst.name)

    shutil.rmtree(tmp, ignore_errors=True)

    if not copied:
        print("[ERROR] zip に .wasm / .js ファイルが見つかりませんでした。")
        sys.exit(1)

    print(f"  インストール完了: {', '.join(copied)} → {WASM_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description="FluidKit WASM ビルドを GitHub Releases から取得")
    parser.add_argument("--tag",   default=None, help="リリースタグ (例: v1.0.0)。省略時は最新")
    parser.add_argument("--force", action="store_true", help="既存のビルドを上書き")
    args = parser.parse_args()

    # ── 既存チェック ──────────────────────────────────────
    if check_already_built() and not args.force:
        print("[OK] sph.wasm と sph.js は既に存在します。")
        print("     上書きする場合は --force を指定してください。")
        print(f"     場所: {WASM_DIR}")
        return

    # ── リリース取得 ───────────────────────────────────────
    tag_str = args.tag or "latest"
    print(f"[1/3] GitHub Releases を確認 ({REPO}, tag={tag_str}) ...")
    release = get_release(args.tag)
    print(f"      リリース: {release['name']} ({release['tag_name']})")
    print(f"      公開日:   {release.get('published_at', '?')}")

    # ── アセット取得 ───────────────────────────────────────
    asset = find_asset(release, ASSET_NAME)
    download_url = asset["browser_download_url"]
    size_kb = asset["size"] // 1024
    print(f"[2/3] ダウンロード中: {ASSET_NAME} ({size_kb} KB) ...")

    tmp_zip = Path(__file__).parent / "_tmp_wasm.zip"
    try:
        _download(download_url, tmp_zip, ASSET_NAME)
    except Exception as e:
        print(f"\n[ERROR] ダウンロード失敗: {e}")
        tmp_zip.unlink(missing_ok=True)
        sys.exit(1)

    # ── インストール ───────────────────────────────────────
    print(f"[3/3] インストール中 → {WASM_DIR} ...")
    install(tmp_zip)
    tmp_zip.unlink(missing_ok=True)

    # ── 確認 ──────────────────────────────────────────────
    wasm_size = (WASM_DIR / "sph.wasm").stat().st_size if (WASM_DIR / "sph.wasm").exists() else 0
    print()
    print("═" * 50)
    print("  FluidKit WASM セットアップ完了!")
    print(f"  sph.wasm : {wasm_size:,} bytes")
    print()
    print("  次のステップ:")
    print("    cd FluidKit/wasm")
    print("    python serve.py")
    print("    → http://localhost:8765/fluid_wasm.html")
    print("═" * 50)


if __name__ == "__main__":
    main()
