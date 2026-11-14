"""docs/diagrams/ の構成図(SVG)を作る。`python tools/make_diagrams.py` で作り直す。部品は tools/svgkit.py。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from svgkit import Diagram  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "docs" / "diagrams"


def architecture() -> Diagram:
    d = Diagram(1500, 760, "FluidKit の構成",
                "Python のソルバー(SPH・Stable Fluids・FLIP)が粒子や格子を計算し、状態ハッシュで再現性を記録する。結果は frames.json や NPZ になり、"
                "Three.js の Viewer・WebAssembly のリアルタイム SPH・Blender で見る。GNN(Neural Fluid)はソルバーの結果を学習して置き換える。")
    d.text(40, 44, "LearningFluidEngine(FluidKit)— 構成", cls="hd")
    d.text(40, 66, "青=主な流れ / 灰の破線=補助。速さ(空間ハッシュ)・再現性(状態ハッシュ)・正しさ(物性テスト)を仕組みで守る", cls="note")

    d.group(40, 100, 500, 560, "SOLVERS — solver/", color="teal")
    sph = d.card(180, 220, "wave", "SPH", "空間ハッシュ\n運動量が保存される")
    sf = d.card(400, 220, "chart", "Stable Fluids", "格子法・numpy\n完全ベクトル化")
    flip = d.card(180, 440, "gear", "FLIP", "粒子 + 格子\nのハイブリッド")
    rec = d.card(400, 440, "check", "再現性", "設定・状態の SHA-256\nverify()")

    d.group(600, 100, 400, 560, "DATA — 結果", color="gray")
    data = d.card(800, 330, "db", "frames.json / NPZ", "gen_sample・xyz_to_json\nblender_export")

    d.group(1060, 100, 410, 560, "VIEW & LEARN", color="blue")
    view = d.card(1180, 220, "browser", "Viewer", "Three.js・WebAssembly SPH\n(リアルタイム)")
    bl = d.card(1360, 220, "film", "Blender", "MCP 連携・GIF")
    nn = d.card(1270, 480, "pulse", "Neural Fluid", "GNN v2/v3\n学習 → 推論")

    d.arrow([sf.right, (570, 220), (570, 330), data.left], "結果を書き出す", at=2, dy=-10)
    d.arrow([rec.right, (570, 440), (570, 330)], "", kind="secondary")
    d.arrow([data.right, (1040, 330), (1040, 220), view.left], "読む", at=2, dx=-20, dy=-10)
    d.arrow([(1040, 330), (1040, 480), nn.left], "学習データ", kind="secondary", at=1, dx=-40, dy=0)
    d.arrow([view.right, bl.left], "", kind="secondary", at=0)
    return d


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "architecture.svg").write_text(architecture().svg(), encoding="utf-8")
    print("wrote", OUT / "architecture.svg")


if __name__ == "__main__":
    main()
