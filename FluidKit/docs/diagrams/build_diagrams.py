"""
build_diagrams.py — docs/diagrams/*.svg を生成する (図は手で描かず、このスクリプトを直して再生成する)。

    python FluidKit/docs/diagrams/build_diagrams.py

図:
  architecture.svg     システム構成図 (グループ化した構成図)
  dataflow.svg         データの流れ (番号つきフロー: gen_sample → JSON → ビューア)
  viewer-internals.svg ビューア内部 (読み込み → データセット → View → 描画 / 時間同期)
  method-selection.svg 手法の選び方 (フローチャート)
  solver-step.svg      SPH 1 ステップの処理 (空間ハッシュ → 密度 → 圧力 → 力 → 積分 → 境界)
"""
from pathlib import Path
from svgkit import Svg, C

OUT = Path(__file__).parent


# ──────────────────────────────────────────────
def architecture():
    s = Svg(1400, 790, "FluidKit システム構成図",
            "生成スクリプト、ソルバー、データ、ブラウザ表示、テストとCIの関係")
    s.text(30, 38, "FluidKit システム構成図", 22, C["text"], 700)
    s.text(30, 60, "番号は処理の順序。実線 = データ/呼び出し、破線 = 検証", 12.5, C["muted"])

    y0 = 80
    s.group(30, y0, 640, 270, "生成・実行 (Python)", "blue", "FluidKit/tools・neural")
    s.group(730, y0, 640, 270, "ソルバー (solver/ ・ neural/model_v2.py)", "orange", "物理の計算本体。UI や I/O は持たない")
    L = [(50, "tools/gen_sample.py", "SPH プリセット → JSON"), (360, "tools/gen_curl.py", "カールノイズ煙 → JSON"),
         (50, "tools/compare_solvers.py", "同条件で 3 手法 → 指標表"), (360, "neural/collect_data.py", "学習データ + cKDTree 近傍"),
         (50, "neural/train_v2.py", "AMP 訓練 → チェックポイント"), (360, "neural/infer_v2.py", "ロールアウト → JSON")]
    for i, (x, t, sub) in enumerate(L):
        s.node(x, y0 + 60 + (i // 2) * 70, 290, 56, t, sub, "blue", mono=True)
    R = [(750, "SPHSolver", "空間ハッシュ + 陽解法"), (1060, "FLIPSolver", "粒子 + MAC 格子"),
         (750, "StableFluids2D", "Stam 格子法 (密度場)"), (1060, "CurlNoise2D / 3D", "発散ゼロの擬似流体"),
         (750, "SimulationRecord", "SHA-256 で再現性検証"), (1060, "NeuralFluidV2 (GNN)", "KNN + メッセージパッシング")]
    for i, (x, t, sub) in enumerate(R):
        s.node(x, y0 + 60 + (i // 2) * 70, 290, 56, t, sub, "orange", mono=True)
    s.arrow([(670, 215), (730, 215)], "blue", label="solver を呼ぶ", lpos=(700, 200))
    s.badge(700, 232, 1)

    y1 = 410
    s.group(30, y1, 640, 160, "データ", "green", "再生成できるものは gitignore")
    s.node(50, y1 + 60, 290, 56, "viewer/data/*.json(.gz)", "metadata / frames / speeds", "green", mono=True)
    s.node(360, y1 + 60, 290, 56, "neural/dataset_v2 ・ checkpoints", ".npy・NBR / fp32・fp16 .pt", "green", mono=True)
    s.group(730, y1, 640, 160, "ブラウザ (ビルド不要・file:// で開ける)", "purple", "FluidKit/viewer")
    s.node(750, y1 + 60, 290, 56, "viewer/index.html", "Three.js・比較・凡例・タイムライン", "purple", mono=True)
    s.node(1060, y1 + 60, 290, 56, "viewer/ripple.html", "js/heightfield.js: 波紋 + 光の集まり", "purple", mono=True)
    s.arrow([(350, 350), (350, 410)], "green", label="出力", lpos=(374, 384))
    s.badge(326, 384, 2, "green")
    s.arrow([(670, 490), (730, 490)], "purple", label="ドロップ", lpos=(700, 475))
    s.badge(700, 507, 3, "purple")
    s.arrow([(540, 410), (540, 350)], "green", dash="6 4", label="学習ループ", lpos=(590, 384))
    s.badge(516, 384, 4, "green")

    y2 = 620
    s.group(30, y2, 1340, 130, "品質・文書", "teal", "変更の安全網")
    Q = [(50, "tests/ (pytest 35 件)", "物性・再現性・I/O・比較指標"), (370, "viewer/js/heightfield.test.js", "Node: 発散・エネルギー・光の集まり"),
         (690, ".github/workflows/test.yml", "push / PR で pytest + node"), (1010, "docs/ (図・講義資料)", "LEARNING_GUIDE.md ほか")]
    for x, t, sub in Q:
        s.node(x, y2 + 54, 295, 56, t, sub, "teal", mono=True)
    s.arrow([(350, 620), (350, 570)], "teal", dash="6 4")
    s.arrow([(1050, 620), (1050, 570)], "teal", dash="6 4", label="検証", lpos=(1076, 600))
    s.badge(326, 598, 5, "teal")
    s.save(OUT / "architecture.svg")


# ──────────────────────────────────────────────
def dataflow():
    s = Svg(1340, 700, "データの流れ", "gen_sample.py が JSON を作り、ビューアが読み込んで描画するまで")
    s.text(30, 38, "データの流れ — gen_sample.py からビューアの描画まで", 22, C["text"], 700)
    s.text(30, 60, "青 = Python、緑 = ファイル、紫 = ブラウザ。番号の順に処理される", 12.5, C["muted"])
    W, H = 380, 100
    xs = [40, 480, 920]
    rows = [120, 310, 500]
    nodes = {
        1: (xs[0], rows[0], "python gen_sample.py", "--preset water_drop --frames 120 [--gzip]", "blue"),
        2: (xs[1], rows[0], "SimpleSPH.step() → SPHSolver.step()", "密度・圧力・粘性・積分 (solver/sph_solver.py)", "blue"),
        3: (xs[2], rows[0], "壁の反射 (restitution 0.3)", "SPHSolver は境界を持たないので gen_sample 側", "blue"),
        4: (xs[2], rows[1], "viewer/data/water_drop.json(.gz)", "metadata(method…) / frames / speeds", "green"),
        5: (xs[1], rows[1], "loadFiles() — ドロップ / 開く", ".json / .json.gz (DecompressionStream)", "purple"),
        6: (xs[0], rows[1], "prepareDataset()", "速度の算出・y と速さの最小最大・粒子間隔", "purple"),
        7: (xs[0], rows[2], "View (Scene + Points)", "色は LUT テクスチャ、サイズは粒子間隔基準", "purple"),
        8: (xs[1], rows[2], "時間同期", "frame = floor(t × fps)。最長データが基準", "purple"),
        9: (xs[2], rows[2], "描画 + 凡例", "色の範囲=実データの最小最大。同じ LUT で凡例", "purple"),
    }
    for n, (x, y, t, sub, col) in nodes.items():
        s.node(x, y, W, H, t, sub, col, mono=(n in (1, 4)))
        s.badge(x + W - 22, y + 22, n, col)
    gy = lambda r: r + H / 2
    s.arrow([(xs[0] + W, gy(rows[0])), (xs[1], gy(rows[0]))], "blue")
    s.arrow([(xs[1] + W, gy(rows[0])), (xs[2], gy(rows[0]))], "blue")
    s.arrow([(xs[2] + W / 2, rows[0] + H), (xs[2] + W / 2, rows[1])], "green", label="書き出し", lpos=(xs[2] + W / 2 + 40, 276))
    s.arrow([(xs[2], gy(rows[1])), (xs[1] + W, gy(rows[1]))], "purple")
    s.arrow([(xs[1], gy(rows[1])), (xs[0] + W, gy(rows[1]))], "purple")
    s.arrow([(xs[0] + W / 2, rows[1] + H), (xs[0] + W / 2, rows[2])], "purple")
    s.arrow([(xs[0] + W, gy(rows[2])), (xs[1], gy(rows[2]))], "purple")
    s.arrow([(xs[1] + W, gy(rows[2])), (xs[2], gy(rows[2]))], "purple")
    s.group(40, 630, 1260, 52, "", "gray")
    s.text(60, 662, "検証の入口:  pytest tests/test_io.py (JSON ラウンドトリップ)  ・  ヘッドレス Chrome (CDP) でビューアの表示・同期を確認", 12.5, C["muted"])
    s.save(OUT / "dataflow.svg")


# ──────────────────────────────────────────────
def viewer_internals():
    s = Svg(1300, 760, "ビューア内部構造", "JSON 読み込みからデータセット、View、描画ループ、時間同期までの構造")
    s.text(30, 38, "ビューア内部 — FluidViewer / View / ui の役割", 22, C["text"], 700)
    s.text(30, 60, "viewer/index.html の単一ファイル構成。状態は FluidViewer、DOM は ui、描画は View が持つ", 12.5, C["muted"])

    s.group(30, 85, 300, 290, "入力", "blue", "どちらも prepareDataset() を通る")
    s.node(50, 135, 260, 56, "ファイル / ドロップ", "複数可 (.json / .json.gz)", "blue")
    s.node(50, 205, 260, 56, "デモ generateDemo()", "重力・反発・粒子数を即時反映", "blue")

    s.group(390, 85, 440, 290, "FluidViewer (状態)", "orange", "datasets[] / activeId / compare / time")
    s.node(410, 135, 400, 56, "prepareDataset() → addDataset()", "範囲・粒子間隔・速度算出 / 同じ method は置換", "orange")
    s.node(410, 205, 400, 56, "datasets[] ・ 手法ごとに 1 本", "DEMO は実データを読むと消える", "orange")
    s.node(410, 275, 400, 56, "visible ・ master ・ time", "単独=active / 比較=最大 4 本 / 最長が基準", "orange")

    s.group(890, 85, 380, 290, "View (データセット 1 本分)", "purple", "Scene・Points・InstancedMesh")
    s.node(910, 135, 340, 56, "Points + ShaderMaterial", "位置 / 速さ属性、LUT で色、粒子間隔でサイズ", "purple")
    s.node(910, 205, 340, 56, "InstancedMesh (メッシュ表示)", "最大 300 個を間引いて描画", "purple")
    s.node(910, 275, 340, 56, "setFrame(idx)", "Float32Array.set でバッファを更新", "purple")

    s.arrow([(310, 163), (410, 163)], "blue")
    s.arrow([(310, 233), (355, 233), (355, 175), (410, 175)], "blue")
    s.arrow([(610, 191), (610, 205)], "orange")
    s.arrow([(610, 261), (610, 275)], "orange")
    s.arrow([(810, 303), (860, 303), (860, 163), (910, 163)], "orange", label="frameOf(ds)", lpos=(860, 225))

    s.group(30, 420, 1240, 310, "毎フレームの描画 (_animate)", "green", "requestAnimationFrame")
    steps = [("時間を進める", "time += Δt × 再生速度\nループ / 停止"), ("各データのフレーム", "frame = floor(time × fps)\n短いデータは最後で停止"),
             ("レイアウト", "コンテンツ領域を 1〜4 分割\nviewport + scissor"), ("uniform 更新", "uPxPerWorld・uSpacing\n色の範囲は全 View 共通"),
             ("render", "同じカメラで各 View を描画"), ("ui", "凡例・ラベル・統計を更新")]
    bw = 190
    for i, (t, sub) in enumerate(steps):
        x = 50 + i * 200
        s.add(f'<rect x="{x}" y="490" width="{bw}" height="150" rx="10" fill="#fff" stroke="#d3d9e6" stroke-width="1.2"/>')
        s.add(f'<rect x="{x}" y="490" width="{bw}" height="6" rx="3" fill="{C["green"]}"/>')
        s.badge(x + 24, 522, i + 1, "green")
        s.text(x + 44, 527, t, 13.5, C["text"], 600)
        for k, ln in enumerate(sub.split("\n")):
            s.text(x + 16, 560 + k * 20, ln, 11.5, C["muted"])
        if i < len(steps) - 1:
            s.arrow([(x + bw, 565), (x + 200, 565)], "green")
    s.text(50, 690, "比較モードの同期: 全 View が同じ time から各自の fps でフレームを選ぶ (fps や長さが違うデータでもずれない)", 12.5, C["text"], 600)
    s.text(50, 712, "10 万粒子でも JS 側は 1 ms 前後 (RTX 3070 で 59 fps)。巨大な JSON は読み込みに時間がかかる点に注意", 12, C["muted"])
    s.save(OUT / "viewer-internals.svg")


# ──────────────────────────────────────────────
def method_selection():
    s = Svg(1240, 700, "手法の選び方", "やりたいことから、SPH・FLIP・格子法・GNN・カールノイズ・高さ場のどれを使うかを選ぶフローチャート")
    s.text(30, 38, "手法の選び方 — やりたいことから選ぶ", 22, C["text"], 700)
    s.text(30, 60, "橙 = 質問、緑 = 答え(手法)。どれも近似で、物理的に厳密ではない", 12.5, C["muted"])

    def q(x, y, w, t):
        s.add(f'<rect x="{x}" y="{y}" width="{w}" height="58" rx="29" fill="{C["orange"]}"/>')
        s.text(x + w / 2, y + 35, t, 13.5, "#fff", 700, "middle")

    def a(x, y, w, t, sub):
        s.add(f'<rect x="{x}" y="{y}" width="{w}" height="76" rx="12" fill="#fff" stroke="{C["green"]}" stroke-width="2"/>')
        s.text(x + w / 2, y + 32, t, 15, C["green"], 700, "middle")
        s.text(x + w / 2, y + 54, sub, 11.5, C["muted"], 400, "middle")

    q(430, 90, 380, "水面の波紋・光の集まりだけ見たい?")
    a(60, 90, 300, "高さ場 + コースティクス", "ripple.html / heightfield.js")
    s.arrow([(430, 119), (360, 128)], "orange", label="はい", lpos=(395, 108))
    q(430, 200, 380, "外力・壁・衝突に反応させたい?")
    s.arrow([(620, 148), (620, 200)], "orange", label="いいえ", lpos=(648, 178))
    # はい → 自由表面?
    q(120, 320, 380, "水のような自由表面がある?")
    s.arrow([(520, 258), (310, 320)], "orange", label="はい", lpos=(400, 280))
    a(40, 450, 300, "格子法 (Stam)", "気体・煙。密度場を流す")
    s.arrow([(250, 378), (190, 450)], "orange", label="いいえ(気体)", lpos=(180, 420))
    q(360, 450, 300, "粒子の動きを直接見たい?")
    s.arrow([(380, 378), (480, 450)], "orange", label="はい(液体)", lpos=(470, 420))
    a(300, 570, 250, "SPH", "簡単・小さい刻みが要る")
    a(590, 570, 250, "FLIP", "粒子 + 格子で速い")
    s.arrow([(460, 508), (425, 570)], "orange", label="はい", lpos=(420, 545))
    s.arrow([(580, 508), (715, 570)], "orange", label="いいえ(圧力重視)", lpos=(690, 540))
    # いいえ → 学習?
    q(800, 320, 380, "学習で速く近似したい?")
    s.arrow([(720, 258), (950, 320)], "orange", label="いいえ", lpos=(860, 280))
    a(780, 450, 200, "GNN (Neural Fluid)", "要データ・訓練")
    a(1010, 450, 190, "カールノイズ", "発散ゼロの飾り")
    s.arrow([(880, 378), (880, 450)], "orange", label="はい", lpos=(906, 420))
    s.arrow([(1100, 378), (1105, 450)], "orange", label="いいえ(見た目だけ)", lpos=(1130, 420))
    s.text(30, 680, "迷ったら: 比較モードで同じ条件を複数手法に与えて見比べる (tools/compare_solvers.py → ビューアに複数 JSON をドロップ)", 12.5, C["muted"])
    s.save(OUT / "method-selection.svg")


# ──────────────────────────────────────────────
def solver_step():
    s = Svg(1300, 520, "SPH 1 ステップ", "SPHSolver.step() の処理順")
    s.text(30, 38, "SPH 1 ステップ — SPHSolver.step() の処理順", 22, C["text"], 700)
    s.text(30, 60, "solver/sph_solver.py。計算は 3D でもそのまま使える (カーネル定数は 2D 用の近似)", 12.5, C["muted"])
    items = [("空間ハッシュ再構築", "SpatialHash.build()\nセル = 平滑化長 h\nO(N)", "blue"),
             ("密度", "_compute_density()\nρ_i = Σ m_j W_poly6\n近傍 = ハッシュで絞る", "orange"),
             ("圧力", "p = max(k(ρ−ρ0), 0)\nTait 風の状態方程式\n負圧は 0 に切る", "orange"),
             ("力", "_compute_forces()\n圧力 −∇p (Spiky)\n粘性 μ∇²u + 重力", "orange"),
             ("時間積分", "Symplectic Euler\nv += dt·a\nx += dt·v", "green"),
             ("境界 (呼び出し側)", "gen_sample / compare で\n壁に反射\n(Solver は持たない)", "purple")]
    w = 190
    for i, (t, sub, col) in enumerate(items):
        x = 30 + i * 205
        s.add(f'<rect x="{x}" y="100" width="{w}" height="200" rx="12" fill="#fff" stroke="#d3d9e6" stroke-width="1.2"/>')
        s.add(f'<rect x="{x}" y="100" width="{w}" height="8" rx="4" fill="{C[col]}"/>')
        s.badge(x + 26, 140, i + 1, col)
        s.text(x + 48, 145, t, 14, C["text"], 700)
        for k, ln in enumerate(sub.split("\n")):
            s.text(x + 18, 190 + k * 24, ln, 12, C["muted"], 400, mono=(k == 0))
        if i < len(items) - 1:
            s.arrow([(x + w, 200), (x + 205, 200)], col)
    s.group(30, 340, 1240, 150, "落とし穴 (実際に踏んだもの)", "red")
    s.text(50, 395, "• _compute_forces の rho[nbrs, None] で (k,) ÷ (k,1) が (k,k) になり、近傍 2 個以上で必ずクラッシュしていた (呼び出し元がなく未検証だった)", 12.5)
    s.text(50, 420, "• 修正: 係数を先に (k,) で作り、prs_coeff[:, None] * grad で (k,dim) に掛ける", 12.5)
    s.text(50, 445, "• 教訓: 「最適化済み」と書かれていても呼び出し元がなければ検証されていない。まず end-to-end で 1 回動かす", 12.5)
    s.text(50, 470, "• 実測は 5.6 倍 (計画は 100 倍)。粒子ごとの Python ループが残るため (docs/PERF_RESULTS.md)", 12.5, C["muted"])
    s.save(OUT / "solver-step.svg")


if __name__ == "__main__":
    for f in (architecture, dataflow, viewer_internals, method_selection, solver_step):
        f()
        print("wrote", f.__name__)
