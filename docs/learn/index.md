# LearningFluidEngine 学習資料(保守・拡張の実務編)

このフォルダは、**FluidKit のソルバー・ツール・検査を直す・足す人**のための資料です。基礎から順に学ぶ講義は [../lecture/index.md](../lecture/index.md)(第 0〜2 回)。既存の資料: [ALGORITHMS.md](../../FluidKit/docs/ALGORITHMS.md)・[PERF_RESULTS.md](../../FluidKit/docs/PERF_RESULTS.md)・`FluidKit/docs/lecture.html`。旧 README(構成・フェーズ別の使い方)は [setup-and-phases.md](../reference/setup-and-phases.md)。

| # | 資料 | いつ読むか |
|---|---|---|
| 1 | [全体像](01-overview.md) | 6 フェーズ・ソルバーの種類・データの形・設定値 |
| 2 | [コードの読み方](02-code-tour.md) | どのファイルから読むか・流れ |
| 3 | [内側のしくみ](03-internals.md) | SPH・Stable Fluids・FLIP・再現性・サンドボックス・カールノイズ |
| 4 | [拡張のレシピ](04-extend.md) | ソルバー・プリセット・設定・比較の指標を**足したい** |
| 5 | [テストの書き方と実行](05-testing.md) | テストを回す・足す・文書の検査・**確かめていないこと** |
| 6 | [トラブルと保守](06-troubleshooting-maintenance.md) | 動かない・**既知の問題**・定期の保守 |
| 7 | [基礎の補強](07-foundations.md) | NumPy・ベクトル化・乱数のシード・dataclass・設計思想・アルゴリズム・流体と VFX の用語を**補強したい** |
| – | [用語集](glossary.md) | 分からない言葉が出た |

## 最初の 15 分

1. `FluidKit/` で `python -m pytest -q tests`(物性テストと文書の検査が通ること)。
2. `python tools/gen_sample.py --preset water_drop --frames 20`(`FluidKit/tools/` で)。`viewer/data/` に JSON ができる。
3. [構成図](../diagrams/architecture.svg)を見ながら、[全体像](01-overview.md)を読む。

## コードの地図(`FluidKit/`)

| 場所 | 役割 |
|---|---|
| `solver/sph_solver.py` | SPH + 空間ハッシュ(`SpatialHash`・`SPHSolver`・`Particle`) |
| `solver/stable_fluids.py` | Stam の Stable Fluids(格子法・numpy) |
| `solver/flip_solver.py` | FLIP(粒子 + MAC 格子) |
| `solver/reproducibility.py` | `SimulationRecord`(状態ハッシュ・監査ログ) |
| `solver/runner.py` | タイムアウト付きのサンドボックス実行・設定の検証 |
| `solver/curl_noise.py` | カールノイズ(発散ゼロの流れ) |
| `utils.py` | 位置・速度の正規化と逆正規化(Neural 用) |
| `tools/` | データ生成(`gen_sample.py`)・変換(`xyz_to_json.py`)・比較(`compare_solvers.py`) |
| `neural/` | GNN(Neural Fluid): `collect_data` → `train_v2` → `infer_v2` |
| `wasm/`・`viewer/`・`blender/` | WebAssembly のリアルタイム SPH・Three.js の Viewer・Blender 連携 |
| `tests/` | 物性テスト・再現性・入出力・カールノイズ・ソルバー比較 |
