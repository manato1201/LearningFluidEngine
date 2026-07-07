# FluidKit

fluid-engine-dev の知見をもとに作った、流体エフェクト作成 & 多ツール連携パイプライン。

## 構成

```
FluidKit/
├── viewer/
│   ├── index.html          ← Three.js パーティクル/メッシュ ビューワー
│   └── data/               ← シミュレーション出力 JSON（gitignore対象）
├── tools/
│   ├── gen_sample.py       ← Python SPH シミュレーター（Poly6カーネル）
│   ├── xyz_to_json.py      ← sph_sim XYZ出力 → JSON 変換
│   └── blender_export.py   ← XYZ → NPZ（Blender連携用）
├── neural/
│   ├── collect_data.py     ← 学習データ収集（速度正規化・cKDTree近傍事前計算）
│   ├── model_v2.py         ← Neural Fluid v2/v3（GNN, 1.13M params）
│   ├── train_v2.py         ← v2/v3 訓練スクリプト（AMP・CosineAnnealing・mmapデータセット）
│   └── infer_v2.py         ← v2/v3 推論 → Viewer JSON出力（速度正規化対応）
├── solver/                 ← 最適化済みソルバーパッケージ
│   ├── stable_fluids.py    ← Stam 安定流体（格子法, numpy ベクトル化）
│   ├── sph_solver.py       ← SPH + 空間ハッシュ近傍探索
│   ├── flip_solver.py      ← FLIP ハイブリッド（numpy scatter）
│   └── reproducibility.py  ← 再現性保証（SHA-256 状態ハッシュ）
├── utils.py                ← 正規化/逆正規化の共通ユーティリティ
├── tests/                  ← pytest テストスイート（物性・再現性・I/O）
├── wasm/
│   ├── sph_wasm.cpp        ← リアルタイムSPH（外部依存ゼロ C++）
│   ├── fluid_wasm.html     ← ブラウザ上で動くリアルタイムビューワー
│   └── serve.py            ← COOP/COEPヘッダー付きHTTPサーバー（port 8765）
├── blender/
│   ├── mcp_setup.py        ← Blender MCP 経由セットアップ（6チャンク）
│   ├── setup_scene.py      ← Blender シーン自動構築スクリプト
│   └── render_to_gif.py    ← PNG → GIF/MP4 変換
└── docs/
    ├── index.html          ← プロジェクトドキュメント
    ├── algorithms.html     ← SPH・GNN アルゴリズム解説（MathJax）
    ├── ALGORITHMS.md       ← アルゴリズム解説（Markdown）
    └── PERF_RESULTS.md     ← 性能改善の実測記録
```

## フェーズ別クイックスタート

### Phase 1｜Python SPH シミュレーター

```bash
cd tools
python gen_sample.py --preset water_drop --frames 80
python gen_sample.py --preset smoke       --frames 80
python gen_sample.py --preset splash      --frames 80

# --gzip を付けると <preset>.json.gz として圧縮出力（サイズ約 60〜70% 減）
python gen_sample.py --preset water_drop --frames 80 --gzip
```

生成された `viewer/data/*.json`（`.json.gz` も対応）を Viewer にドラッグ&ドロップ。

### Phase 2｜Three.js Viewer（インストール不要）

```
viewer/index.html をブラウザで直接開くだけ。
```

| 操作 | 内容 |
|---|---|
| マウスドラッグ | カメラ回転 |
| スクロール | ズーム |
| 📂 JSON を開く | カスタムデータ読み込み |
| ドラッグ&ドロップ | JSONファイルをドロップ |
| ● / ▲ ボタン | パーティクル / メッシュ切替 |
| カラーマップ | 深度 / 速度 / Heat / Ocean |

### Phase 3｜fluid-engine-dev 連携（高精度PCISPH）

```bash
# sph_sim.exe で高精度シミュレーション実行
sph_sim.exe -e 1 -s 0.03 -f 80 -p 30 -o ./sph_output -m xyz

# XYZ → JSON 変換
python tools/xyz_to_json.py ./sph_output --output viewer/data/sph_real.json
```

### Phase 4｜Blender 連携

```
1. Blender を起動、MCPアドオンを有効化（port 9876）
2. blender/mcp_setup.py の Chunk 1〜6 を MCP 経由で順番に実行
3. Blender タイムラインで再生 → 粒子アニメーション確認
```

※ `fluid_data.npz` が必要な場合は先に生成：
```bash
python tools/blender_export.py
```

### Phase 5｜WebAssembly リアルタイム SPH

```bash
cd wasm
python serve.py
# → http://localhost:8765/fluid_wasm.html をブラウザで開く
```

gravity / viscosity / stiffness をスライダーでリアルタイム調整可能。

### Phase 6｜Neural Fluid v3（GNN）

```bash
pip install torch numpy scipy

# Step 1: 学習データ生成（100シミュレーション、cKDTreeで近傍を事前計算）
cd neural
python collect_data.py --simulations 100 --output ./dataset_v2

# Step 2: 訓練（GPU推奨、約70分）
python train_v2.py --ckpt ./checkpoints_v3 --epochs 120

# Step 3: 推論 → Viewer JSON出力
python infer_v2.py --checkpoint ./checkpoints_v3/best.pt --frames 120
```

## テスト

```bash
# FluidKit/ ディレクトリで実行
pip install numpy scipy pytest
pytest tests/ -v
```

運動量保存・境界貫通なし・再現性ハッシュ一致・JSON/NPZ ラウンドトリップを検証する
22 テストが含まれる（`.github/workflows/test.yml` で push / PR ごとに自動実行）。

## Neural Fluid モデル比較

| バージョン | パラメータ数 | best_val | 特徴 |
|---|---|---|---|
| v1 | 184K | 0.053 | シンプルなInteraction Network |
| v2（バグあり） | 1.13M | 16,838 | 速度未正規化 → 損失爆発 |
| v2（修正後） | 1.13M | 2.862 | 速度正規化済みだが損失空間不一致 |
| **v3** | **1.13M** | **0.097** | 位置損失のみ・GlobalContext・LayerNorm |

v3 アーキテクチャ:
- `ParticleEncoder` → `GlobalContext`（全粒子平均） → `MessagePassing × 3`（KNN k=16） → デコーダ
- AMP（混合精度）+ CosineAnnealingLR + AdamW

## プリセット一覧

| 名前 | 重力 | 粘性 | 見た目 |
|---|---|---|---|
| `water_drop` | -9.8 | 0.01 | 水滴落下・床に衝突 |
| `smoke` | -1.0 | 0.05 | 上昇する煙（浮力付き） |
| `splash` | -9.8 | 0.003 | 左右の流れが中央で衝突 |

## gitignore 対象（再生成可能な大容量ファイル）

| 対象 | 再生成方法 |
|---|---|
| `neural/dataset*/` | `collect_data.py` |
| `neural/checkpoints*/` | `train_v2.py` |
| `viewer/data/*.json` | `gen_sample.py` / `infer_v2.py` |
| `blender/render/` | Blender でレンダリング |
| `tools/sph_output/` | `sph_sim.exe` |
| `wasm/sph.wasm`, `sph.js` | `emcc sph_wasm.cpp` |
