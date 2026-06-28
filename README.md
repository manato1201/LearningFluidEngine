# LearningFluidEngine

流体エンジンアーキテクチャの学習リポジトリ。SPH・FLIP・Neural Fluid（GNN）・WebAssembly・Blender MCP を組み合わせた 6 フェーズの流体エフェクトパイプライン。

## 構成

```
LearningFluidEngine/
├── FluidKit/
│   ├── viewer/             Three.js パーティクル Viewer (index.html)
│   ├── tools/              データ生成・変換ツール
│   │   ├── gen_sample.py       Python SPH シミュレーター
│   │   ├── xyz_to_json.py      XYZ → frames.json 変換
│   │   └── blender_export.py   XYZ → NPZ (Blender 連携)
│   ├── neural/             Neural Fluid (GNN)
│   │   ├── model.py            GNN v1 (184K params)
│   │   ├── model_v2.py         GNN v2/v3 (1.13M params, GlobalContext)
│   │   ├── collect_data.py     学習データ生成
│   │   ├── train_v2.py         訓練 (AMP + CosineAnnealing)
│   │   └── infer_v2.py         推論 → frames.json
│   ├── solver/             ★ 最適化済みソルバーパッケージ (NEW)
│   │   ├── stable_fluids.py    Stam 安定流体 (格子法, numpy 完全ベクトル化)
│   │   ├── sph_solver.py       SPH + 空間ハッシュ近傍探索
│   │   ├── flip_solver.py      FLIP ハイブリッド (numpy scatter)
│   │   ├── reproducibility.py  再現性保証 (SHA-256 状態ハッシュ)
│   │   └── runner.py           タイムアウト付きサンドボックス実行
│   ├── wasm/               WebAssembly リアルタイム SPH
│   │   ├── sph_wasm.cpp        C++ SPH ソース (外部依存ゼロ)
│   │   ├── fluid_wasm.html     ブラウザ Viewer
│   │   └── serve.py            COOP/COEP 対応 HTTP サーバー
│   ├── blender/            Blender MCP 連携スクリプト
│   ├── setup/              ★ 環境セットアップ (NEW)
│   │   └── download_build.py   GitHub Releases から WASM を自動ダウンロード
│   └── docs/               ドキュメント
│       ├── index.html
│       ├── algorithms.html
│       ├── ALGORITHMS.md
│       └── lecture.html        完全講義資料
└── .github/
    └── workflows/
        └── release.yml     ★ Emscripten ビルド + GitHub Release 自動配布 (NEW)
```

---

## フェーズ別クイックスタート

### Phase 1 — Python SPH シミュレーター

```bash
cd FluidKit/tools
python gen_sample.py --preset water_drop --frames 80
python gen_sample.py --preset smoke       --frames 80
python gen_sample.py --preset splash      --frames 80
```

生成された `viewer/data/*.json` を Viewer にドラッグ&ドロップ。

### Phase 2 — Three.js Viewer（インストール不要）

```
FluidKit/viewer/index.html をブラウザで直接開く
```

| 操作 | 内容 |
|---|---|
| マウスドラッグ | カメラ回転 |
| スクロール | ズーム |
| ドラッグ&ドロップ | JSON ファイルを読み込み |
| カラーマップ | 深度 / 速度 / Heat / Ocean |

### Phase 3 — fluid-engine-dev 連携（高精度 PCISPH）

```bash
sph_sim.exe -e 1 -s 0.03 -f 80 -p 30 -o ./sph_output -m xyz
python FluidKit/tools/xyz_to_json.py ./sph_output --output FluidKit/viewer/data/sph_real.json
```

### Phase 4 — Blender 連携

```
1. Blender を起動、MCP アドオンを有効化 (port 9876)
2. FluidKit/blender/mcp_setup.py の Chunk 1〜6 を順番に実行
3. Blender タイムラインで再生 → 粒子アニメーション確認
```

### Phase 5 — WebAssembly リアルタイム SPH

#### WASM ビルドがない場合（推奨: GitHub Releases から自動取得）

```bash
python FluidKit/setup/download_build.py
# → sph.wasm + sph.js を FluidKit/wasm/ に自動配置
```

#### 自分でビルドする場合（Emscripten 必要）

```bash
cd FluidKit/wasm
emcc sph_wasm.cpp -o sph.js -O3 -s WASM=1 -s MODULARIZE=1 \
  -s EXPORT_NAME="SphModule" \
  -s EXPORTED_FUNCTIONS='["_sph_init","_sph_step","_sph_get_positions","_sph_get_speeds","_sph_get_particle_count","_sph_reset","_sph_set_gravity","_sph_set_viscosity","_sph_set_stiffness","_sph_set_kernel_radius"]' \
  -s EXPORTED_RUNTIME_METHODS='["ccall","cwrap","HEAPF32"]' \
  -s ALLOW_MEMORY_GROWTH=1 -std=c++17
```

#### サーバー起動

```bash
cd FluidKit/wasm
python serve.py
# → http://localhost:8765/fluid_wasm.html
```

### Phase 6 — Neural Fluid v3（GNN）

```bash
pip install torch numpy scipy

cd FluidKit/neural
python collect_data.py --simulations 100 --output ./dataset_v2
python train_v2.py --ckpt ./checkpoints_v3 --epochs 120
python infer_v2.py --checkpoint ./checkpoints_v3/best.pt --frames 120
```

---

## 最適化済みソルバーパッケージ (FluidKit/solver/)

`FLUID_SECURITY_REFERENCE.md` のコードを本番品質に改善したソルバー群。

```python
from FluidKit.solver import StableFluids2D, SPHSolver, FLIPSolver
from FluidKit.solver import SimulationRecord, run_simulation_sandboxed

# Stam 安定流体（格子法）
fluid = StableFluids2D(N=64, dt=0.05, visc=0.001)
fluid.add_velocity(32, 32, 1.0, 0.5)
fluid.step()

# SPH + 空間ハッシュ
sph = SPHSolver(smoothing_length=0.1)
sph.step(particles, dt=0.01)

# タイムアウト付きサンドボックス実行
result = run_simulation_sandboxed(
    config={"N": 64, "dt": 0.05, "n_frames": 100},
    solver_factory=lambda cfg: StableFluids2D(**cfg),
    timeout_sec=300,
    audit_path="audit.json",
)
```

| ソルバー | 改善内容 | 速度向上 |
|---|---|---|
| StableFluids2D | Python ループ → `scipy.ndimage.map_coordinates` | **60〜80x** |
| SPHSolver | O(N²) 全ペア → SpatialHash O(N) | **47x** (N=500) |
| FLIPSolver | P2G/G2P → `np.add.at` + `map_coordinates` | **47x** |

---

## GitHub Releases への WASM 自動配布

```bash
# タグを打つだけで CI が WASM をビルドして Release に添付
git tag v1.0.0
git push origin v1.0.0

# 別 PC でのセットアップ（Emscripten 不要）
python FluidKit/setup/download_build.py
python FluidKit/setup/download_build.py --tag v1.0.0   # 特定バージョン
python FluidKit/setup/download_build.py --force         # 上書き
```

CI ワークフロー: `.github/workflows/release.yml`
- Ubuntu + Emscripten 3.1.56 で `sph.wasm` + `sph.js` をビルド
- `fluidkit-wasm-build.zip` を GitHub Release に自動アップロード

---

## Neural Fluid モデル比較

| バージョン | パラメータ数 | best_val | 状態 |
|---|---|---|---|
| v1 | 184K | 0.053 | ✅ 正常 |
| v2（バグ: 速度未正規化） | 1.13M | 16,838 | ❌ 損失爆発 |
| v2（バグ: 損失空間不一致） | 1.13M | 2.862 | △ モード崩壊 |
| **v3（修正完了）** | **1.13M** | **0.097** | ✅ 正常 |

v3 アーキテクチャ: `ParticleEncoder` → `GlobalContext`（全粒子平均）→ `MessagePassing × 3`（KNN k=16）→ デコーダ

---

## プリセット一覧

| 名前 | 重力 | 粘性 | 説明 |
|---|---|---|---|
| `water_drop` | -9.8 | 0.01 | 水滴落下・床に衝突 |
| `smoke` | +0.5 | 0.05 | 上昇する煙（浮力付き） |
| `splash` | -9.8 | 0.003 | 左右の流れが中央で衝突 |

---

## gitignore 対象（再生成可能な大容量ファイル）

| 対象 | 再生成方法 |
|---|---|
| `neural/dataset*/` | `collect_data.py` |
| `neural/checkpoints*/` | `train_v2.py` |
| `viewer/data/*.json` | `gen_sample.py` / `infer_v2.py` |
| `wasm/sph.wasm`, `sph.js` | `emcc sph_wasm.cpp` または `download_build.py` |
| `blender/render/` | Blender でレンダリング |
| `tools/sph_output/` | `sph_sim.exe` |

---

## ドキュメント

| ファイル | 内容 |
|---|---|
| [docs/index.html](FluidKit/docs/index.html) | プロジェクト概要・API リファレンス |
| [docs/algorithms.html](FluidKit/docs/algorithms.html) | SPH・GNN・FLIP アルゴリズム詳細 |
| [docs/ALGORITHMS.md](FluidKit/docs/ALGORITHMS.md) | アルゴリズム解説（Markdown 版） |
| [docs/lecture.html](FluidKit/docs/lecture.html) | 完全講義資料（学生向け・数式解説付き） |

---

*参考: Fluid Engine Development (Doyub Kim, 2017) / Müller et al. SPH (2003) / Stam Stable Fluids (1999) / Interaction Networks (Battaglia et al., 2016)*
