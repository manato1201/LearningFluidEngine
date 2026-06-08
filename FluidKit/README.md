# FluidKit

fluid-engine-dev の知見をもとに作った、流体エフェクト作成 & 多ツール連携パイプライン。

## 構成

```
FluidKit/
├── viewer/
│   ├── index.html          ← Three.js パーティクル/メッシュ ビューワー
│   └── data/               ← シミュレーション出力 JSON
├── tools/
│   ├── gen_sample.py       ← サンプルデータ生成（pyjet不要）
│   └── xyz_to_json.py      ← sph_sim等のXYZ出力をJSONに変換
└── neural/
    ├── collect_data.py     ← 学習データ収集（多数シム実行）
    ├── model.py            ← Neural Fluid モデル（GNN）
    ├── train.py            ← 訓練スクリプト
    └── infer_to_json.py    ← 推論結果をViewerで表示
```

## クイックスタート

### 1. Three.js Viewer を開く（インストール不要）

```
viewer/index.html をブラウザで開くだけ。
デモデータが自動再生されます。
```

### 2. サンプルデータ生成

```bash
cd tools
python gen_sample.py --preset water_drop --frames 120
python gen_sample.py --preset smoke       --frames 120
python gen_sample.py --preset splash      --frames 100
```

生成された `viewer/data/*.json` を Viewer にドラッグ&ドロップ。

### 3. fluid-engine-dev の出力を変換

```bash
# sph_sim の出力フォルダを変換
python tools/xyz_to_json.py \
  --input  ../fluid-engine-dev-1.3.3/build/bin/sph_sim_output \
  --output viewer/data/sph_real.json \
  --fps 60
```

### 4. Neural Fluid（PyTorch が必要）

```bash
pip install torch numpy

# データ収集（20シミュレーション）
cd neural
python collect_data.py --simulations 20 --frames 80

# 訓練（CPU でも動く、GPU 推奨）
python train.py --epochs 80

# 推論 → Viewer で表示
python infer_to_json.py --checkpoint ./checkpoints/best.pt
```

## Viewer 操作

| 操作 | 内容 |
|---|---|
| マウスドラッグ | カメラ回転 |
| スクロール | ズーム |
| 📂 JSON を開く | カスタムデータ読み込み |
| ドラッグ&ドロップ | JSONファイルをドロップ |
| ● / ▲ ボタン | パーティクル / メッシュ切替 |
| カラーマップ | 深度 / 速度 / Heat / Ocean |

## プリセット一覧

| 名前 | 内容 |
|---|---|
| `water_drop` | 球形水滴が落下・床に衝突 |
| `smoke` | 上昇する煙（浮力付き） |
| `splash` | 左右の流れが中央で衝突 |
