# LearningFluidEngine

流体エンジンアーキテクチャの学習リポジトリ。**SPH・FLIP・Stable Fluids・Neural Fluid(GNN)・WebAssembly・Blender MCP** を組み合わせた、6 フェーズの流体エフェクトのパイプラインです。

![構成図](docs/diagrams/architecture.svg)

## 状態

| 項目 | 内容 |
|---|---|
| ソルバー | SPH(空間ハッシュ)・Stable Fluids・FLIP。NumPy でベクトル化 |
| テスト | `cd FluidKit && python -m pytest -q`(運動量・質量の保存、境界、再現性ハッシュ、入出力 + 文書の検査) |
| 未確認 | WebAssembly のビルド・Blender 連携・GNN の学習は、この文書の検査では**実行していません** |
| 学習資料 | 講義 3 回あり。**保守・拡張の詳しい資料は未作成**(簡易版のみ) |
| 既知の問題 | [docs/learn/index.md](docs/learn/index.md)(`verify` がステップ数を見ない 等) |

## クイックスタート

```bash
cd FluidKit/tools
python gen_sample.py --preset water_drop --frames 80      # SPH のサンプル → frames.json
cd .. && python -m pytest -q                              # テスト
```

フェーズ別の使い方(Viewer・Neural Fluid・WebAssembly・Blender)は、[旧 README の詳細](docs/reference/setup-and-phases.md)。

## ドキュメント

| 読みたいもの | 場所 |
|---|---|
| 全体の地図 | [docs/index.md](docs/index.md) |
| **基礎から学ぶ**(講義 3 回・実行できる例つき) | [docs/lecture/](docs/lecture/index.md) |
| コードの地図・既知の問題 | [docs/learn/index.md](docs/learn/index.md) |
| 構成・フェーズ別の使い方の**詳細**(旧 README) | [docs/reference/setup-and-phases.md](docs/reference/setup-and-phases.md) |
| アルゴリズム・性能の記録 | [ALGORITHMS.md](FluidKit/docs/ALGORITHMS.md)・[PERF_RESULTS.md](FluidKit/docs/PERF_RESULTS.md) |

## 限界(正直に)

- 教育用の実装で、**厳密な物理の検証**(実験との比較)はしていません。正しさは、保存則・境界のテストで確かめています。
- 性能の数字は、書かれた時点の環境のものです。
