# LearningFluidEngine 講義(基礎から)

流体シミュレーション(SPH・格子法・FLIP)の**考え方**と、**速さ・再現性・正しさ**の守り方を、動かしながら学ぶ講義です(`FluidKit/tests/test_docs.py` が、ずれと出力を検査します)。

| 回 | 題 | 学ぶこと | 目安 |
|---|---|---|---|
| [0](00-introduction.md) | 全体像 | 粒子と格子の 2 つの見方・3 つのソルバー | 1 h |
| [1](01-sph.md) | SPH | **カーネル**・**空間ハッシュ**(O(N²) → O(N))・運動量の保存 | 3.5 h |
| [2](02-reproducibility-and-tests.md) | 再現性とテスト | **状態の SHA-256**・ビット単位の一致・物性テスト | 2.5 h |

例は、リポジトリのルートで動かします(NumPy だけ。`FluidKit/` を import パスに足してあります)。講義のあとは [学習資料](../learn/index.md)。
