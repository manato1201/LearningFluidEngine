# 7. 基礎の補強(言語・設計思想・アルゴリズム・用語)

講義と学習資料の**前提になっている知識**を、ここで補います。「Python・NumPy・ベクトルの基本」と、流体の用語を、**このリポジトリのコードで実際に使われている形**で、動かして確かめます。

## 1. 言語の基礎: FluidKit で使う Python と NumPy

### 1.1 NumPy の配列: 粒子を「表」で持つ

粒子の位置は、**(粒子の数, 2) の表**。NumPy は、**全粒子の計算を、ループなしで一度に**行う。`shape`・スライス・集計(`sum`・`mean`)を使う。

```python
# 実行できる
import numpy as np

pos = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])         # 3 粒子の位置
vel = np.array([[1.0, 0.0], [0.0, 2.0], [-1.0, 0.0]])
mass = np.array([1.0, 2.0, 1.0])

print("形:", pos.shape)
print("全粒子を 0.1 秒進める:", (pos + 0.1 * vel).round(2).tolist())
print("総運動量(質量 × 速度の和):", (mass[:, None] * vel).sum(axis=0).tolist())     # mass[:, None] は列にして、速度と掛ける
print("重心:", np.average(pos, axis=0, weights=mass).round(3).tolist())
```

```text
# 出力
形: (3, 2)
全粒子を 0.1 秒進める: [[0.1, 0.0], [1.0, 0.2], [0.9, 1.0]]
総運動量(質量 × 速度の和): [0.0, 4.0]
重心: [0.75, 0.25]
```

### 1.2 ブロードキャストと「ベクトル化」

形の違う配列でも、**規則に沿って自動で広げて**計算する(ブロードキャスト)。**Python の `for` で 1 粒子ずつ**書くより、**桁違いに速い**。ただし、**どこがベクトル化されていて、どこがまだループか**を意識する(`SpatialHash` はループ)。

```python
# 実行できる
import time

import numpy as np

n = 20000
x = np.random.RandomState(0).rand(n)

start = time.perf_counter()
slow = [v * v + 1.0 for v in x]                    # Python のループ
t_loop = time.perf_counter() - start

start = time.perf_counter()
fast = x * x + 1.0                                 # ベクトル化
t_vec = time.perf_counter() - start

print("結果は同じ:", bool(np.allclose(slow, fast)))
print("ベクトル化の方が速い:", t_vec < t_loop)
```

```text
# 出力
結果は同じ: True
ベクトル化の方が速い: True
```

### 1.3 乱数: シードで再現する

`np.random.RandomState(seed)` は、**同じ seed なら、同じ数列**。**再現できる実験**の基本(テストが、同じハッシュを期待できる)。**グローバルな `np.random.seed` を触ると、他の場所に影響する**ので注意(→ 既知の問題)。

```python
# 実行できる
import numpy as np

a = np.random.RandomState(42).rand(3)
b = np.random.RandomState(42).rand(3)
c = np.random.RandomState(43).rand(3)
print("同じ seed:", bool((a == b).all()), " 違う seed:", bool((a == c).all()))
```

```text
# 出力
同じ seed: True  違う seed: False
```

### 1.4 `@dataclass` と型ヒント

`Particle` のような**データを持つだけのクラス**は `@dataclass`。`field(default_factory=...)` で、**インスタンスごとに別の配列**を持たせる(共有してしまう事故を防ぐ)。

```python
# 実行できる
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Particle:
    pos: np.ndarray = field(default_factory=lambda: np.zeros(2))
    mass: float = 1.0


a, b = Particle(), Particle()
a.pos[0] = 5.0
print("別々の配列(共有しない):", a.pos.tolist(), b.pos.tolist())
```

```text
# 出力
別々の配列(共有しない): [5.0, 0.0] [0.0, 0.0]
```

## 2. 設計思想

| 原則 | 意味 | FluidKit のどこに出るか | 破るとどうなるか |
|---|---|---|---|
| **同じ問題を、別の方法で解いて比べる** | 粒子法・格子法・ハイブリッドで、得意・不得意を見る | `SPHSolver`・`StableFluids2D`・`FLIPSolver`・`compare_solvers.py` | 1 つの方法の癖を、物理だと思い込む |
| **成り立つべき性質をテストにする** | 正解の数値が無くても、保存則は確かめられる | 運動量・質量・境界のテスト | 壊れても通る |
| **再現性を保証する** | 同じ入力 → 同じ状態ハッシュ | `SimulationRecord`・シード | 差の原因が、実装か偶然か分からない |
| **測ってから最適化する** | 数字(実測)を残してから直す | `PERF_RESULTS.md`・コメントの実測 | 遅い所を、当てずっぽうで直す |
| **試して、だめなら記録して差し戻す** | 失敗も資産にする | `_project` の CG の試行の記録 | 同じ失敗を、繰り返す |
| **安全に実行する** | タイムアウト・検証・記録つきで動かす | `run_simulation_sandboxed` | 暴走して止まらない |
| **見た目用の近似と、物理を分ける** | カールノイズは物理を解かないと明記する | `curl_noise.py` のコメント | 見た目用を、物理の結果と誤解する |

## 3. 関数・アルゴリズム

| 名前 | 場所 | 何をする | 計算量・注意 |
|---|---|---|---|
| **空間ハッシュ** | `SpatialHash` | 空間を `h` 四方のセルに分け、**近い粒子だけ**を探す | 平均 O(N)。全ペアの O(N²) より桁違いに速い |
| **SPH のカーネル** | `_W_poly6` ほか | 距離に応じた**重み**で、周りの粒子の寄与を足す | 近傍の数に比例 |
| **Symplectic Euler** | `SPHSolver.step` | 先に速度、**新しい速度**で位置を更新 | O(N)。エネルギーが安定しやすい |
| **セミラグランジュ移流** | `StableFluids2D._advect` | 各格子点から**逆にたどり**、その位置の値を補間で取る | O(格子点数)。大きな `dt` でも発散しない(ぼやける) |
| **赤黒の反復(射影)** | `_project` | 格子を市松に分け、同じ色どうしを同時に更新して圧力を解く | O(格子 × 反復) |
| **SHA-256(状態ハッシュ)** | `SimulationRecord` | 状態配列のバイト列から、固定長の値 | 1 ビットでも違えば別 |

```python
# 実行できる
import numpy as np

from solver import SpatialHash

rng = np.random.RandomState(0)
pos = rng.rand(400, 2)
h = 0.08
table = SpatialHash(cell_size=h)
table.build(pos)

i = 0
brute = {j for j in range(len(pos)) if ((pos[j] - pos[i]) ** 2).sum() < h * h}                       # 全ペア: O(N)
hashed = {j for j in table.query(pos[i], h) if ((pos[j] - pos[i]) ** 2).sum() < h * h}               # 空間ハッシュ
candidates = len(table.query(pos[i], h))
print("近傍は同じ:", brute == hashed, " 候補の数:", candidates, "/ 全粒子:", len(pos))

# 計算量の感覚: 全ペアは N² 回、空間ハッシュは N × 平均の候補数
n = 400
print("全ペアの距離計算:", n * n, " ハッシュ(おおよそ):", n * candidates)
```

```text
# 出力
近傍は同じ: True  候補の数: 22 / 全粒子: 400
全ペアの距離計算: 160000  ハッシュ(おおよそ): 8800
```

## 4. 流体・VFX の用語の補強

| 用語 | 意味 | FluidKit での出番 |
|---|---|---|
| **ナビエ–ストークス方程式** | 流体の動きを表す基本の方程式(運動量の保存) | すべてのソルバーの土台 |
| **ラグランジュ法 / オイラー法** | 流れに乗って(粒子)見る / 場所を固定して(格子)見る | SPH / Stable Fluids |
| **密度・圧力・粘性** | 詰まり具合 / 押し返す力 / ねばり | SPH の力の計算 |
| **非圧縮(発散ゼロ)** | 体積が縮まない流れ。速度場の発散が 0 | 射影・カールノイズ |
| **CFL 条件(時間刻みの目安)** | 1 ステップで進む距離が、格子や粒子の間隔を超えない | `dt` の選び方 |
| **PIC / FLIP** | 格子の速度をそのまま / 速度の変化だけを、粒子に渡す | `FLIPSolver`(`alpha`) |
| **フレーム / キャッシュ(.vol・.abc)** | 1 コマ / 計算結果をファイルに保存したもの | 出力 |
| **Houdini / Blender / Unity VFX** | 映像・ゲームの 3D・エフェクトのツール | 連携 |
| **WebAssembly(WASM)** | ブラウザで、ネイティブ並みの速度で動かす形式 | `wasm/`(リアルタイム SPH) |
| **GNN(グラフニューラルネット)** | 粒子の近傍をグラフにして学ぶ AI | `neural/`(Neural Fluid) |
| **カールノイズ** | ノイズの回転から作る、発散ゼロの流れ(見た目用) | `curl_noise.py` |
| **ベンチマーク / 実測** | 時間を測ること | `PERF_RESULTS.md` |

## 確認問題

1. NumPy のベクトル化が速いのは、なぜですか。
2. `RandomState(seed)` を使うのは、なぜですか。
3. 空間ハッシュが、全ペアより速いのは、なぜですか。
4. 「粒子法」と「格子法」の、見方の違いは何ですか。

## 解答

1. ループを、C で書かれた配列全体の演算に任せるから(Python のループの遅さを避ける)。
2. 同じ seed で同じ数列になり、実験・テストを再現できるから。
3. 近くにない粒子を調べずに済み、調べる数が、粒子の数に比例する程度になるから。
4. 粒子法は流れに乗る粒子を追う(ラグランジュ)。格子法は場所を固定して、そこを通る値を見る(オイラー)。
