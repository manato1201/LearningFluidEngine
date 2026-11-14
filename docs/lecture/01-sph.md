# 第 1 回 SPH: カーネルと空間ハッシュ

> 所要時間の目安: 3.5 時間 / 前提: 第 0 回・ベクトルの内積・密度と圧力の基本

## この回のゴール

- SPH の考え方(**周りの粒子の寄与を、カーネルで重みづけて足す**)を説明できる。
- **空間ハッシュ**で、近傍探索が O(N²) から平均 O(N) になる理由を説明できる。
- 1 ステップの流れ(ハッシュ → 密度 → 圧力 → 力 → 積分)と、**運動量が保存される**理由を説明できる。

## 1. 考え方: カーネルで足す

粒子 i の**密度**は、周りの粒子の質量を、**距離に応じた重み**で足して求めます。

```text
ρ_i = Σ_j  m_j · W(|r_i − r_j|, h)
```

`W` が**カーネル**(滑らかに 0 へ落ちる関数)、`h` が**影響半径**(`smoothing_length`)。`h` より遠い粒子は、重みが **0**。

| カーネル | 使いどころ |
|---|---|
| **Poly6** | 密度(距離の**二乗**を受け取る。`sqrt` を省ける) |
| **Spiky の勾配** | 圧力の力(中心でも勾配が 0 にならない = 粒子が**近づきすぎない**) |
| **Viscosity のラプラシアン** | 粘性(常に正 = 速度を**なめらかに**する) |

```python
# 引用: FluidKit/solver/sph_solver.py::SPHSolver._W_poly6
def _W_poly6(self, r2: np.ndarray) -> np.ndarray:
    """Poly6 カーネル (距離の二乗を受け取る — sqrt 省略で高速化)。"""
    h2 = self.h ** 2
    mask = r2 < h2
    val = np.zeros_like(r2)
    diff = h2 - r2[mask]
    val[mask] = self._W_poly6_coeff * diff ** 3
    return val
```

- `mask = r2 < h2`: **半径の内側だけ**を計算する。
- `(h² − r²)³`: 距離が `h` に近づくと、滑らかに 0 へ。

**圧力**は、密度が基準(`rest_density`)より**高い**ときだけ生まれます(`k · (ρ − ρ0)`、負なら 0)。圧縮されたら押し返す、という**状態方程式**です。

## 2. 近傍探索の高速化: 空間ハッシュ

粒子 N 個の全ペア(N²)を調べると、遅い。**影響半径 `h` より遠い粒子は、重みが 0** なので、**近い粒子だけ**を探せば足ります。

**空間ハッシュ**: 空間を、**1 辺 `h` の格子**に区切り、粒子を**セル**に振り分けます。ある粒子の近傍は、**自分のセルと隣の 3 × 3 セル**にしかいません。

```python
# 引用: FluidKit/solver/sph_solver.py::SpatialHash
class SpatialHash:
    """
    2D 空間ハッシュ。
    セルサイズ = カーネル半径 h とすることで、
    近傍候補を 9 セル (3×3) だけに絞れる。
    """

    def __init__(self, cell_size: float, n_buckets: int = 4096):
        self.cell_size = cell_size
        self.n_buckets = n_buckets
        self._table: dict[int, list[int]] = {}

    def _hash(self, cx: int, cy: int) -> int:
        return (cx * 92837111 ^ cy * 689287499) % self.n_buckets

    def build(self, positions: np.ndarray) -> None:
        """全粒子位置からハッシュテーブルを再構築 O(N)。"""
        self._table = {}
        cs = self.cell_size
        for idx, pos in enumerate(positions):
            cx, cy = int(pos[0] / cs), int(pos[1] / cs)
            key = self._hash(cx, cy)
            self._table.setdefault(key, []).append(idx)

    def query(self, pos: np.ndarray, r: float) -> list[int]:
        """pos から距離 r 以内の粒子インデックスを返す O(k)。"""
        cs = self.cell_size
        cx0, cy0 = int((pos[0] - r) / cs), int((pos[1] - r) / cs)
        cx1, cy1 = int((pos[0] + r) / cs), int((pos[1] + r) / cs)
        result: list[int] = []
        for cx in range(cx0, cx1 + 1):
            for cy in range(cy0, cy1 + 1):
                key = self._hash(cx, cy)
                result.extend(self._table.get(key, []))
        return result
```

- `build`: 全粒子を 1 回なでて、セルに入れる(**O(N)**)。
- セル番号 → **ハッシュ関数**(大きな素数を掛けて XOR・`% n_buckets`)で、バケツ番号にする。セルの数が膨大でも、**メモリは粒子の数ぶん**で済む。
- `query`: 探す範囲のセルだけを見る(**O(k)**)。
- **衝突**: 別のセルが同じバケツに入ることがある。返る候補には**遠い粒子が混ざる**ので、**後で距離を確かめる**(カーネルが 0 を返すので、結果は変わらない)。

```python
# 実行できる
import numpy as np

from solver import SpatialHash

rng = np.random.default_rng(0)
positions = rng.random((300, 2))          # 単位正方形に 300 粒子
h = 0.1

table = SpatialHash(cell_size=h)
table.build(positions)

same = 0
for i in range(300):
    brute = set(np.nonzero(((positions - positions[i]) ** 2).sum(axis=1) < h * h)[0])             # 全ペア
    fast = {j for j in table.query(positions[i], h) if ((positions[j] - positions[i]) ** 2).sum() < h * h}  # ハッシュ + 距離の確認
    same += brute == fast
candidates = np.mean([len(table.query(positions[i], h)) for i in range(300)])
print("全ペアと同じ近傍になった粒子:", same, "/ 300")
print("1 粒子あたりの候補数(平均):", round(float(candidates), 1), "(全ペアなら 300)")
```

```text
# 出力
全ペアと同じ近傍になった粒子: 300 / 300
1 粒子あたりの候補数(平均): 24.4 (全ペアなら 300)
```

近傍の**結果は、全ペアと同じ**で、調べる粒子は**数十個**に減ります(約 47 倍速くなった、とコードのコメントにある実測は、N = 500 の場合)。

## 3. 1 ステップと、保存則

`SPHSolver.step`:

1. **空間ハッシュを作り直す**(粒子が動くので、毎回)。
2. **密度**を求め、**圧力**にする。
3. **加速度**(重力 + 圧力の力 + 粘性の力)。
4. **Symplectic Euler**(先に速度を更新し、**新しい速度**で位置を更新)。

圧力・粘性の力は、粒子ペアの**作用・反作用**(`r_ij = −r_ji`)で作られているので、**外力(重力)を除くと、総運動量は保存**します。

```python
# 実行できる
import numpy as np

from solver import Particle, SPHSolver

# 5 x 8 の格子状に並べた粒子。重力なし(外力なし)で 20 ステップ
particles = [Particle(pos=np.array([0.1 + 0.05 * (i % 5), 0.1 + 0.05 * (i // 5)]), vel=np.array([0.1 * (i % 3), 0.0]))
             for i in range(40)]
solver = SPHSolver(smoothing_length=0.1, gravity=np.zeros(2))

before = sum(p.mass * p.vel for p in particles)
for _ in range(20):
    solver.step(particles, dt=0.001)
after = sum(p.mass * p.vel for p in particles)
print("総運動量(前):", np.round(before, 6))
print("総運動量(後):", np.round(after, 6))
print("ずれ:", float(np.abs(after - before).max()) < 1e-9)
```

```text
# 出力
総運動量(前): [3.9 0. ]
総運動量(後): [3.9 0. ]
ずれ: True
```

内部の力だけでは、**運動量は増えも減りもしません**。もしずれたら、力の計算に**バグ**がある(作用・反作用が崩れている)。だから、これは**良い単体テスト**になります。

## 確認問題

1. 影響半径 `h` を、空間ハッシュのセルの大きさにするのは、なぜですか。
2. ハッシュの衝突で、遠い粒子が候補に混ざっても、結果が変わらないのは、なぜですか。
3. 重力が無いとき、総運動量が保存されるのは、なぜですか。

## 解答

1. 近傍は、必ず**自分のセルと隣の 3 × 3 セル**の中にいるから(それ以上遠いと重みが 0)。
2. 後でカーネルが、`h` より遠い粒子に **0** を返す(寄与しない)から。
3. 圧力・粘性の力が、粒子ペアで**大きさが等しく逆向き**(作用・反作用)で、足すと打ち消し合うから。

## まとめ

- SPH = 周りの粒子を、**カーネル**で重みづけて足す。圧力は、基準密度より高いときだけ。
- 近傍探索は **空間ハッシュ**(セル = `h`・3 × 3 を見る)で、全ペアと同じ結果を、少ない候補で得る。
- 内部の力は**作用・反作用**で、重力を除けば**運動量が保存**する(良いテストになる)。
