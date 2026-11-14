# 5. テストの書き方と実行

## 実行

```bash
python -m pytest -q FluidKit/tests                       # 全部(リポジトリのルートから。文書の検査を含む)
python -m pytest -q FluidKit/tests/test_solvers.py       # 物性テストだけ
python tools/check_docs.py                               # 文書の検査(標準ライブラリだけ)
```

## テストの置き場所と、何を確かめるか

| ファイル | 確かめるもの |
|---|---|
| `test_solvers.py` | 3 ソルバーの**物性**: SPH は運動量保存(重力なし)・重力のインパルスとの一致・粒子数の保存・NaN や暴走なし・境界を貫通しない。格子法は境界の外へ出ない・密度が有限で有界・NaN なし。FLIP は粒子数・境界・運動量の符号 |
| `test_reproducibility.py` | 同じシード+設定 → 同じハッシュ・別のシード → 別のハッシュ・ハッシュが状態の決定的な関数 |
| `test_io.py` | JSON・gzip・NPZ の**往復**(書いて読んで、同じ値)・gzip が小さい・後片付けが `tmp_path` だけ |
| `test_curl_noise.py` | 発散が小さい・格子を細かくすると h² で収束・時間依存でも発散ゼロ・同じシード → 同じ場 |
| `test_compare_solvers.py` | 全手法が**同じ指標(同じキー)**を返す・初期の重心が揃う・落下する塊が下向きの運動量を得る |
| `test_heightfield_js.py` | JavaScript の高さ場のテスト(`node --test`。**Node が無ければスキップ**) |
| `test_docs.py` | 文書の検査(下) |

## よく使う型

1. **物性で確かめる**(運動量・質量・境界)。**正解の数値を持たない**ソルバーでも、**成り立つべき性質**は書ける。
2. **小さな実行**(20〜50 粒子・10〜50 ステップ)。**モックを使わず、本物のソルバーを回す**。
3. **シードを固定**する(`np.random.RandomState(seed)`)。再現できないテストは、落ちたときに原因を追えない。
4. **許容誤差を、理由つきで書く**。運動量の保存は**機械精度に近い**(厳密に近い)ので小さく、収束のテストは**傾き**(h²)で見る。
5. **ファイルは `tmp_path`** に書く(リポジトリを汚さない)。

## 新しいソルバーのテストを書く手順

1. **成り立つべき性質を 1 行で言う**(例: 「重力なしなら、総運動量は変わらない」)。
2. それを**小さな実行**で確かめるテストを書く。
3. **わざと壊して、落ちる**ことを確かめる(通るだけのテストは、何も守らない)。
4. **再現性**(同じシード → 同じハッシュ)も、1 つ書く。

```python
# 実行できる
import numpy as np

from solver import Particle, SPHSolver

# 物性テストの型: 重力なしなら、総運動量は変わらない
rng = np.random.RandomState(0)
positions = rng.uniform(-0.3, 0.3, size=(30, 2))
particles = [Particle(pos=positions[i].copy(), vel=np.zeros(2)) for i in range(30)]
solver = SPHSolver(smoothing_length=0.15, gas_constant=80.0, viscosity=0.01, gravity=np.zeros(2))

p0 = sum((p.mass * p.vel for p in particles), start=np.zeros(2))
for _ in range(30):
    solver.step(particles, dt=0.005)
p1 = sum((p.mass * p.vel for p in particles), start=np.zeros(2))
print("運動量が保たれた:", bool(np.abs(p1 - p0).max() < 1e-9))
print("粒子数が保たれた:", len(particles) == 30)
print("NaN なし:", bool(np.isfinite([p.pos for p in particles]).all()))
```

```text
# 出力
運動量が保たれた: True
粒子数が保たれた: True
NaN なし: True
```

## 文書の検査

```bash
python tools/check_docs.py             # 検査(標準ライブラリだけ)
python tools/check_docs.py --update    # 引用の更新・実行例の出力の作り直し
python tools/check_mermaid.py          # Mermaid が描画できるか(mmdc が要る)
python tools/make_diagrams.py          # 構成図(SVG)を作り直す
```

`FluidKit/tests/test_docs.py` が、同じ検査を pytest で行います(引用がソースと一致・実行例が 2 回動いて出力が一致・講義の標準の節・リンク・Mermaid の標準行)。

## 確かめていないこと(正直に)

- **WebAssembly 版**(`wasm/`)・**Blender 連携**(`blender/`)・**GNN の学習と推論**(`neural/`)は、**自動テストと文書の検査に含まれない**(Emscripten・Blender・GPU が要る)。
- **物性テストは、必要条件**であって、**物理的に正しい**ことの証明ではない(運動量が保存しても、水らしく振る舞うとは限らない)。
- **性能の数字**(`PERF_RESULTS.md`)は、書かれた日・環境(RTX 3070・Python 3.12)の実測。**この資料の検査では再測していない**。
- **Viewer(Three.js)の見た目**は、手で開いて確かめる(`test_heightfield_js.py` は、高さ場の計算だけ)。
