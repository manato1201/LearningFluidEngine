# 第 2 回 再現性(状態ハッシュ)とテスト

> 所要時間の目安: 2.5 時間 / 前提: 第 0〜1 回

## この回のゴール

- 「同じ設定なら同じ結果」を、**ハッシュで確かめる**方法を説明できる。
- ハッシュ比較の**落とし穴**(浮動小数点・比較の範囲)を、動かして確かめられる。
- 物性テストを、自分で足せる。

## 1. 状態ハッシュ

シミュレーションが**再現できる**とは、同じ設定・同じシードで、**同じ結果**になること。結果の配列そのものを比べる代わりに、**SHA-256 のハッシュ**(指紋)を比べます。

```python
# 引用: FluidKit/solver/reproducibility.py::SimulationRecord.log_step
def log_step(
    self,
    step: int,
    state: np.ndarray,
    *,
    extra: dict | None = None,
) -> None:
    """ステップごとに状態ハッシュを記録。"""
    state_arr = np.asarray(state, dtype=np.float64)
    entry: dict[str, Any] = {
        "step":       step,
        "state_hash": self._hash_array(state_arr),
        "elapsed_ms": (time.perf_counter_ns() - self._start_ns) / 1e6,
    }
    if extra:
        entry.update(extra)
    self._steps.append(entry)
    self._states.append(state_arr.copy())
```

- 設定(`config`)は **キーを並べ替えて JSON にしてからハッシュ**(`sort_keys=True`)。書き順が違っても、同じ設定なら同じハッシュ。
- 各ステップの状態配列を **`float64` にそろえて**ハッシュ。
- NumPy のバージョン・シードも記録する。

```python
# 実行できる
import numpy as np

from solver import SimulationRecord


def run(seed, steps=5, perturb=0.0):
    rng = np.random.default_rng(seed)
    state = rng.random((4, 2))
    rec = SimulationRecord({"solver": "demo", "dt": 0.01}, seed=seed)
    for step in range(steps):
        state = state + 0.01 * np.sin(state)
        if perturb and step == 3:
            state = state + perturb              # 4 ステップ目で、わずかにずらす
        rec.log_step(step, state)
    return rec


a, b = run(1), run(1)
print("同じ設定・同じシード:", a.verify(b))
print("1e-15 のずれ(丸め誤差ほど):", a.verify(run(1, perturb=1e-15)))
print("別のシード:", a.verify(run(2)))
short = run(1, steps=3)
print("ステップ数が違う(3 と 5):", a.verify(short), "← 短いほうまでしか比べない")
```

```text
# 出力
同じ設定・同じシード: True
1e-15 のずれ(丸め誤差ほど): False
別のシード: False
ステップ数が違う(3 と 5): True ← 短いほうまでしか比べない
```

注目: **1e-15 のわずかなずれでも、ハッシュは完全に変わります**(ビットが 1 つ違えば別の指紋)。だから、ハッシュ一致は**「ビット単位で同じ」**という、とても強い保証です。逆に、**別のマシン・NumPy の版で、丸めがわずかに違うだけで不一致**になります(記録に NumPy の版を残すのは、そのため)。

**落とし穴**: `verify` は、2 つの記録を `zip` で比べるので、**ステップ数が違っても、短いほうの分だけ一致すれば `True`** になります(上の最後の行)。ステップ数も比べる必要がある(学習資料の「既知の問題」)。

## 2. 物性テスト

`tests/test_solvers.py` は、**小さな系を、本物のソルバーで**動かし、必ず成り立つ性質を確かめます(モックなし)。

| 性質 | 意味 |
|---|---|
| **運動量の保存**(SPH・重力なし) | 内部の力が作用・反作用になっている(第 1 回) |
| **重力あり**: 既知の外力インパルス | 運動量の変化 = `g · dt · ステップ数 · 総質量` |
| **質量の保存** | 粒子の数・質量が変わらない |
| **境界を貫通しない** | 全粒子が、箱の中に残る |

**性質をテストする利点**: 「この数値になる」という**答えの暗記**でなく、**成り立つべき関係**を確かめるので、ソルバーを速くしても、壊れていなければ通る。

## 確認問題

1. 設定を JSON にするときに `sort_keys=True` にするのは、なぜですか。
2. 1e-15 のずれでも、ハッシュが変わるのは、なぜですか。それは長所ですか、短所ですか。
3. `verify` で、ステップ数の違いが見逃されるのは、なぜですか。

## 解答

1. 辞書のキーの**並び順が違うだけ**で、別のハッシュにならないように。
2. ハッシュは、入力のビットが 1 つでも違えば、まったく別の値になる。**ビット単位の一致を保証できる長所**で、**丸め誤差でも不一致になる短所**でもある。
3. `zip` が、短いほうの長さで止まるから。長さの一致も、確かめる必要がある。

## まとめ

- 再現性は、**設定と各ステップの状態の SHA-256** で確かめる。ビット単位で同じことが保証される(マシン・版が違うと不一致になりうる)。
- `verify` は、長さを比べない(既知の問題)。
- 物性テスト(保存則・境界)は、答えの暗記でなく、**必ず成り立つ性質**を確かめる。
