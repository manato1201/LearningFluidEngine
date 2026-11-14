# 4. 拡張のレシピ

どのレシピも、**同じ順**で進めます: ①挙動を決める → ②**性質のテストを先に** → ③実装 → ④公開(`__init__.py`)→ ⑤文書。

## レシピ A: 新しいソルバーを足す

`solver/` に新しいファイルを作り、`solver/__init__.py` の `import` と `__all__` に足す。

**サンドボックスに載せるなら**、`runner.py` の約束を守る: **`solver_factory(cfg)` がソルバーを返し、`solver.step()` が、引数なしで「状態の配列」を返す**。

```python
# 実行できる
import numpy as np

from solver import run_simulation_sandboxed


class Drift:
    """一定の速度で流れる粒子(最小のソルバー)。cfg の dt を使う。"""

    def __init__(self, cfg):
        self.dt = cfg["dt"]
        self.pos = np.linspace(0.0, 1.0, 4)

    def step(self):
        self.pos = self.pos + self.dt * 2.0     # 速度 2
        return self.pos                          # 状態の配列を返す(ハッシュの対象)


result = run_simulation_sandboxed({"dt": 0.1, "n_frames": 3}, Drift)
print(result["status"], result["frames"])

# dt を 1e-9 にしても、検証で下限の 1e-5 に直される(エラーにならない)
again = run_simulation_sandboxed({"dt": 1e-9, "n_frames": 3}, Drift)
print(again["status"], again["state_hash"] != result["state_hash"])
```

```text
# 出力
completed 3
completed True
```

- **`SPHSolver` や `FLIPSolver` を、そのままは載せられない**(`step` の形が違う)。包むクラスを書いて、`step()` が状態を返すようにする(`gen_sample.py` の `SimpleSPH` が、その見本)。
- 返す状態は、**ハッシュに使う全て**にする。位置だけを返すと、速度の違いが**ハッシュに出ない**。
- **足したら**: ①物性テスト(運動量・境界・NaN → `tests/test_solvers.py`)②再現性(同じシード → 同じハッシュ)③`compare_solvers.py` に載せるなら、`Runner` クラス(`advance`・`metrics`・`snapshot`)。

## レシピ B: プリセットを足す(`gen_sample.py`)

1. `PRESETS` に、辞書を足す(`description`・`gravity`・`particle_count`・`h`・`stiffness`・`rest_density`・`viscosity`・`bounds`・`init`)。
2. **初期配置の関数**(`_init_xxx(n, bounds)` が `(pos, vel)` を返す)を書き、`INITIALIZERS` に登録する。
3. `python tools/gen_sample.py --preset xxx --frames 20` で、出力を Viewer で見る。

**調整の目安**(SPH の一般的な経験則。このリポジトリでは実測していない): `h`(カーネル半径)は、粒子の間隔の 1.5〜2 倍ほど。`stiffness` が大きすぎると**爆発**し、小さすぎると**つぶれる**。`dt` は、`stiffness` が大きいほど小さく。**既存のプリセットの値は、見た目で決めた値**(式から導いたものではない)。

## レシピ C: 設定の項目を足す(`validate_config`)

1. `_CONFIG_SCHEMA` に `"key": (型, 最小, 最大)` を足す。
2. `validate_config` の **`defaults` にも既定値を足す**(**2 か所**。片方だけだと、`KeyError`)。
3. テスト: 範囲外・型の違う値・欠けた値が、**どう直るか**を書く。

**注意**: 検証は、不正な値を**黙って直す**。エラーにしたい項目を足すときは、その項目だけ `ValueError` を出す分岐が要る(既存の振る舞いは変えない)。

## レシピ D: 比較の指標を足す(`compare_solvers.py`)

`metrics()` の返す辞書(いまは `mass_ratio`・`ke`・`py`・`com_y`)に、**キーを足す**。全ソルバーの `metrics()` が、**同じキーを返す**ことが、テスト(`test_all_methods_report_same_metric_keys`)で確かめられている。**1 つのソルバーだけに足すと、このテストが落ちる**ので、全部に足す(該当しないものは `None` などで)。

## レシピ E: 状態の記録を足す(`SimulationRecord`)

`log_step(step, state, extra={...})` の `extra` は、**監査ログにそのまま入る**(JSON にできる値だけ)。**ハッシュに入れたいもの**は、`state` の側に含める(`extra` はハッシュの対象ではない)。

## 足したら、必ずやること

1. `python -m pytest -q FluidKit/tests`(リポジトリのルートから。物性テスト・再現性・文書の検査)。
2. `python tools/check_docs.py`(文書の引用・実行例が、ソースと合っているか)。
3. **性能に関わる変更**は、測り直して `PERF_RESULTS.md` に残す(その記録は、**測った日・環境**つき。数字を書き換えるときも、環境を書く)。
