# 3. 内側のしくみ

講義(第 1・2 回)で扱った「カーネルと空間ハッシュ」「運動量の保存」「状態ハッシュ」「物性テスト」は、繰り返しません。ここでは、**講義に出ない部分**を読みます。

## SPH の式の読み方

`SPHSolver` は Müller らの 2003 年の方法です。式と、コードの対応:

| 量 | 式(2D) | コード |
|---|---|---|
| 密度 | ρᵢ = Σ mⱼ W_poly6(rᵢⱼ) | `_compute_density`(`W_poly6` は**距離の二乗**を受け取り、`sqrt` を省く) |
| 圧力 | p = max(k(ρ − ρ₀), 0) | `step` の `prs`(**負の圧力は 0**にする。引っ張り合わない) |
| 圧力の力 | −Σ mⱼ (pᵢ+pⱼ)/2 / ρⱼ ∇W_spiky | `_compute_forces`(**対称化**した平均の圧力。作用・反作用になる) |
| 粘性の力 | μ Σ mⱼ (vⱼ−vᵢ)/ρⱼ ∇²W_visc | 同上 |
| 積分 | v ← v + dt·a、x ← x + dt·v(新しい v) | Symplectic Euler |

カーネルを**3 種類使い分ける**理由: 密度には滑らかな Poly6、圧力には**中心で勾配が 0 にならない** Spiky(0 だと、近づきすぎた粒子が反発しない)、粘性には**ラプラシアンが正**の専用カーネル。

```python
# 抜粋ファイル: FluidKit/solver/sph_solver.py
self._W_poly6_coeff     =  4.0 / (np.pi * h ** 8)
self._W_spiky_grad_coeff= -10.0 / (np.pi * h ** 5)
self._W_visc_lap_coeff  =  40.0 / (np.pi * h ** 5)
```

- **正規化係数は 2D 用**。3D にするなら係数を変える(別の式になる)。
- 密度の下限 `1e-6`(`np.maximum(rho, 1e-6)`)は、**ゼロ割を防ぐ**ため。近傍の無い粒子が、`rho = 0` で後の割り算を壊さない。
- 力の計算で、**自分自身(`nbrs_arr != i`)は除く**。密度の計算は、**自分を含む**(距離 0 のカーネル値が、自分の寄与)。

## Stable Fluids: 1 ステップの順序

```python
# 抜粋ファイル: FluidKit/solver/stable_fluids.py
self.u = self._diffuse(self.u, visc)
self.u, self.v = self._project(self.u, self.v)
self.u = self._advect(u0, u0, v0)
self.u, self.v = self._project(self.u, self.v)
self.dens = self._advect(self.dens, self.u, self.v)
```

**拡散 → 射影 → 移流 → 射影**。射影(圧力で発散を消す)を**2 回**やるのは、移流の後で、また発散が出るから。

- **移流**(`_advect`)は**セミラグランジュ**: 各格子点から速度で**逆にたどり**、その位置の値を補間で取る(`map_coordinates`、`order=1`)。**どんなに大きな `dt` でも発散しない**のが「安定」の意味(ただし拡散して、ぼやける)。
- **射影**(`_project`)は、ポアソン方程式を**赤黒の反復**(20 回)で解く。**反復の回数が少ないと、発散が残る**(速度と引き換え)。
- **共役勾配法(CG)に差し替える案は、試して差し戻した**(純ノイマン境界で行列が特異。N=128 で 500 ms 近くかかった)。**docstring に、数字つきで残っている**。同じ案を、もう一度試す前に読む。
- **格子は `(N+2)²`**(外側の 1 マスが境界)。内側は `[1:N+1, 1:N+1]`。**`N` ではなく `N+2`** を間違えるのが、最も多い事故。

## FLIP: PIC と FLIP の混合

`alpha`(既定 0.05)は **PIC の割合**。`vel_new = alpha·(PIC) + (1−alpha)·(FLIP)`。

- **PIC**(格子の速度を、そのまま粒子へ): 安定だが、**ぼやける**(粘っこくなる)。
- **FLIP**(格子の**速度の変化**を、粒子の速度に足す): 細かく動くが、**ノイズが出やすい**。
- 混ぜて、両方の欠点を抑える(`alpha = 0.05` は、FLIP 寄り)。
- 流れは **P2G(粒子 → 格子)→ 格子で重力と射影 → G2P(格子 → 粒子)→ 位置を進める**。
- 格子の射影は `_project_mac`(**簡易な反復を 20 回**)。`step` のコメントには「`StableFluids` の `_project` を流用」とあるが、**実際に呼ぶのは `_project_mac`**(別の実装。`__init__` で作る `_stam` は、`step` では使われていない)。
- `_clamp_positions` の docstring は「クランプ + 速度ゼロ化」だが、**コードは位置のクランプだけ**(速度は変えない)。

## 再現性: 何をハッシュにしているか

- **設定のハッシュ**: `json.dumps(config, sort_keys=True, default=str)` を SHA-256。**キーの順序に依存しない**。
- **状態のハッシュ**: 状態を **`float64` にそろえて** `tobytes()` を SHA-256。**バイト列が 1 ビットでも違えば別**。
- `SimulationRecord(config, seed)` は、**`np.random.seed(seed)`(グローバルな乱数)を設定する**。作るたびに、**プロセス全体の乱数の状態を上書きする**ので、別の場所で `np.random` を使っているコードに影響する。
- **履歴の全状態をメモリに持つ**(`_states`)。`save_npz` のためだが、長い実行・大きな状態では、メモリが増える。

## サンドボックス実行(`runner.py`)

```python
# 実行できる
import numpy as np

from solver import run_simulation_sandboxed, validate_config

# 1. 設定の検証: 不正な値は、エラーにならず「既定値か、範囲の端」に置き換わる
print(validate_config({"N": 9999, "dt": "x", "visc": -1, "n_frames": 0.5, "seed": -5, "extra": 1}))


class Counter:
    """step() が引数なしで状態の配列を返す、最小のソルバー。"""

    def __init__(self, cfg):
        self.x = np.zeros(3)

    def step(self):
        self.x = self.x + 1.0
        return self.x


class Broken(Counter):
    def step(self):
        raise RuntimeError("boom")


a = run_simulation_sandboxed({"n_frames": 5}, Counter)
b = run_simulation_sandboxed({"n_frames": 5}, Counter)
print("状態:", a["status"], "フレーム数:", a["frames"], "ハッシュの長さ:", len(a["state_hash"]))
print("同じ設定なら、最終状態のハッシュが一致:", a["state_hash"] == b["state_hash"])

c = run_simulation_sandboxed({"n_frames": 5}, Broken)
print("例外:", c["status"], c["frames"], c["error"])
```

```text
# 出力
{'N': 512, 'dt': 0.05, 'visc': 0.0, 'n_frames': 1, 'seed': 0, 'extra': 1}
状態: completed フレーム数: 5 ハッシュの長さ: 64
同じ設定なら、最終状態のハッシュが一致: True
例外: error 0 boom
```

- **`validate_config` は、不正な値を黙って直す**: `N=9999` → 512(上限)・`dt="x"`(数にできない)→ 既定の 0.05・`visc=-1` → 0.0(下限)・`n_frames=0.5` → `int` で 0 になり、下限の 1・`seed=-5` → 0。**未知のキー(`extra`)は、そのまま通る**。エラーにしないので、**打ち間違いに気づきにくい**(→ [既知の問題](06-troubleshooting-maintenance.md))。
- 戻り値は、`status`(`completed`・`timeout`・`error`)・`frames`・`state_hash`・`elapsed_ms`(エラーのときは `error`)。**例外は握りつぶして、辞書で返す**(呼び出し側は、例外処理が要らない)。
- **タイムアウト**: Unix は `SIGALRM`、**Windows は別スレッドのタイマーが、メイン スレッドへ例外を送る**(`PyThreadState_SetAsyncExc`)。**例外は、Python のコードを実行しているときにだけ届く**。長い C の呼び出し(`time.sleep` など)の間は、割り込めない(→ トラブル)。
- `audit_path` を渡すと、**完了したときだけ**監査ログを保存する(タイムアウト・エラーでは保存しない)。

## カールノイズ: 発散ゼロを、式で保証する

ノイズの回転(curl)を速度場にすると、**∇·(∇×A) = 0 が恒等的に成り立つ**ので、圧力の射影なしで非圧縮の流れになる。ノイズは「ランダムな向きの正弦波の和」で、**解析的に微分できる**ので、curl も解析式。

```python
# 実行できる
import numpy as np

from solver import CurlNoise2D, divergence_residual

field = CurlNoise2D(seed=3)
max_div, grad_scale = divergence_residual(field, (-1.0, -1.0), (1.0, 1.0))
print("発散は、速度勾配の大きさに比べて十分小さい:", max_div < 1e-4 * grad_scale)

p = np.array([[0.1, 0.2], [0.5, -0.3]])
print("同じシードなら同じ場:", np.allclose(CurlNoise2D(seed=3).velocity(p, 0.5), field.velocity(p, 0.5)))
print("別のシードなら別の場:", not np.allclose(CurlNoise2D(seed=4).velocity(p, 0.5), field.velocity(p, 0.5)))
```

```text
# 出力
発散は、速度勾配の大きさに比べて十分小さい: True
同じシードなら同じ場: True
別のシードなら別の場: True
```

- **物理(ナビエ–ストークス)は解いていない**。見た目用の近似で、**壁での反射・圧力・粘性は無い**。
- 発散は**ゼロではなく、有限差分の誤差**(測るときの `h` の O(h²))。`h` を小さくしすぎると、丸め誤差が支配的になる(docstring の注意)。

## Neural 用の正規化(`utils.py`)

位置を `[min, max] → [-1, 1]`、速度を `vel / (std · 3)`(±3σ が `[-1, 1]` におおむね収まる)。**学習(`collect_data.py`)と推論(`infer_v2.py`)で同じ式を使う**ため、1 か所に集めてある。**片方だけ式を変えると、推論の結果が静かにずれる**。

```python
# 実行できる
import numpy as np

from utils import denormalize_pos, denormalize_vel, normalize_pos, normalize_vel

pos = np.array([0.0, 5.0, 10.0])
print(normalize_pos(pos, 0.0, 10.0))
print(np.allclose(denormalize_pos(normalize_pos(pos, 0.0, 10.0), 0.0, 10.0), pos))

vel = np.array([-3.0, 0.0, 1.5])
print(normalize_vel(vel, vel_std=1.0))
print(np.allclose(denormalize_vel(normalize_vel(vel, 1.0), 1.0), vel))
```

```text
# 出力
[-1.  0.  1.]
True
[-1.   0.   0.5]
True
```
