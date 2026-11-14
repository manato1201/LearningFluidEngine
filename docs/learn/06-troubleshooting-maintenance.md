# 6. トラブルと保守

## 症状から探す

| 症状 | 原因の候補 | 確かめ方・直し方 |
|---|---|---|
| **粒子が爆発する・NaN になる** | `dt` が大きすぎる・`gas_constant`(`stiffness`)が大きすぎる・粒子が重なって密度が極端 | `dt` を小さく・剛性を下げる。初期配置の間隔を `h` と比べる |
| **粒子がつぶれて、水にならない** | 剛性が小さすぎる・`rest_density` が粒子の並びに合っていない | `rest_density` と、初期の密度(`Particle.density`)を見比べる |
| **煙が下に落ちる** | `smoke` は重力が**上向き**(浮力の代わり)。向きを間違えている | プリセットの `gravity` |
| **設定が効かない・既定のまま動く** | `validate_config` が、不正な値を**黙って直した**(打ち間違いのキーは、無視されて通る) | `validate_config(cfg)` の戻り値を見る |
| `timeout` なのに、**上限より長くかかる** | **長い C の呼び出し**(`time.sleep`・大きな numpy の演算)の間は、割り込めない | 下の「既知の問題」の 3 |
| **ハッシュが別の PC で一致しない** | NumPy の版・CPU が違うと、丸めが違う(ビット単位の一致を見ている) | 監査ログの `numpy_version` を比べる |
| `solver` が import できない | `FluidKit/` が `sys.path` に無い | リポジトリのルートから `python -m pytest FluidKit/tests`、または `PYTHONPATH=FluidKit` |
| `test_heightfield_js` がスキップされる | Node が無い | 仕様(Node を入れれば動く) |
| WebAssembly が動かない | ビルドが無い・COOP/COEP のヘッダーが無い | `python FluidKit/setup/download_build.py`・`wasm/serve.py` で起動 |
| 文書の検査が落ちる | ソースが変わって、引用・実行例の出力が古い | `python tools/check_docs.py --update` のあと、**差分を読む** |

## 既知の問題・注意(読んで見つけたもの。コードは変えていない)

1. **`SimulationRecord.verify` は、ステップ数の違いを見ない**(`zip` で短いほうまで比べる)。片方が途中で止まっても、一致と判定される。
2. **ハッシュ一致は、ビット単位の一致**。別のマシン・NumPy の版で、丸めがわずかに違うだけで不一致になる(記録に NumPy の版を残している)。
3. **Windows のタイムアウトは、長い C の呼び出しの間は効かない**。別スレッドのタイマーが、メイン スレッドへ例外を送る方式で、**例外は Python のコードを実行しているときにだけ届く**。実測では、Python のループは上限(1 秒)で止まり、`time.sleep(3)` のステップは **3 秒かかってから** `timeout` になった。
4. **`SpatialHash` は、粒子ごとの Python のループ**(`build`・`query`)。NumPy のベクトル化は、密度・力の計算の中だけ。粒子が増えると、ここが支配的になる(実測は `PERF_RESULTS.md`)。
5. 旧 README の性能の数字(「47 倍」など)は、**書かれた時点の実測**。環境で変わる。
6. **`validate_config` は、不正な値を黙って直す**(`n_frames=0.5` → `int` で 0 → 下限の 1・`seed=-5` → 0・数にできない値 → 既定値)。**未知のキーはそのまま通る**ので、打ち間違いに気づかない。
7. **`SimulationRecord(...)` は、グローバルな `np.random` のシードを設定する**。作るたびに、プロセス全体の乱数の状態が上書きされる。
8. **`gen_sample.py` は、2D 用の `SPHSolver` に 3 次元の座標を渡している**。カーネルの正規化係数は 2D の式で、空間ハッシュは x・y だけを見る(z が違う粒子も候補に入る。距離は 3 次元で測るので、結果は正しいが、候補は多い)。プリセットの剛性(80〜100)は、`SPHSolver` の既定(2000)と違い、**見た目で決めた値**。
9. **FLIP の注意書きと実装の食い違い**: `step` のコメントは「`StableFluids` の `_project` を流用」だが、実際は `_project_mac`。`_clamp_positions` の docstring は「速度ゼロ化」だが、コードは位置のクランプだけ。`_project_mac` の反復は、docstring では Gauss-Seidel だが、**更新に前の反復の値だけを使う形**(Jacobi 法に近い)。
10. **`run_simulation_sandboxed` は、完了したときだけ監査ログを保存する**。タイムアウト・エラーのときは、途中までの記録も残らない。
11. WebAssembly・Blender・GNN の学習は、この文書の検査では**実行していません**。

```python
# 実行できる
import numpy as np

from solver import SimulationRecord

full = SimulationRecord({"case": "demo"}, seed=1)
partial = SimulationRecord({"case": "demo"}, seed=1)
for step in range(3):
    full.log_step(step, np.arange(4.0) + step)
partial.log_step(0, np.arange(4.0))        # 1 ステップで止まった実行

print("3 ステップ対 1 ステップ:", full.verify(partial))
print("設定が違えば不一致:", full.verify(SimulationRecord({"case": "other"}, seed=1)))
```

```text
# 出力
3 ステップ対 1 ステップ: True
設定が違えば不一致: False
```

## 定期の保守

| いつ | 何を |
|---|---|
| 依存(NumPy・SciPy)を上げたとき | **ハッシュが変わる**ことがある。再現性のテストと、保存済みの監査ログの扱いを確かめる |
| ソルバーを変えたとき | `test_solvers.py`・`test_compare_solvers.py`。**性能の数字を書き換えるときは、測った環境も** |
| Viewer・データ形式を変えたとき | `frames.json` の形(`metadata`・`frames`・`speeds`)を、生成側と Viewer の両方で |
| WASM のリリース | `release.yml` の Emscripten ビルドと、`download_build.py` の取得先を揃える |
| 文書を触ったとき | `python tools/check_docs.py` |

## 直す前の確認

1. **数値の挙動を変える修正は、前後の出力を比べる**(`utils.py` のように「完全に同じ」を目指すなら、テストで示す)。
2. **保存則のテストが落ちたら、まず式の符号と、作用・反作用の対称性**を疑う。
3. **既知の問題を直すときは、この一覧と、関連する講義・学習資料の記述を一緒に直す**(文書の検査が、実行例の食い違いを見つける)。
