# 2. コードの読み方

## どこから読むか(おすすめの順)

| 順 | ファイル | 見るところ |
|---|---|---|
| 1 | `solver/__init__.py` | 公開している名前の一覧(使う側は、ここからだけ import する) |
| 2 | `solver/sph_solver.py` | `SpatialHash` → カーネル 3 つ → `step`。**約 210 行で全部読める** |
| 3 | `solver/reproducibility.py` | 112 行。状態ハッシュと監査ログ |
| 4 | `solver/runner.py` | `validate_config` → `timeout` → `run_simulation_sandboxed` |
| 5 | `solver/stable_fluids.py` | `step` の順序(拡散 → 射影 → 移流 → 射影) |
| 6 | `solver/flip_solver.py` | P2G → グリッドで解く → G2P(PIC と FLIP の混合) |
| 7 | `tools/gen_sample.py` | プリセット・初期配置・壁での反射 |
| 8 | `tests/test_solvers.py` | 「何が成り立つべきか」の一覧(物性テスト) |
| 9 | `tools/compare_solvers.py` | 3 つのソルバーを同じ条件で比べる |
| 10 | `neural/`(`collect_data` → `train_v2` → `infer_v2`)・`utils.py` | Neural Fluid。正規化は `utils.py` に集約 |

## import の流儀(つまずきやすい)

`FluidKit` 内のモジュールは、**相対 import ではなく `sys.path` を足して**互いを見つけます(例: `gen_sample.py` は `solver/` を `sys.path` に足して `from sph_solver import ...`)。

- 使う側は `from solver import SPHSolver, Particle`(`FluidKit/` を `sys.path` に置く)。
- `tests/conftest.py` が、**`FluidKit/`・`tools/`・`solver/` の 3 つ**を `sys.path` に足す。テストからは、どちらの書き方でも import できる。
- 文書の実行例は、`tools/docs_pythonpath.txt`(中身は `FluidKit`)で、`FluidKit/` を `PYTHONPATH` に足して動かす。

## 1 ステップの流れ(SPH)

1. 粒子のリスト → `pos`・`vel`・`mass` の配列(`np.stack`)。
2. 空間ハッシュを**毎回作り直す**(`build`)。
3. 粒子ごとに近傍を引いて**密度** → **圧力**(`max(k(ρ−ρ₀), 0)`)。
4. 粒子ごとに**加速度**(重力 + 圧力の力 + 粘性の力)。
5. Symplectic Euler: **先に速度**、**新しい速度で位置**。結果を `Particle` に書き戻す。

**壁での反射はソルバーにない**(`SPHSolver` は境界を扱わない)。`gen_sample.py` の `SimpleSPH.step` が、ステップの後で、境界の外に出た粒子を戻して速度を反転(反発係数 0.3)する。

## 流儀(このコードの約束)

- **ソルバーは `solver/` に、表示・変換は `tools/` に**。`gen_sample.py` のように、`tools/` が `solver/` を呼ぶ向きだけ(逆は無い)。
- **コメントに「なぜ」と「実測」を書く**(`stable_fluids.py` の `_project` に、CG を試して差し戻した記録がある)。変えるときも同じ粒度で残す。
- **数値の挙動を変えない共通化**(`utils.py` は、元の式と完全に同じ)。変えるなら、テストで**前後が同じ**ことを示す。
- **乱数はシードを渡す**(`seed`)。再現できないテストは書かない。
