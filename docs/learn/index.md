# LearningFluidEngine 学習資料(簡易版)

保守・拡張の詳しい資料は**未作成**です。基礎は [講義](../lecture/index.md)(第 0〜2 回)。既存の資料: [ALGORITHMS.md](../../FluidKit/docs/ALGORITHMS.md)・[PERF_RESULTS.md](../../FluidKit/docs/PERF_RESULTS.md)・`FluidKit/docs/lecture.html`。旧 README(構成・フェーズ別の使い方)は [setup-and-phases.md](../reference/setup-and-phases.md)。

## コードの地図(`FluidKit/`)

| 場所 | 役割 |
|---|---|
| `solver/sph_solver.py` | SPH + 空間ハッシュ(`SpatialHash`・`SPHSolver`・`Particle`) |
| `solver/stable_fluids.py` | Stam の Stable Fluids(格子法・numpy) |
| `solver/flip_solver.py` | FLIP(粒子 + MAC 格子) |
| `solver/reproducibility.py` | `SimulationRecord`(状態ハッシュ・監査ログ) |
| `solver/runner.py` | タイムアウト付きのサンドボックス実行・設定の検証 |
| `solver/curl_noise.py` | カールノイズ(発散ゼロの流れ) |
| `tools/` | データ生成(`gen_sample.py`)・変換(`xyz_to_json.py`)・比較(`compare_solvers.py`) |
| `neural/` | GNN(Neural Fluid): `collect_data` → `train_v2` → `infer_v2` |
| `wasm/`・`viewer/`・`blender/` | WebAssembly のリアルタイム SPH・Three.js の Viewer・Blender 連携 |
| `tests/` | 物性テスト(`test_solvers.py`)・再現性(`test_reproducibility.py`)・入出力(`test_io.py`) |

## 既知の問題・注意(講義の作成中に読んで見つけたもの。コードは変えていない)

1. **`SimulationRecord.verify` は、ステップ数の違いを見ない**(`zip` で短いほうまで比べる)。片方が途中で止まっても、一致と判定される。
2. **ハッシュ一致は、ビット単位の一致**。別のマシン・NumPy の版で、丸めがわずかに違うだけで不一致になる(記録に NumPy の版を残している)。
3. **`SpatialHash` は、粒子ごとの Python のループ**(`build`・`query`)。NumPy のベクトル化は、密度・力の計算の中だけ。粒子が増えると、ここが支配的になる(実測は PERF_RESULTS.md)。
4. 旧 README の性能の数字(「47 倍」など)は、**書かれた時点の実測**。環境で変わる。
5. WebAssembly・Blender・GNN の学習は、この文書の検査では**実行していません**。
