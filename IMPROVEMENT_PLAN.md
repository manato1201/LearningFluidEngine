# LearningFluidEngine 改善・リファクタリング計画書

**改善指標: 各ツールの性能強化・動作効率の向上**
作成日: 2026-07-03 / 調査範囲: FluidKit全体(Python 3,741行 + C++ 257行)

---

## Phase 0: 現状分析(調査済み)

### 構成
| モジュール | 内容 | 最適化状況 |
|---|---|---|
| `FluidKit/solver/` | StableFluids2D / SPHSolver / FLIPSolver | ✅ 最適化済み(47〜80倍改善実績) |
| `FluidKit/tools/gen_sample.py` (299行) | Viewer用データ生成(SimpleSPH内蔵) | ❌ **純Python O(N²) — 最大の改善対象** |
| `FluidKit/neural/` | GNN流体モデル(v1/v2並存) | ⚠️ KNNがO(N²)、v1がデッドコード化 |
| `FluidKit/wasm/sph_wasm.cpp` | ブラウザ用SPH | ✅ SoA・C++で十分高速 |
| `FluidKit/viewer/` | Three.js表示(JSON 4〜8MB/プリセット) | ⚠️ 非圧縮JSON |

### 利用可能な既存資産(コピー元)
- **`solver/sph_solver.py`**: 空間ハッシュ済みSPH(N=500で18ms/step)。`SpatialHash.build()`(L43-50)、`_compute_density()`(L162-173)
- **`solver/flip_solver.py`**: `np.add.at` scatter(L106-164)、`scipy.ndimage.map_coordinates`補間(L170-206)
- **`solver/reproducibility.py`**: SHA-256による結果検証 — リファクタ前後の同一性確認に使える

### アンチパターン(全フェーズ共通)
- solver/は既に高度に最適化済み。「書き直し」ではなく既存solverの**再利用**を優先する
- 数値計算の変更は`reproducibility.py`のハッシュ or 統計値(粒子位置の平均/分散)で前後比較する。目視のみの確認は不可
- torch/scipyのAPIはrequirements記載バージョンの範囲で使う

---

## Phase 1: gen_sample.py の solver 再利用化(効果最大: 約100倍)

**現状:** `tools/gen_sample.py:137-227` の `SimpleSPH` クラスが純Python二重ループ(N=1500 × 100フレーム ≈ 2.25億相互作用、`math.sqrt`を毎回呼ぶ)。同等機能の`solver/sph_solver.py`(空間ハッシュ+numpy、47倍高速)が既に存在する。

**実装内容:**
1. `gen_sample.py` の `SimpleSPH` を削除し、`from solver.sph_solver import SPHSolver` に置換
2. プリセット定義(初期配置・重力・境界)はgen_sample.py側に残し、ステップ実行のみsolverに委譲
3. 出力JSONのフォーマット(`frames`/`speeds`/`metadata`)は**完全互換を維持**(viewerが依存)

**検証チェックリスト:**
- [ ] 全プリセットでJSON生成が完走し、`frameCount`/`particleCount`が変更前と一致
- [ ] viewer/index.htmlで再生し破綻がないこと
- [ ] 生成時間の前後比較を記録(目標: 5分→10秒未満)

**アンチパターン:** SimpleSPHとSPHSolverはカーネル定数が微妙に異なる可能性がある。粒子挙動の完全一致は求めず「物理的に妥当な水滴挙動」を合格基準とする(README記載のプリセット見た目と比較)。

---

## Phase 2: Neural Fluid KNNグラフの O(N log N) 化(GPU メモリ 約10分の1)

**現状:** `neural/model_v2.py:33-39` の `knn_graph()` が毎forward で `(B,N,N,3)` テンソルを生成(O(N²)メモリ+計算)。学習全体で約120万回呼ばれる。

**実装内容:**
1. 学習データは事前計算可能 — `collect_data.py` で `scipy.spatial.cKDTree` により近傍インデックスを前計算し、データセットに含める:
   ```python
   from scipy.spatial import cKDTree
   tree = cKDTree(pos)
   _, idx = tree.query(pos, k=k+1)
   idx = idx[:, 1:]  # 自身を除外
   ```
2. 推論時(`infer_v2.py`)はフレームごとに同cKDTreeで計算(CPUでもN=400なら十分高速)
3. `model_v2.py` のforwardは近傍インデックスを引数で受け取る形にシグネチャ変更

**検証チェックリスト:**
- [ ] 小規模学習(5エポック)でlossカーブが変更前と同等水準
- [ ] `torch.cuda.max_memory_allocated()` の前後比較を記録
- [ ] forward1回の時間を前後比較(目標: 4〜6倍短縮)

**アンチパターン:** cKDTreeのquery結果は自分自身(距離0)を含む — 必ず先頭を除外する。GPUテンソルを直接cKDTreeに渡さない(`.cpu().numpy()`必須)。

---

## Phase 3: ソルバー微修正 + 圧力投影の反復削減

**実装内容:**
1. **軽微(即実施)**: `solver/sph_solver.py:190` の `np.array([j for j in nbrs if j != i])` をnumpyマスク化(5-10%改善)
2. **圧力投影のCG法化(要検証)**: `stable_fluids.py:126-134` と `flip_solver.py:219-227` の20回Gauss-Seidel反復を `scipy.sparse.linalg.cg` に置換。ポアソン行列は5点ステンシルの疎行列として`scipy.sparse.diags`で構築

**検証チェックリスト:**
- [ ] `reproducibility.py`で数値挙動の比較(CG化は厳密一致しないため、発散量の残差ノルムが同等以下であることを基準にする)
- [ ] N=64/128でのstep時間を前後比較(目標: 2〜5倍)
- [ ] 発散チェック: `_project`後の速度場divergenceの最大値が従来以下

**アンチパターン:** CG法は境界条件の扱いを誤ると静かに間違う。`_set_bnd()`の適用タイミングを既存実装と揃え、残差で必ず検証する。数値的に不安定なら本項は見送り可(独立タスク)。

---

## Phase 4: I/O・メモリ効率(工数小の積み上げ)

| 施策 | 対象 | 実装 | 効果 |
|---|---|---|---|
| JSON gzip化 | `gen_sample.py:276` | `gzip.open(path + '.gz', 'wt')` + viewer側で`DecompressionStream`対応 | 転送量 50-70%減 |
| データセットmmap | `train_v2.py:60-61` | `np.load(..., mmap_mode='r')` | ホストメモリ約10分の1 |
| チェックポイントfp16保存 | `train_v2.py:121` | state_dictをhalf化して保存、ロード時にfloat昇格 | 30MB→15MB |
| NPZ圧縮 | `reproducibility.py:95` | `np.savez_compressed` | ディスク約50%減 |

**検証チェックリスト:**
- [ ] gzip化後もviewerが正常再生(非対応ブラウザ向けに非圧縮フォールバックを残す)
- [ ] mmap化後の学習1エポックのlossが変更前と一致(データ読み込みの正しさ確認)
- [ ] fp16チェックポイントからの推論結果が許容誤差内(相対誤差 <1e-3)

---

## Phase 5: リファクタリング(重複排除・デッドコード削除)

**実装内容:**
1. `FluidKit/utils.py` を新設し以下を集約:
   - `DataNormalizer`(正規化ロジック — 現在 `collect_data.py:143-149` と `infer_v2.py:44-45` に重複)
   - Poly6等のSPHカーネル関数(`gen_sample.py:20` と `sph_solver.py:90-93` に重複 — Phase 1完了後は自然解消の見込み)
   - `knn_graph`(`model.py:36` と `model_v2.py:33` に重複)
2. v1系デッドコードの削除: `neural/train.py`, `neural/infer_to_json.py`, `neural/model.py`(v2が採用版)。削除前に`git log`でv1参照が残っていないことを確認
3. `neural/` の分割: `model_v2.py`(231行: モデル+損失+utils混在)→ `model.py` / `loss.py` / `dataset.py`

**検証チェックリスト:**
- [ ] `grep -rn "import model\b\|from model import\|train\.py" FluidKit/` でv1参照が残っていないこと
- [ ] 学習・推論のスモークテスト(5エポック+1推論)が完走
- [ ] 正規化の統一後、推論結果が変更前と一致

**アンチパターン:** リネームと挙動変更を同一コミットに混ぜない(bisect不能になる)。

---

## Phase 6: テストスイート追加

**実装内容:** `FluidKit/tests/` を新設:
- `test_solvers.py`: 運動量保存・質量保存・境界貫通なしの物性テスト(3ソルバー)
- `test_reproducibility.py`: 同一シード・同一設定で結果ハッシュが一致
- `test_io.py`: JSON/NPZ書き出し→読み戻しのラウンドトリップ

**検証チェックリスト:**
- [ ] `pytest FluidKit/tests/ -v` 全パス
- [ ] CI(release.yml)にテストステップを追加

---

## Final Phase: 統合検証

- [ ] 全プリセットのデータ生成 → viewer再生 → 学習5エポック → 推論、のエンドツーエンドが完走
- [ ] 性能前後比較表を `FluidKit/docs/PERF_RESULTS.md` に記録(gen_sample生成時間、forward時間、GPUメモリ、ファイルサイズ)
- [ ] `pytest` 全パス
- [ ] README.mdの性能表を実測値で更新

**優先順位まとめ: Phase 1(2h・100倍) → Phase 2(4h・5倍+メモリ10分の1) → Phase 4(3h・I/O半減) → Phase 5/6(保守性) → Phase 3(8h・要数値検証)**
