"""
test_curl_noise.py — カールノイズ (solver/curl_noise.py) と比較 Atlas の指標定義のテスト。

検証の考え方:
  - カールノイズは解析的に発散ゼロ。有限差分で測った発散は離散化誤差 O(h²) だけのはずで、
    速度勾配の大きさに対して十分小さいことを数値で確認する (改善書 N1 の検証項目)。
  - 比較 Atlas の指標は、全手法で同じ定義 (重心・比運動エネルギー等) を使うこと (N4 の検証項目)。
"""

from __future__ import annotations

import numpy as np
import pytest

from curl_noise import CurlNoise2D, CurlNoise3D, divergence_residual


class TestCurlNoiseDivergence:
    @pytest.mark.parametrize("cls,lo,hi", [
        (CurlNoise2D, [0.0, 0.0], [3.0, 3.0]),
        (CurlNoise3D, [0.0, 0.0, 0.0], [3.0, 3.0, 3.0]),
    ])
    def test_divergence_is_small_relative_to_gradient(self, cls, lo, hi):
        div, grad = divergence_residual(cls(seed=3), lo, hi, h=1e-3)
        assert div / grad < 1e-4

    def test_divergence_converges_as_h_squared(self):
        """h を 1/10 にすると残差は約 1/100 (= 解析的にゼロで、差は離散化誤差のみ)。"""
        field = CurlNoise2D(seed=1)
        d_coarse, _ = divergence_residual(field, [0, 0], [3, 3], h=1e-2)
        d_fine, _ = divergence_residual(field, [0, 0], [3, 3], h=1e-3)
        assert d_fine < d_coarse / 50

    def test_time_dependent_field_stays_divergence_free(self):
        div, grad = divergence_residual(CurlNoise3D(seed=2), [0] * 3, [3] * 3, h=1e-3, t=2.5)
        assert div / grad < 1e-4


class TestCurlNoiseField:
    def test_deterministic_for_same_seed(self):
        p = np.random.RandomState(0).uniform(0, 3, size=(50, 3))
        a = CurlNoise3D(seed=7).velocity(p, 0.3)
        b = CurlNoise3D(seed=7).velocity(p, 0.3)
        assert np.array_equal(a, b)

    def test_different_seed_gives_different_field(self):
        p = np.random.RandomState(0).uniform(0, 3, size=(50, 2))
        assert not np.allclose(CurlNoise2D(seed=1).velocity(p), CurlNoise2D(seed=2).velocity(p))

    def test_velocity_is_finite_and_bounded(self):
        p = np.random.RandomState(1).uniform(-5, 5, size=(2000, 3))
        v = CurlNoise3D(seed=4, strength=1.0).velocity(p, 1.0)
        assert np.isfinite(v).all()
        assert np.linalg.norm(v, axis=1).max() < 10.0     # strength=1 で速さのオーダーは ~1


class TestCompareSolversMetrics:
    """比較 Atlas: 全手法が同じ指標 (同じ定義) を返し、初期条件が揃っていること。"""

    @pytest.fixture(scope="class")
    def runs(self):
        import compare_solvers as cs
        out = {}
        for cls in (cs.SPHRunner, cs.FLIPRunner, cs.GridRunner):
            runner = cls(60)
            times, rows, _, _ = cs.run_scenario(runner, duration=0.1, sample_dt=0.1,
                                                export_frames=False)
            out[runner.method] = (cs, runner, rows[-1])
        return out

    def test_all_methods_report_same_metric_keys(self, runs):
        keys = {frozenset(m) for _, _, m in runs.values()}
        assert keys == {frozenset({"mass_ratio", "ke", "py", "com_y"})}

    def test_metrics_are_finite(self, runs):
        for _, _, m in runs.values():
            assert all(np.isfinite(v) for v in m.values())

    def test_particle_methods_conserve_mass_by_construction(self, runs):
        assert runs["SPH"][2]["mass_ratio"] == 1.0
        assert runs["FLIP"][2]["mass_ratio"] == 1.0

    def test_initial_center_of_mass_matches_across_methods(self, runs):
        """同じ初期条件: 0.1 秒後の重心高さは、自由落下の理論値から大きく外れない。"""
        cs = runs["SPH"][0]
        theory = cs.free_fall_y(0.1)
        for method, (_, _, m) in runs.items():
            assert abs(m["com_y"] - theory) < 0.05, method

    def test_falling_blob_gains_downward_momentum(self, runs):
        for method, (_, _, m) in runs.items():
            assert m["py"] < 0.0, method
