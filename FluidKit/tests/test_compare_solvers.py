"""
test_compare_solvers.py — 比較 Atlas (tools/compare_solvers.py) の指標定義のテスト。

全手法 (SPH / FLIP / 格子法) が同じ定義の指標を返し、同じ初期条件から始まっていること (改善書 N4 の検証項目)。
"""

from __future__ import annotations

import numpy as np
import pytest


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
