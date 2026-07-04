"""
test_reproducibility.py — SimulationRecord (solver/reproducibility.py) の
再現性検証: 同一シード・同一設定 → 同一ハッシュ、異なるシード → 異なるハッシュ。

実ソルバー (SPHSolver) を小規模・短時間で実際に走らせ、その状態列を
SimulationRecord に記録してハッシュを比較する。
"""

from __future__ import annotations

import numpy as np
import pytest

from solver import SimulationRecord, SPHSolver, Particle


def _run_tiny_sph_simulation(seed: int, n_steps: int = 10) -> SimulationRecord:
    """
    seed から初期粒子配置を決定的に生成し、SPHSolver で n_steps 進めながら
    各ステップの位置スナップショットを SimulationRecord に記録する。
    """
    config = {
        "solver": "SPHSolver",
        "n_particles": 20,
        "h": 0.15,
        "dt": 0.005,
        "seed": seed,
    }
    record = SimulationRecord(config, seed=seed)

    # np.random.seed は SimulationRecord.__init__ 内で既に呼ばれているため、
    # ここでの粒子初期配置生成は再現可能になる。
    positions = np.random.uniform(-0.3, 0.3, size=(config["n_particles"], 2))
    particles = [
        Particle(pos=positions[i].copy(), vel=np.zeros(2), mass=1.0)
        for i in range(config["n_particles"])
    ]
    solver = SPHSolver(
        smoothing_length=config["h"],
        rest_density=1000.0,
        gas_constant=80.0,
        viscosity=0.01,
        gravity=np.array([0.0, -9.8]),
    )

    for step in range(n_steps):
        solver.step(particles, dt=config["dt"])
        state = np.stack([p.pos for p in particles])
        record.log_step(step, state)

    return record


class TestReproducibility:
    def test_same_seed_same_config_identical_hash(self):
        """同一シード・同一設定で2回実行すると、全ステップのハッシュが一致する。"""
        r1 = _run_tiny_sph_simulation(seed=42)
        r2 = _run_tiny_sph_simulation(seed=42)

        assert r1.verify(r2)
        assert r1.record["config_hash"] == r2.record["config_hash"]
        assert len(r1._steps) == len(r2._steps) == 10
        for s1, s2 in zip(r1._steps, r2._steps):
            assert s1["state_hash"] == s2["state_hash"]

    def test_different_seed_different_hash(self):
        """
        異なるシードでは初期配置が変わり、結果として状態ハッシュも変わる
        ことを確認する（ハッシュが定数化していないかのサニティチェック）。
        """
        r1 = _run_tiny_sph_simulation(seed=42)
        r3 = _run_tiny_sph_simulation(seed=123)

        assert not r1.verify(r3)
        # config 自体 (seed含む) が異なるため config_hash も異なる
        assert r1.record["config_hash"] != r3.record["config_hash"]
        # 状態ハッシュも(初期配置が異なるため)異なるはず
        assert r1._steps[0]["state_hash"] != r3._steps[0]["state_hash"]

    def test_hash_is_deterministic_function_of_state(self):
        """同一の状態配列からは常に同一のハッシュが得られる（ハッシュ関数自体の健全性）。"""
        record = SimulationRecord({"dummy": True}, seed=1)
        state = np.array([[1.0, 2.0], [3.0, 4.0]])
        record.log_step(0, state)
        h1 = record._steps[0]["state_hash"]

        record2 = SimulationRecord({"dummy": True}, seed=1)
        record2.log_step(0, state.copy())
        h2 = record2._steps[0]["state_hash"]

        assert h1 == h2

    def test_hash_changes_with_state(self):
        """状態が変われば state_hash も変わる。"""
        record = SimulationRecord({"dummy": True}, seed=1)
        record.log_step(0, np.array([[1.0, 2.0]]))
        record.log_step(1, np.array([[1.0, 2.0000001]]))

        assert record._steps[0]["state_hash"] != record._steps[1]["state_hash"]
