"""
test_solvers.py — 3 ソルバー (StableFluids2D / SPHSolver / FLIPSolver) の
物性テスト: 運動量保存・質量保存・境界貫通なし。

各テストは小規模 (N=20〜50 粒子, 10〜50 グリッド, 10〜50 ステップ) の
実ソルバー実行で検証する（モックなし）。
"""

from __future__ import annotations

import numpy as np
import pytest

from solver import StableFluids2D, SPHSolver, Particle, FLIPSolver


# ──────────────────────────────────────────────────────────────
#  SPHSolver
# ──────────────────────────────────────────────────────────────

class TestSPHSolverMomentum:
    """
    SPHSolver の内部力 (圧力・粘性) は作用・反作用の対で
    r_vecs[i,j] = -r_vecs[j,i] となるよう構成されているため、
    重力を除いた総運動量 sum(mass*vel) は理論上厳密に保存される
    (数値誤差は浮動小数点丸め程度 = 相対誤差 ~1e-10 のオーダー)。

    重力ありの場合は、既知の外力インパルス g*dt*n_steps*total_mass だけ
    運動量が変化するはず、という枠組みで検証する。
    """

    def _make_particles(self, n: int, seed: int) -> list[Particle]:
        rng = np.random.RandomState(seed)
        positions = rng.uniform(-0.3, 0.3, size=(n, 2))
        return [
            Particle(pos=positions[i].copy(), vel=np.zeros(2), mass=1.0)
            for i in range(n)
        ]

    def test_momentum_conserved_without_gravity(self):
        """重力ゼロなら運動量は厳密に保存される（内力は作用反作用対）。"""
        n_particles = 30
        particles = self._make_particles(n_particles, seed=0)
        solver = SPHSolver(
            smoothing_length=0.15,
            rest_density=1000.0,
            gas_constant=80.0,
            viscosity=0.01,
            gravity=np.array([0.0, 0.0]),
        )

        def total_momentum(ps: list[Particle]) -> np.ndarray:
            return sum((p.mass * p.vel for p in ps), start=np.zeros(2))

        p0 = total_momentum(particles)
        for _ in range(30):
            solver.step(particles, dt=0.005)
        p1 = total_momentum(particles)

        # 内力のみなので機械精度に近い一致を期待
        assert np.allclose(p1, p0, atol=1e-8)

    def test_momentum_matches_gravity_impulse(self):
        """重力ありの場合、運動量変化は g*dt*n_steps*total_mass に一致する。"""
        n_particles = 30
        particles = self._make_particles(n_particles, seed=1)
        g = np.array([0.0, -9.8])
        solver = SPHSolver(
            smoothing_length=0.15,
            rest_density=1000.0,
            gas_constant=80.0,
            viscosity=0.01,
            gravity=g,
        )

        def total_momentum(ps: list[Particle]) -> np.ndarray:
            return sum((p.mass * p.vel for p in ps), start=np.zeros(2))

        total_mass = sum(p.mass for p in particles)
        p0 = total_momentum(particles)
        dt = 0.005
        n_steps = 30
        for _ in range(n_steps):
            solver.step(particles, dt=dt)
        p1 = total_momentum(particles)

        expected_impulse = g * dt * n_steps * total_mass
        actual_change = p1 - p0
        # 内力は対で相殺するので理論上ほぼ厳密一致 (相対誤差 ~1e-2 を許容枠として設定)
        rel_err = np.linalg.norm(actual_change - expected_impulse) / np.linalg.norm(expected_impulse)
        assert rel_err < 1e-2

    def test_particle_count_conserved(self):
        """SPHSolver は粒子を追加/削除しない。"""
        particles = self._make_particles(25, seed=2)
        solver = SPHSolver(smoothing_length=0.15)
        n0 = len(particles)
        for _ in range(20):
            solver.step(particles, dt=0.005)
        assert len(particles) == n0

    def test_no_nan_or_runaway(self):
        """短時間シミュレーションで発散・NaN が出ないこと（境界貫通なしの前提条件）。"""
        particles = self._make_particles(40, seed=3)
        g = np.array([0.0, -9.8])
        solver = SPHSolver(
            smoothing_length=0.12,
            rest_density=1000.0,
            gas_constant=80.0,
            viscosity=0.01,
            gravity=g,
        )
        for _ in range(50):
            solver.step(particles, dt=0.005)
        positions = np.stack([p.pos for p in particles])
        velocities = np.stack([p.vel for p in particles])
        assert np.isfinite(positions).all()
        assert np.isfinite(velocities).all()


class TestSPHSolverBoundary:
    """
    SPHSolver 自体は境界を扱わないため (gen_sample.py 側で反射処理を行う)、
    ここでは gen_sample.py と同じ反射ロジックを適用したうえで
    N ステップ後に全粒子が指定境界内に収まることを検証する。
    """

    def test_no_boundary_penetration_with_reflection(self):
        rng = np.random.RandomState(4)
        n_particles = 30
        bounds_lo = np.array([-1.0, 0.0])
        bounds_hi = np.array([1.0, 2.0])

        positions = rng.uniform(-0.3, 0.3, size=(n_particles, 2)) + np.array([0.0, 1.0])
        particles = [
            Particle(pos=positions[i].copy(), vel=np.zeros(2), mass=1.0)
            for i in range(n_particles)
        ]
        solver = SPHSolver(
            smoothing_length=0.15,
            rest_density=1000.0,
            gas_constant=80.0,
            viscosity=0.01,
            gravity=np.array([0.0, -9.8]),
        )

        restitution = 0.3
        for _ in range(50):
            solver.step(particles, dt=0.005)
            # gen_sample.py の境界反射ロジックと同一
            for p in particles:
                for d in range(2):
                    if p.pos[d] < bounds_lo[d]:
                        p.pos[d] = bounds_lo[d]
                        p.vel[d] *= -restitution
                    elif p.pos[d] > bounds_hi[d]:
                        p.pos[d] = bounds_hi[d]
                        p.vel[d] *= -restitution

        for p in particles:
            assert (p.pos >= bounds_lo - 1e-9).all()
            assert (p.pos <= bounds_hi + 1e-9).all()


# ──────────────────────────────────────────────────────────────
#  StableFluids2D
# ──────────────────────────────────────────────────────────────

class TestStableFluids2D:
    """
    グリッドベースソルバーのため「質量保存」は総密度、「境界貫通なし」は
    壁面法線速度がゼロになる no-slip 条件で検証する。
    """

    def test_no_boundary_penetration(self):
        """
        no-slip 境界条件下では、壁面での法線速度 (ゴーストセルと内部セルの平均)
        は常にゼロになるはず (StableFluids2D._set_bnd の b=1/2 分岐参照)。
        """
        N = 16
        sf = StableFluids2D(N=N, dt=0.1, visc=0.0, diff=0.0)
        sf.add_density(N // 2, N // 2, 50.0)
        sf.add_velocity(N // 2, N // 2, 3.0, 2.0)
        sf.add_velocity(N // 2 + 2, N // 2 - 1, -2.0, 1.5)

        for _ in range(15):
            sf.step()

        # x 方向の壁 (i=0, i=N+1) での法線速度 u
        wall_u_lo = 0.5 * (sf.u[0, 1:N + 1] + sf.u[1, 1:N + 1])
        wall_u_hi = 0.5 * (sf.u[N + 1, 1:N + 1] + sf.u[N, 1:N + 1])
        # y 方向の壁 (j=0, j=N+1) での法線速度 v
        wall_v_lo = 0.5 * (sf.v[1:N + 1, 0] + sf.v[1:N + 1, 1])
        wall_v_hi = 0.5 * (sf.v[1:N + 1, N + 1] + sf.v[1:N + 1, N])

        assert np.allclose(wall_u_lo, 0.0, atol=1e-10)
        assert np.allclose(wall_u_hi, 0.0, atol=1e-10)
        assert np.allclose(wall_v_lo, 0.0, atol=1e-10)
        assert np.allclose(wall_v_hi, 0.0, atol=1e-10)

    def test_density_stays_finite_and_bounded(self):
        """
        密度は移流・拡散で減衰しうる（数値粘性・境界クランプにより
        総和は保存されない）が、NaN/発散なく非負に保たれることを確認する。
        """
        N = 16
        sf = StableFluids2D(N=N, dt=0.1, visc=0.0001, diff=0.0001)
        sf.add_density(N // 2, N // 2, 100.0)
        sf.add_velocity(N // 2, N // 2, 2.0, 1.0)

        initial_total = sf.dens.sum()
        for _ in range(20):
            sf.step()

        assert np.isfinite(sf.dens).all()
        assert (sf.dens >= -1e-8).all()
        # 総量は増えない（発生源なしなので保存 or 減衰のみ）— 緩い上限チェック
        assert sf.dens.sum() <= initial_total + 1e-6

    def test_no_nan_after_many_steps(self):
        """N=20 グリッドで50ステップ実行しても発散しないこと。"""
        N = 20
        sf = StableFluids2D(N=N, dt=0.05, visc=0.001, diff=0.001)
        sf.add_density(N // 2, N // 2, 30.0)
        sf.add_velocity(N // 2, N // 2, 1.0, 1.0)
        for _ in range(50):
            sf.step()
        assert np.isfinite(sf.u).all()
        assert np.isfinite(sf.v).all()
        assert np.isfinite(sf.dens).all()


# ──────────────────────────────────────────────────────────────
#  FLIPSolver
# ──────────────────────────────────────────────────────────────

class TestFLIPSolver:
    """
    FLIP は grid P2G/G2P (線形補間の散布・収集) を経由するため、
    SPH と異なり運動量は厳密には保存されない（補間による数値粘性がある）。
    そのため「重力による期待インパルスに対して同符号・同オーダーで変化する」
    ことをゆるい許容誤差 (相対誤差 ~5e-1) で確認するに留める。
    厳密な物理不変量としては粒子数保存と境界内クランプを検証する。
    """

    def test_particle_count_conserved(self):
        rng = np.random.RandomState(5)
        Nx, Ny = 16, 16
        flip = FLIPSolver(Nx=Nx, Ny=Ny, dx=1.0, dt=0.02, alpha=0.05)
        n_particles = 25
        pos = rng.uniform(2.0, Nx - 2.0, size=(n_particles, 2))
        vel = np.zeros((n_particles, 2))
        for _ in range(20):
            pos, vel = flip.step(pos, vel, gravity=np.array([0.0, -9.8]))
        assert pos.shape[0] == n_particles
        assert vel.shape[0] == n_particles

    def test_no_boundary_penetration(self):
        """_clamp_positions によりグリッド範囲 [0, Nx*dx] x [0, Ny*dx] に収まる。"""
        rng = np.random.RandomState(6)
        Nx, Ny = 16, 16
        dx = 1.0
        flip = FLIPSolver(Nx=Nx, Ny=Ny, dx=dx, dt=0.02, alpha=0.05)
        n_particles = 25
        pos = rng.uniform(2.0, Nx * dx - 2.0, size=(n_particles, 2))
        vel = np.zeros((n_particles, 2))
        for _ in range(30):
            pos, vel = flip.step(pos, vel, gravity=np.array([0.0, -9.8]))

        assert (pos[:, 0] >= 0.0 - 1e-9).all()
        assert (pos[:, 0] <= Nx * dx + 1e-9).all()
        assert (pos[:, 1] >= 0.0 - 1e-9).all()
        assert (pos[:, 1] <= Ny * dx + 1e-9).all()

    def test_momentum_same_sign_as_gravity_impulse(self):
        """
        数値粘性で厳密一致はしないが、鉛直方向の運動量変化は重力方向と同符号で、
        期待インパルスのオーダー（同じ桁数、過大でない）に収まるはず。
        """
        rng = np.random.RandomState(7)
        Nx, Ny = 16, 16
        dx = 1.0
        flip = FLIPSolver(Nx=Nx, Ny=Ny, dx=dx, dt=0.02, alpha=0.05)
        n_particles = 25
        pos = rng.uniform(2.0, Nx * dx - 2.0, size=(n_particles, 2))
        vel = np.zeros((n_particles, 2))
        g = np.array([0.0, -9.8])
        dt = 0.02
        n_steps = 20

        p0 = vel.sum(axis=0)
        for _ in range(n_steps):
            pos, vel = flip.step(pos, vel, gravity=g)
        p1 = vel.sum(axis=0)

        dv_y = p1[1] - p0[1]
        expected_y = g[1] * dt * n_steps * n_particles
        # 同符号（下向きに加速している）
        assert dv_y < 0
        # 数値粘性・境界クランプで expected の何倍にも過大にはならない
        assert abs(dv_y) <= abs(expected_y) * 1.5
