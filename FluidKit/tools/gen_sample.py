"""
gen_sample.py  —  FluidKit サンプルデータ生成
pyjet なしで動くシンプルな SPH 風シミュレーションを実行し、
Three.js Viewer 用の frames.json を生成します。

使い方:
    python gen_sample.py                        # デフォルト（water_drop）
    python gen_sample.py --preset smoke         # 煙
    python gen_sample.py --preset splash        # 水しぶき
    python gen_sample.py --frames 150 --particles 2000
"""

import json, math, random, argparse, sys
from pathlib import Path

import numpy as np

# solver/ の SPHSolver (空間ハッシュ + numpy ベクトル化) を流用
sys.path.insert(0, str(Path(__file__).parent.parent / "solver"))
from sph_solver import SPHSolver, Particle

# ──────────────────────────────────────────
#  tiny SPH helpers
# ──────────────────────────────────────────

def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


# ──────────────────────────────────────────
#  Preset definitions
# ──────────────────────────────────────────

PRESETS = {
    "water_drop": dict(
        description="球形の水滴が落下して床に衝突",
        gravity=(0, -9.8, 0),
        particle_count=1500,
        h=0.12,
        stiffness=80.0,
        rest_density=1000.0,
        viscosity=0.01,
        bounds=(-1.5, 1.5, 0.0, 3.0, -1.5, 1.5),  # xmin xmax ymin ymax zmin zmax
        init="drop",
    ),
    "smoke": dict(
        description="下から上昇する煙（浮力付き）",
        gravity=(0, 0.5, 0),      # 正方向（上昇）
        particle_count=1200,
        h=0.15,
        stiffness=20.0,
        rest_density=1.2,
        viscosity=0.05,
        bounds=(-1.5, 1.5, 0.0, 4.0, -1.5, 1.5),
        init="smoke_source",
    ),
    "splash": dict(
        description="左右からの流れが中央で衝突",
        gravity=(0, -9.8, 0),
        particle_count=2000,
        h=0.10,
        stiffness=100.0,
        rest_density=1000.0,
        viscosity=0.005,
        bounds=(-2.0, 2.0, 0.0, 3.0, -1.0, 1.0),
        init="two_streams",
    ),
}


# ──────────────────────────────────────────
#  Initializers
# ──────────────────────────────────────────

def _init_drop(n, bounds):
    """球形の水滴を空中に配置."""
    cx, cy, cz = 0.0, 2.2, 0.0
    r0 = 0.55
    positions, velocities = [], []
    while len(positions) < n:
        x = random.uniform(-r0, r0)
        y = random.uniform(-r0, r0)
        z = random.uniform(-r0, r0)
        if x*x + y*y + z*z <= r0*r0:
            positions.append([cx+x, cy+y, cz+z])
            velocities.append([random.gauss(0, 0.05),
                               random.gauss(0, 0.05),
                               random.gauss(0, 0.05)])
    return positions, velocities


def _init_smoke_source(n, bounds):
    """床付近に広く分散."""
    positions, velocities = [], []
    xlo, xhi, ylo, _, zlo, zhi = bounds
    for _ in range(n):
        positions.append([
            random.uniform(xlo*0.6, xhi*0.6),
            random.uniform(ylo, ylo + 0.4),
            random.uniform(zlo*0.6, zhi*0.6),
        ])
        velocities.append([random.gauss(0, 0.1),
                           random.uniform(0.3, 0.8),
                           random.gauss(0, 0.1)])
    return positions, velocities


def _init_two_streams(n, bounds):
    """左右から流れ込む粒子群."""
    positions, velocities = [], []
    half = n // 2
    xlo, xhi, ylo, _, zlo, zhi = bounds
    yw = 0.5
    for i in range(n):
        side = -1 if i < half else 1
        x = side * random.uniform(xhi * 0.5, xhi * 0.9)
        y = random.uniform(ylo + 0.1, ylo + yw)
        z = random.uniform(zlo * 0.5, zhi * 0.5)
        vx = -side * random.uniform(1.5, 2.5)
        positions.append([x, y, z])
        velocities.append([vx, random.gauss(0, 0.05), random.gauss(0, 0.05)])
    return positions, velocities


INITIALIZERS = {
    "drop": _init_drop,
    "smoke_source": _init_smoke_source,
    "two_streams": _init_two_streams,
}


# ──────────────────────────────────────────
#  Simple SPH step
# ──────────────────────────────────────────

class SimpleSPH:
    """
    Preset 駆動の初期配置・境界（壁反射）ロジックはそのまま保持しつつ、
    密度・圧力・粘性・積分の物理計算は solver/sph_solver.py の
    SPHSolver（空間ハッシュ + numpy ベクトル化）に委譲するラッパー。

    公開 API (pos, vel, step, snapshot, velocities_flat) は
    neural/collect_data.py, neural/infer_to_json.py, neural/infer_v2.py が
    直接依存しているため変更しない。
    """

    def __init__(self, preset: dict, seed: int = 42):
        random.seed(seed)
        cfg = preset
        self.n = cfg["particle_count"]
        self.h = cfg["h"]
        self.k = cfg["stiffness"]
        self.rho0 = cfg["rest_density"]
        self.mu = cfg["viscosity"]
        self.g = list(cfg["gravity"])
        self.bounds = cfg["bounds"]   # xmin xmax ymin ymax zmin zmax

        init_fn = INITIALIZERS[cfg["init"]]
        pos, vel = init_fn(self.n, self.bounds)

        # SPHSolver 用の Particle リストを構築（3D pos/vel）
        self._particles = [
            Particle(pos=np.array(p, dtype=np.float64),
                      vel=np.array(v, dtype=np.float64))
            for p, v in zip(pos, vel)
        ]

        self._solver = SPHSolver(
            smoothing_length=self.h,
            rest_density=self.rho0,
            gas_constant=self.k,
            viscosity=self.mu,
            gravity=np.array(self.g, dtype=np.float64),
        )

        # pos / vel は従来通り外部から参照可能な形（(N,3) ndarray, リストとしてスライス可）で保持
        self._sync_from_particles()

    # ── internal: Particle リスト ⇔ pos/vel ndarray 同期 ──
    def _sync_from_particles(self):
        self.pos = np.stack([p.pos for p in self._particles])
        self.vel = np.stack([p.vel for p in self._particles])

    # ── one timestep ─────────────────────
    def step(self, dt: float = 0.016):
        # 密度・圧力・粘性・積分（Symplectic Euler）は SPHSolver に委譲
        self._solver.step(self._particles, dt=dt)

        # 境界反射（SPHSolver 自体は境界を扱わないので gen_sample 側で適用）
        xlo, xhi, ylo, yhi, zlo, zhi = self.bounds
        restitution = 0.3
        lo = np.array([xlo, ylo, zlo])
        hi = np.array([xhi, yhi, zhi])
        for p in self._particles:
            for d in range(3):
                if p.pos[d] < lo[d]:
                    p.pos[d] = lo[d]
                    p.vel[d] *= -restitution
                elif p.pos[d] > hi[d]:
                    p.pos[d] = hi[d]
                    p.vel[d] *= -restitution

        self._sync_from_particles()

    def snapshot(self):
        """現在の positions を flat list で返す."""
        flat = []
        for p in self.pos:
            flat.extend([round(float(p[0]), 4), round(float(p[1]), 4), round(float(p[2]), 4)])
        return flat

    def velocities_flat(self):
        flat = []
        for v in self.vel:
            spd = math.sqrt(float(v[0])**2 + float(v[1])**2 + float(v[2])**2)
            flat.append(round(spd, 4))
        return flat


# ──────────────────────────────────────────
#  Main
# ──────────────────────────────────────────

def run(preset_name: str, frames: int, dt: float, output: Path, seed: int):
    preset = PRESETS[preset_name]
    print(f"[FluidKit] preset={preset_name}  particles={preset['particle_count']}  frames={frames}")
    print(f"           {preset['description']}")

    sph = SimpleSPH(preset, seed=seed)

    frame_data = []
    speed_data = []
    for f in range(frames):
        if f % 10 == 0:
            pct = f / frames * 100
            print(f"  simulating... {pct:5.1f}%  frame {f}/{frames}", end="\r")
        frame_data.append(sph.snapshot())
        speed_data.append(sph.velocities_flat())
        sph.step(dt)

    print(f"  simulating... 100.0%  frame {frames}/{frames}  done     ")

    # メタデータ
    bounds = preset["bounds"]
    meta = {
        "preset": preset_name,
        "description": preset["description"],
        "frameCount": frames,
        "particleCount": preset["particle_count"],
        "fps": round(1.0 / dt),
        "dt": dt,
        "bounds": {
            "min": [bounds[0], bounds[2], bounds[4]],
            "max": [bounds[1], bounds[3], bounds[5]],
        },
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump({"metadata": meta, "frames": frame_data, "speeds": speed_data}, f,
                  separators=(",", ":"))

    size_kb = output.stat().st_size / 1024
    print(f"[FluidKit] saved → {output}  ({size_kb:.1f} KB)")


def main():
    ap = argparse.ArgumentParser(description="FluidKit サンプルデータ生成")
    ap.add_argument("--preset",    default="water_drop", choices=list(PRESETS.keys()))
    ap.add_argument("--frames",    type=int,   default=120)
    ap.add_argument("--dt",        type=float, default=0.016)
    ap.add_argument("--seed",      type=int,   default=42)
    ap.add_argument("--output",    default=None)
    args = ap.parse_args()

    out = Path(args.output) if args.output else \
          Path(__file__).parent.parent / "viewer" / "data" / f"{args.preset}.json"

    run(args.preset, args.frames, args.dt, out, args.seed)


if __name__ == "__main__":
    main()
