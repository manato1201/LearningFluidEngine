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

# ──────────────────────────────────────────
#  tiny SPH helpers
# ──────────────────────────────────────────

def _kernel(r, h):
    """Poly6 kernel (smoothed weight)."""
    if r >= h:
        return 0.0
    q = 1.0 - (r / h) ** 2
    return q * q * q

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
        self.pos, self.vel = init_fn(self.n, self.bounds)
        self.pos = [list(p) for p in self.pos]
        self.vel = [list(v) for v in self.vel]

    # ── density ──────────────────────────
    def _density(self, i):
        pi = self.pos[i]
        rho = 0.0
        for j in range(self.n):
            pj = self.pos[j]
            dx = pi[0]-pj[0]; dy = pi[1]-pj[1]; dz = pi[2]-pj[2]
            r = math.sqrt(dx*dx + dy*dy + dz*dz)
            rho += _kernel(r, self.h)
        return max(rho, 1e-6)

    # ── one timestep ─────────────────────
    def step(self, dt: float = 0.016):
        h, k, rho0, mu = self.h, self.k, self.rho0, self.mu
        n = self.n
        pos, vel = self.pos, self.vel

        # 密度・圧力
        densities = [self._density(i) for i in range(n)]
        pressures = [k * (d - rho0) for d in densities]

        # 加速度
        acc = [[gx, gy, gz] for gx, gy, gz in [self.g] * n]

        for i in range(n):
            pi = pos[i]
            for j in range(n):
                if i == j:
                    continue
                pj = pos[j]
                dx = pi[0]-pj[0]; dy = pi[1]-pj[1]; dz = pi[2]-pj[2]
                r = math.sqrt(dx*dx + dy*dy + dz*dz) + 1e-8
                if r >= h:
                    continue
                # pressure gradient (symmetric)
                w = _kernel(r, h)
                pf = -(pressures[i] + pressures[j]) / (2.0 * densities[j]) * w / r
                acc[i][0] += pf * dx
                acc[i][1] += pf * dy
                acc[i][2] += pf * dz
                # viscosity
                dvx = vel[j][0]-vel[i][0]
                dvy = vel[j][1]-vel[i][1]
                dvz = vel[j][2]-vel[i][2]
                vf = mu * w / densities[j]
                acc[i][0] += vf * dvx
                acc[i][1] += vf * dvy
                acc[i][2] += vf * dvz

        # 積分
        xlo, xhi, ylo, yhi, zlo, zhi = self.bounds
        restitution = 0.3
        for i in range(n):
            vel[i][0] += acc[i][0] * dt
            vel[i][1] += acc[i][1] * dt
            vel[i][2] += acc[i][2] * dt
            pos[i][0] += vel[i][0] * dt
            pos[i][1] += vel[i][1] * dt
            pos[i][2] += vel[i][2] * dt
            # 境界反射
            if pos[i][0] < xlo: pos[i][0] = xlo; vel[i][0] *= -restitution
            if pos[i][0] > xhi: pos[i][0] = xhi; vel[i][0] *= -restitution
            if pos[i][1] < ylo: pos[i][1] = ylo; vel[i][1] *= -restitution
            if pos[i][1] > yhi: pos[i][1] = yhi; vel[i][1] *= -restitution
            if pos[i][2] < zlo: pos[i][2] = zlo; vel[i][2] *= -restitution
            if pos[i][2] > zhi: pos[i][2] = zhi; vel[i][2] *= -restitution

    def snapshot(self):
        """現在の positions を flat list で返す."""
        flat = []
        for p in self.pos:
            flat.extend([round(p[0], 4), round(p[1], 4), round(p[2], 4)])
        return flat

    def velocities_flat(self):
        flat = []
        for v in self.vel:
            spd = math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
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
