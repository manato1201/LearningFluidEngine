"""
compare_solvers.py  —  比較 Atlas: 同じ初期条件を SPH / FLIP / 格子法(Stam) に与えて並べる

シナリオ: 単位正方形の箱 [0,1]² の上部に円形の水塊 (中心 (0.5,0.75), 半径 0.12) を置き、
重力 -9.8 で落とす。初期速度ゼロ。各手法は自分の安定な時間刻みでサブステップし、
**同じシミュレーション時刻**で指標を比べる。

指標 (すべての手法で同じ定義。質量は初期の総質量 M0 で正規化するため単位系に依存しない):
  - 計算時間      : ms / step と ms / シミュレーション秒 (後者が手法間で比較可能なコスト)
  - 質量比        : M(t)/M0   粒子法は粒子数一定で 1.0、格子法は Σ ρ·dx² の変化
  - 比運動エネルギー: KE/M0 = Σ ½ m|v|² / M0 [m²/s²]
  - 比運動量(y)   : P_y/M0 = Σ m v_y / M0 [m/s]
  - 重心の高さ    : Σ m y / Σ m [m] と、空気抵抗なしの自由落下の理論値 y0 - ½ g t²

GNN (Neural Fluid) は学習済みチェックポイントが要るため、この表には含めない
(neural/infer_v2.py で別途生成し、viewer に一緒にドロップして見比べる)。

注意: 格子法 (Stam) は「煙/気体向けの密度場」で、水塊の自由表面は表現しない。
空気側も非圧縮として解くため、水塊は自由落下せず遅く動く。これは手法の違いであり、バグではない。
SPH / FLIP も簡易実装で、この表は**手法の性格の比較**であり精度の優劣ではない。

使い方:
    python compare_solvers.py                          # 表を標準出力へ
    python compare_solvers.py --output ../docs/COMPARE_ATLAS.md
    python compare_solvers.py --export-viewer          # viewer/data/compare_*.json も出力
"""

from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np

_FLUIDKIT = Path(__file__).parent.parent
sys.path.insert(0, str(_FLUIDKIT))
sys.path.insert(0, str(_FLUIDKIT / "solver"))
from solver import StableFluids2D, SPHSolver, Particle, FLIPSolver   # noqa: E402
from solver.curl_noise import CurlNoise2D, divergence_residual     # noqa: E402

G = 9.8
CENTER = np.array([0.5, 0.75])
RADIUS = 0.12
EXPORT_SCALE = 3.0           # viewer の既存シーン (3 単位幅) に合わせる
EXPORT_FPS = 60


# ──────────────────────────────────────────────────────────────
#  初期条件
# ──────────────────────────────────────────────────────────────

def blob_points(n_target: int) -> tuple[np.ndarray, float]:
    """円内の正方格子点。(points[N,2], spacing) を返す。"""
    spacing = np.sqrt(np.pi * RADIUS ** 2 / n_target)
    k = int(np.ceil(RADIUS / spacing)) + 1
    xs = np.arange(-k, k + 1) * spacing
    gx, gy = np.meshgrid(xs, xs, indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
    pts = pts[np.linalg.norm(pts, axis=1) <= RADIUS]
    return pts + CENTER, float(spacing)


def free_fall_y(t: float) -> float:
    return float(CENTER[1] - 0.5 * G * t * t)


# ──────────────────────────────────────────────────────────────
#  各手法のランナー: 共通インターフェース
#    advance(dt_sim)  : dt_sim だけ進める (内部でサブステップ)
#    metrics()        : dict(mass_ratio, ke, py, com_y)
#    snapshot()       : (positions[M,2], speeds[M]) — viewer 出力用
# ──────────────────────────────────────────────────────────────

class SPHRunner:
    name, method = "SPH", "SPH"
    restitution = 0.3

    def __init__(self, n_target: int):
        pts, s = blob_points(n_target)
        self.n = len(pts)
        rho0 = 1000.0
        mass = rho0 * np.pi * RADIUS ** 2 / self.n
        self.particles = [Particle(pos=p.copy(), vel=np.zeros(2), mass=mass) for p in pts]
        self.mass = mass
        self.solver = SPHSolver(smoothing_length=3.2 * s, rest_density=rho0,
                                gas_constant=60.0, viscosity=0.5,
                                gravity=np.array([0.0, -G]))
        self.dt = 0.001
        self.elements = f"{self.n} 粒子"
        self.step_ms: list[float] = []

    def advance(self, dt_sim: float) -> None:
        for _ in range(int(round(dt_sim / self.dt))):
            t0 = time.perf_counter()
            self.solver.step(self.particles, dt=self.dt)
            for p in self.particles:                      # 箱の壁 (gen_sample と同じ反射)
                for d in range(2):
                    if p.pos[d] < 0.0:
                        p.pos[d] = 0.0
                        p.vel[d] *= -self.restitution
                    elif p.pos[d] > 1.0:
                        p.pos[d] = 1.0
                        p.vel[d] *= -self.restitution
            self.step_ms.append((time.perf_counter() - t0) * 1000)

    def _arrays(self):
        return (np.stack([p.pos for p in self.particles]),
                np.stack([p.vel for p in self.particles]))

    def metrics(self) -> dict:
        pos, vel = self._arrays()
        return dict(mass_ratio=1.0,
                    ke=float(0.5 * (vel ** 2).sum() / self.n),
                    py=float(vel[:, 1].mean()),
                    com_y=float(pos[:, 1].mean()))

    def snapshot(self):
        pos, vel = self._arrays()
        return pos, np.linalg.norm(vel, axis=1)


class FLIPRunner:
    name, method = "FLIP", "FLIP"

    def __init__(self, n_target: int, grid: int = 32):
        pts, _ = blob_points(n_target)
        self.n = len(pts)
        self.pos, self.vel = pts.copy(), np.zeros_like(pts)
        self.dt = 0.01
        self.solver = FLIPSolver(Nx=grid, Ny=grid, dx=1.0 / grid, dt=self.dt)
        self.elements = f"{self.n} 粒子 + {grid}×{grid} 格子"
        self.step_ms: list[float] = []

    def advance(self, dt_sim: float) -> None:
        for _ in range(int(round(dt_sim / self.dt))):
            t0 = time.perf_counter()
            self.pos, self.vel = self.solver.step(self.pos, self.vel,
                                                  gravity=np.array([0.0, -G]))
            self.step_ms.append((time.perf_counter() - t0) * 1000)

    def metrics(self) -> dict:
        return dict(mass_ratio=1.0,
                    ke=float(0.5 * (self.vel ** 2).sum() / self.n),
                    py=float(self.vel[:, 1].mean()),
                    com_y=float(self.pos[:, 1].mean()))

    def snapshot(self):
        return self.pos.copy(), np.linalg.norm(self.vel, axis=1)


class GridRunner:
    name, method = "格子法 (Stam)", "GRID"
    thresh = 0.3      # viewer 出力で「水のある格子」とみなす密度

    def __init__(self, n_target: int, grid: int = 64):
        self.N = grid
        self.dx = 1.0 / grid
        self.fluid = StableFluids2D(grid, dt=0.01)
        c = (np.arange(1, grid + 1) - 0.5) * self.dx          # セル中心
        cx, cy = np.meshgrid(c, c, indexing="ij")
        inside = (cx - CENTER[0]) ** 2 + (cy - CENTER[1]) ** 2 <= RADIUS ** 2
        self.fluid.dens[1:grid + 1, 1:grid + 1] = inside.astype(float)
        self._cx, self._cy = cx, cy
        self.m0 = float(self.fluid.dens.sum() * self.dx ** 2)
        self.dt = self.fluid.dt
        self.elements = f"{grid}×{grid} セル"
        self.step_ms: list[float] = []

    def _interior(self):
        s = slice(1, self.N + 1)
        f = self.fluid
        return f.dens[s, s], f.u[s, s], f.v[s, s]

    def advance(self, dt_sim: float) -> None:
        f = self.fluid
        for _ in range(int(round(dt_sim / self.dt))):
            t0 = time.perf_counter()
            dens, _, _ = self._interior()
            mask = dens > 0.05                                 # 水のある格子にだけ重力を足す
            s = slice(1, self.N + 1)
            f.v[s, s] -= G * self.dt * mask
            f.step()
            self.step_ms.append((time.perf_counter() - t0) * 1000)

    def metrics(self) -> dict:
        dens, u, v = self._interior()
        m = float(dens.sum() * self.dx ** 2)
        w = dens * self.dx ** 2
        return dict(mass_ratio=m / self.m0,
                    ke=float(0.5 * (w * (u ** 2 + v ** 2)).sum() / m),
                    py=float((w * v).sum() / m),
                    com_y=float((w * self._cy).sum() / w.sum()))

    def snapshot(self):
        dens, u, v = self._interior()
        sel = dens > self.thresh
        pos = np.stack([self._cx[sel], self._cy[sel]], axis=1)
        return pos, np.hypot(u[sel], v[sel])



# ──────────────────────────────────────────────────────────────
#  カールノイズ vs 格子法 (Stam): 「軽さ」と「物理らしさ」(改善書 N1)
# ──────────────────────────────────────────────────────────────

def stam_divergence(fluid: StableFluids2D) -> tuple[float, float]:
    """内部セルの中心差分の発散の最大値と、速度勾配の rms (= 「発散が小さい」の基準)。"""
    N, dx = fluid.N, 1.0 / fluid.N
    u, v = fluid.u, fluid.v
    dudx = (u[2:N + 2, 1:N + 1] - u[0:N, 1:N + 1]) / (2 * dx)
    dvdy = (v[1:N + 1, 2:N + 2] - v[1:N + 1, 0:N]) / (2 * dx)
    dudy = (u[1:N + 1, 2:N + 2] - u[1:N + 1, 0:N]) / (2 * dx)
    dvdx = (v[2:N + 2, 1:N + 1] - v[0:N, 1:N + 1]) / (2 * dx)
    rms = float(np.sqrt((dudx ** 2 + dvdy ** 2 + dudy ** 2 + dvdx ** 2).mean() / 4))
    return float(np.abs(dudx + dvdy).max()), rms


def compare_curl_vs_stam(grid: int = 64, steps: int = 30, n_points: int = 4096) -> dict:
    """同じ 2D 領域で、Stam の 1 ステップと、カールノイズ速度場での粒子移流 1 ステップを比べる。"""
    fluid = StableFluids2D(grid, dt=0.01)
    for i, j, du, dv in [(grid // 3, grid // 2, 8.0, 3.0), (2 * grid // 3, grid // 2, -8.0, -3.0)]:
        fluid.add_velocity(i, j, du, dv)
    times = []
    for _ in range(steps):
        t0 = time.perf_counter()
        fluid.step()
        times.append((time.perf_counter() - t0) * 1000)
    stam_div, stam_grad = stam_divergence(fluid)

    field = CurlNoise2D(seed=0, strength=1.0)
    pts = np.random.RandomState(0).uniform(0, 1, size=(n_points, 2))
    ctimes = []
    for k in range(steps):
        t0 = time.perf_counter()
        mid = pts + 0.005 * field.velocity(pts, k * 0.01)           # RK2 (中点法) 1 ステップ
        pts = np.clip(pts + 0.01 * field.velocity(mid, k * 0.01), 0, 1)
        ctimes.append((time.perf_counter() - t0) * 1000)
    c_div, c_grad = divergence_residual(field, [0, 0], [1, 1], h=1e-3)
    return dict(grid=grid, n_points=n_points,
                stam_ms=float(np.mean(times)), curl_ms=float(np.mean(ctimes)),
                stam_div=stam_div, stam_grad=stam_grad, curl_div=c_div, curl_grad=c_grad)


def curl_markdown(r: dict) -> list[str]:
    return [
        "",
        "## 4. カールノイズと格子法 (Stam) — 軽さと物理らしさ",
        "",
        "カールノイズは「ノイズの回転を速度場にする」だけの擬似流体 (圧力も保存則も解かない)。"
        f"同じ単位正方形で、Stam ({r['grid']}×{r['grid']} 格子, 圧力投影あり) と比べる。",
        "",
        "| 項目 | 格子法 (Stam) | カールノイズ |",
        "|---|---|---|",
        f"| 1 ステップの時間 | {r['stam_ms']:.2f} ms (格子全体。コストは格子数で決まる) | {r['curl_ms']:.2f} ms ({r['n_points']} 粒子の移流, RK2。コストは粒子数にほぼ比例) |",
        f"| 速度場の発散 (最大) / 速度勾配 rms | {r['stam_div']:.2e} / {r['stam_grad']:.2e} = {r['stam_div'] / r['stam_grad']:.2e} "
        f"| {r['curl_div']:.2e} / {r['curl_grad']:.2e} = {r['curl_div'] / r['curl_grad']:.2e} |",
        "| 発散ゼロの根拠 | 圧力投影を 20 回の反復で近似 (残差が残る) | 解析的に ∇·(∇×A) = 0 (差は離散化誤差 O(h²) のみ) |",
        "| 解いているもの | 近似的な非圧縮ナビエ–ストークス | なし (ノイズの回転) |",
        "| 壁・障害物 | あり (no-slip) | なし (粒子を反射させるのは別処理) |",
        "| 外力への応答 | あり (重力や衝撃に反応する) | なし (入力に反応しない) |",
        "",
        f"読み方: コストは Stam が格子数、カールノイズが粒子数 × 波の数で決まる。この条件では約 {r['stam_ms'] / (r['curl_ms'] / r['n_points']):.0f} 粒子"
        "より少なければカールノイズの方が軽く、多ければ Stam の方が軽い (**常にカールノイズが軽いわけではない**)。"
        "カールノイズは発散が小さい「流れっぽい動き」を作れるが、外力や壁に応答しないので**物理らしさは別物**。"
        "飾りの煙や水流に向き、衝突・落下の再現には SPH / FLIP / 格子法を使う。"
        "計測値は学習用の簡易実装・この環境での一例。",
        "",
    ]


# ──────────────────────────────────────────────────────────────
#  実行・出力
# ──────────────────────────────────────────────────────────────

def run_scenario(runner, duration: float, sample_dt: float, export_frames: bool):
    """sample_dt ごとに指標を記録し、export_frames なら 1/EXPORT_FPS ごとに粒子も記録。"""
    times, rows, frames, speeds = [], [], [], []
    frame_dt = 1.0 / EXPORT_FPS
    n_sub = int(round(duration / frame_dt))
    next_sample = sample_dt
    for k in range(n_sub + 1):
        t = k * frame_dt
        if export_frames:
            pos, spd = runner.snapshot()
            frames.append(pos)
            speeds.append(spd)
        if t + 1e-9 >= next_sample:
            times.append(t)
            rows.append(runner.metrics())
            next_sample += sample_dt
        if k < n_sub:
            runner.advance(frame_dt)
    return times, rows, frames, speeds


def export_viewer(runner, frames, speeds, out_dir: Path) -> Path:
    s = EXPORT_SCALE
    flat = [[round(float(v), 4) for pt in f for v in (pt[0] * s, pt[1] * s, 0.0)] for f in frames]
    spd = [[round(float(v), 4) for v in sp] for sp in speeds]
    meta = {
        "preset": f"compare_{runner.method.lower()}",
        "method": runner.method,
        "description": f"比較 Atlas: {runner.name}。円形の水塊の落下 (2D を z=0 に配置, 長さ×{s:g})",
        "frameCount": len(flat),
        "particleCount": max(len(f) // 3 for f in flat),
        "fps": EXPORT_FPS,
        "dt": 1.0 / EXPORT_FPS,
        "bounds": {"min": [0.0, 0.0, -0.15], "max": [s, s, 0.15]},
    }
    out = out_dir / f"compare_{runner.method.lower()}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"metadata": meta, "frames": flat, "speeds": spd}, f, separators=(",", ":"))
    return out


def build_markdown(results, sample_times, particles, duration, curl=None) -> str:
    final_t = sample_times[-1]
    lines = [
        "# 比較 Atlas — 同じ初期条件を複数手法に与えた結果",
        "",
        f"- シナリオ: 単位箱 [0,1]² の上部の円形水塊 (半径 {RADIUS}, 中心 ({CENTER[0]:g}, {CENTER[1]:g})) を重力 {G} で落とす。初期速度ゼロ。",
        f"- 目標粒子数 {particles}、シミュレーション時間 {duration:g} 秒。指標は同じシミュレーション時刻で比較。",
        "- 質量・エネルギー・運動量は初期総質量 M0 で割った**比**で、単位系に依存しない。",
        "- 再現: `python FluidKit/tools/compare_solvers.py --output FluidKit/docs/COMPARE_ATLAS.md`",
        "",
        f"## 1. 要約 (t = {final_t:.2f} s)",
        "",
        "| 手法 | 要素 | 時間刻み | ms/step | ms/シミュ秒 | 質量比 | KE/M0 [m²/s²] | P_y/M0 [m/s] | 重心 y [m] |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        m = r["rows"][-1]
        ms = float(np.mean(r["runner"].step_ms))
        per_sec = ms / r["runner"].dt
        lines.append(
            f"| {r['runner'].name} | {r['runner'].elements} | {r['runner'].dt:g} s | "
            f"{ms:.2f} | {per_sec:.0f} | {m['mass_ratio']:.3f} | {m['ke']:.3f} | "
            f"{m['py']:.3f} | {m['com_y']:.3f} |")
    lines.append("| GNN (Neural Fluid) | — | — | — | — | — | — | — | — |")
    lines += [
        "",
        "GNN は学習済みチェックポイントが必要なため未計測 (`neural/infer_v2.py` で生成して viewer に重ねる)。",
        "",
        "## 2. 重心の高さの推移 [m]",
        "",
        "| t [s] | 理論 (自由落下) | " + " | ".join(r["runner"].name for r in results) + " |",
        "|---|---|" + "---|" * len(results),
    ]
    for i, t in enumerate(sample_times):
        ff = free_fall_y(t)
        theory = f"{ff:.3f}" if ff >= RADIUS else "— (床に接触)"
        cells = " | ".join(f"{r['rows'][i]['com_y']:.3f}" for r in results)
        lines.append(f"| {t:.2f} | {theory} | {cells} |")
    lines += [
        "",
        "理論値は空気抵抗を無視した自由落下 y₀ − ½gt²。水塊の下端が床に届く (重心 ≈ 半径 0.12) 以降は定義できないため省略。",
        "",
        "## 3. 読み方の注意",
        "",
        "- **格子法 (Stam)** は気体向けの密度場で、空気側も非圧縮として解くため、水塊は自由落下せず遅れる。"
        "質量比が 1 から大きく外れる (密度が保存されない) 原因は未調査 (圧力投影の残差・移流の補間・境界処理が候補)。",
        "- **SPH** の壁は反射のみ (床の斥力なし) で、着地後は粒子が床に押しつぶされる。",
        "- **SPH / FLIP** は粒子数が一定なので質量比は構成上 1.0。保存するかどうかを見るには運動量・エネルギーを見る。",
        "- ms/シミュ秒は時間刻みの違いを吸収したコスト。SPH は陽解法のため小さい刻みが要る。",
        "- いずれも学習用の簡易実装の値で、**物理的に厳密ではない**。手法の性格の比較として読むこと。",
        "",
    ]
    if curl:
        lines += curl_markdown(curl)
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="比較 Atlas: 同じ初期条件を複数手法に与える")
    ap.add_argument("--particles", type=int, default=400, help="目標粒子数 (粒子法)")
    ap.add_argument("--duration",  type=float, default=0.6, help="シミュレーション時間 [s]")
    ap.add_argument("--sample-dt", type=float, default=0.1, help="指標を記録する間隔 [s]")
    ap.add_argument("--output", default=None, help="Markdown の出力先")
    ap.add_argument("--export-viewer", action="store_true", help="viewer/data/compare_*.json も出力")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # Windows(cp932) 対策

    results = []
    sample_times = None
    for cls in (SPHRunner, FLIPRunner, GridRunner):
        runner = cls(args.particles)
        print(f"[compare] {runner.name}: {runner.elements} ...", flush=True)
        times, rows, frames, speeds = run_scenario(runner, args.duration, args.sample_dt,
                                                   args.export_viewer)
        sample_times = sample_times or times
        results.append(dict(runner=runner, rows=rows))
        if args.export_viewer:
            out = export_viewer(runner, frames, speeds, _FLUIDKIT / "viewer" / "data")
            print(f"           viewer 用 JSON -> {out}")

    print("[compare] カールノイズ vs 格子法 ...", flush=True)
    md = build_markdown(results, sample_times, args.particles, args.duration, compare_curl_vs_stam())
    if args.output:
        Path(args.output).write_text(md, encoding="utf-8")
        print(f"[compare] saved -> {args.output}")
    else:
        print(md)


if __name__ == "__main__":
    main()
