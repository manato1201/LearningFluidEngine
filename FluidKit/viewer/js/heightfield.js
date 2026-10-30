/**
 * heightfield.js — 波紋の高さ場シミュレーションと、床のコースティクス(光の集まり)の計算
 *
 * DOM に依存しない純粋なロジック。ブラウザでは <script src> で読み込み (globalThis.FluidKitHeightField)、
 * Node からは require() でテストできる (file:// で開く viewer から ES module を読めないための形式)。
 *
 * 高さ場 (2D 波動方程式、減衰つき):
 *     h_tt = c² ∇²h − γ h_t
 * 半陰的 (symplectic) Euler で解く。5 点ラプラシアンの最大固有値は 8/dx² なので、
 * 安定条件は dt ≤ dx / (c√2)。step() は要求された dt がこれを超えると自動でサブステップに分割する
 * (= 時間刻みを大きくしても発散しない)。
 *
 * コースティクス: 鉛直入射の光が水面の傾きで屈折し、深さ D の床に届く位置を求めて、
 * 着地点に重み 1 を撒き (photon splat)、床のセルあたりの本数 = 強度 (一様なら 1) とする。
 * 小さい傾きの近似: 着地点 = 水面の点 + D (1−η) ∇h,  η = 1/n_water (≈ 1/1.33)。
 * (山の斜面は ∇h が山頂と逆向きなので、光は山頂側へ曲がる = 山は凸レンズのように集光する)
 *
 * これは見た目用の近似。水面の非線形な効果・多重反射・波長依存は扱わない。
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.FluidKitHeightField = factory();
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';

  class HeightField {
    /**
     * @param {object} o
     * @param {number} o.n        一辺の格子点数
     * @param {number} o.size     水面の一辺の長さ [m]
     * @param {number} o.speed    波速 c [m/s]
     * @param {number} o.damping  減衰 γ [1/s]
     * @param {number} o.sponge   縁の吸収層の幅 (格子点数)。反射波が戻らないようにする
     */
    constructor({ n = 128, size = 2.0, speed = 1.2, damping = 0.3, sponge = 10 } = {}) {
      this.n = n; this.size = size; this.dx = size / (n - 1);
      this.speed = speed; this.damping = damping; this.sponge = sponge;
      this.h = new Float32Array(n * n);
      this.v = new Float32Array(n * n);
      this.time = 0;
      this.lastSubsteps = 1; this.lastDt = 0;
    }

    reset() { this.h.fill(0); this.v.fill(0); this.time = 0; }

    /** 安定な最大時間刻み。安全率 0.9 を掛ける。 */
    stableDt() { return 0.9 * this.dx / (this.speed * Math.SQRT2); }

    /** 点 (x, y) [0,size] に半径 radius のガウス状の盛り上がりを足す。 */
    disturb(x, y, radius, amplitude) {
      const { n, dx, h } = this;
      const cx = x / dx, cy = y / dx, r = radius / dx, r2 = r * r;
      const i0 = Math.max(0, Math.floor(cx - 3 * r)), i1 = Math.min(n - 1, Math.ceil(cx + 3 * r));
      const j0 = Math.max(0, Math.floor(cy - 3 * r)), j1 = Math.min(n - 1, Math.ceil(cy + 3 * r));
      for (let j = j0; j <= j1; j++) for (let i = i0; i <= i1; i++) {
        const d2 = (i - cx) * (i - cx) + (j - cy) * (j - cy);
        h[j * n + i] += amplitude * Math.exp(-d2 / (2 * r2 / 4));
      }
    }

    /** dtFrame だけ進める。必要ならサブステップに分割する。 */
    step(dtFrame) {
      const sub = Math.max(1, Math.ceil(dtFrame / this.stableDt()));
      const dt = dtFrame / sub;
      for (let s = 0; s < sub; s++) this._substep(dt);
      this.time += dtFrame;
      this.lastSubsteps = sub; this.lastDt = dt;
      return sub;
    }

    _substep(dt) {
      const { n, h, v, dx, speed, damping, sponge } = this;
      const k = (speed * speed) / (dx * dx);
      for (let j = 0; j < n; j++) {
        const jm = j > 0 ? j - 1 : 1, jp = j < n - 1 ? j + 1 : n - 2;       // 端は鏡像 (Neumann)
        for (let i = 0; i < n; i++) {
          const im = i > 0 ? i - 1 : 1, ip = i < n - 1 ? i + 1 : n - 2;
          const idx = j * n + i, c = h[idx];
          const lap = h[j * n + im] + h[j * n + ip] + h[jm * n + i] + h[jp * n + i] - 4 * c;
          const edge = Math.min(i, j, n - 1 - i, n - 1 - j);
          const gamma = damping + (edge < sponge ? 6.0 * (1 - edge / sponge) ** 2 : 0);   // 縁の吸収層
          v[idx] += dt * (k * lap - gamma * v[idx]);
        }
      }
      for (let idx = 0; idx < h.length; idx++) h[idx] += dt * v[idx];
    }

    /**
     * 運動エネルギー + 位置エネルギー。位置エネルギーは格子の全ての辺 (隣り合う 2 点) の高さの差から。
     * 減衰・吸収層がなく、波が縁に届く前なら (ほぼ) 保存される量。
     */
    energy() {
      const { n, h, v, dx, speed } = this;
      let kin = 0, pot = 0;
      for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
        const idx = j * n + i;
        kin += v[idx] * v[idx];
        if (i < n - 1) { const d = h[idx + 1] - h[idx]; pot += d * d; }
        if (j < n - 1) { const d = h[idx + n] - h[idx]; pot += d * d; }
      }
      return 0.5 * dx * dx * kin + 0.5 * speed * speed * pot;
    }

    maxAbs() { let m = 0; for (const x of this.h) m = Math.max(m, Math.abs(x)); return m; }

    /** 中心差分の傾き (∂h/∂x, ∂h/∂y)。端は片側差分。 */
    gradient(i, j) {
      const { n, h, dx } = this;
      const im = Math.max(i - 1, 0), ip = Math.min(i + 1, n - 1), jm = Math.max(j - 1, 0), jp = Math.min(j + 1, n - 1);
      return [(h[j * n + ip] - h[j * n + im]) / ((ip - im) * dx), (h[jp * n + i] - h[jm * n + i]) / ((jp - jm) * dx)];
    }

    /**
     * 床のコースティクス強度を計算する。
     * @param {number} depth  水深 D [m]
     * @param {number} eta    屈折率の比 (空気/水 ≈ 0.75)
     * @param {number} res    床の解像度 (格子点数)。水面より粗くして光線ノイズを減らす
     * @returns {{floor: Float32Array, res: number, mean: number, max: number, landed: number}}
     *   mean は床に着いた光線数 / 床の格子数。光線が床の外へ出なければ ≈ 1 (エネルギー保存の目安)
     */
    caustics(depth, eta = 1 / 1.33, res = Math.floor(this.n / 2)) {
      const { n, dx, size } = this;
      const floor = new Float32Array(res * res), cell = size / (res - 1);
      let landed = 0;
      const bend = depth * (1 - eta);
      for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
        const [gx, gy] = this.gradient(i, j);
        const fx = (i * dx + bend * gx) / cell, fy = (j * dx + bend * gy) / cell;     // 床の格子座標
        if (fx < 0 || fy < 0 || fx > res - 1 || fy > res - 1) continue;             // 床の外へ出た光線
        const x0 = Math.min(Math.floor(fx), res - 2), y0 = Math.min(Math.floor(fy), res - 2);
        const tx = fx - x0, ty = fy - y0;
        floor[y0 * res + x0] += (1 - tx) * (1 - ty); floor[y0 * res + x0 + 1] += tx * (1 - ty);
        floor[(y0 + 1) * res + x0] += (1 - tx) * ty; floor[(y0 + 1) * res + x0 + 1] += tx * ty;
        landed++;
      }
      // 1 光線 = 水面 1 セル分の光。床 1 セルが受ける平均を 1 に正規化する
      const scale = (res * res) / (n * n);
      let sum = 0, max = 0;
      for (let k = 0; k < floor.length; k++) { floor[k] *= scale; sum += floor[k]; if (floor[k] > max) max = floor[k]; }
      return { floor, res, mean: sum / floor.length, max, landed: landed / (n * n) };
    }
  }

  return { HeightField };
});
