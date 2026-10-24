// heightfield.js のテスト (node --test)。pytest からは tests/test_heightfield_js.py 経由で実行される。
'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const { HeightField } = require('./heightfield.js');

function bump(hf, amp = 0.1) { hf.disturb(hf.size / 2, hf.size / 2, 0.12, amp); }
function run(hf, seconds, dtFrame) { for (let t = 0; t < seconds; t += dtFrame) hf.step(dtFrame); }

test('時間刻みを変えても発散しない (自動サブステップ)', () => {
  for (const dtFrame of [1 / 240, 1 / 60, 1 / 15, 0.25, 1.0]) {
    const hf = new HeightField({ n: 64, speed: 1.2, damping: 0.3 });
    bump(hf, 0.1);
    run(hf, 3, dtFrame);
    const m = hf.maxAbs();
    assert.ok(Number.isFinite(m), `dt=${dtFrame}: 非有限値`);
    assert.ok(m <= 0.1, `dt=${dtFrame}: 振幅が初期値 0.1 を超えた (${m})`);
  }
});

test('大きい dt は安定刻み以下のサブステップに分割される', () => {
  const hf = new HeightField({ n: 64, speed: 2.0 });
  bump(hf);
  const sub = hf.step(0.25);
  assert.ok(sub > 1);
  assert.ok(hf.lastDt <= hf.stableDt() + 1e-12);
});

test('安定条件を無視して直接 1 ステップ進めると発散する (自動分割の必要性の確認)', () => {
  const hf = new HeightField({ n: 64, speed: 1.2, damping: 0, sponge: 0 });
  bump(hf);
  for (let s = 0; s < 200; s++) hf._substep(hf.stableDt() * 3);      // 安定限界の 3 倍
  assert.ok(!(hf.maxAbs() < 1), '不安定になるはずが有界のまま');
});

// 以下 2 つは波が縁に届く前 (< 0.6 s) に、安定限界の 1/4 の細かい刻みで測る。
// 限界ぎりぎりの刻みでは離散スキームの保存量が影のエネルギーとずれるため、ここでは検証対象にしない。
test('減衰・吸収層なしならエネルギーはほぼ保存される', () => {
  const hf = new HeightField({ n: 96, speed: 1.2, damping: 0, sponge: 0 });
  bump(hf);
  const dt = 0.25 * hf.stableDt();
  hf._substep(dt);
  const e0 = hf.energy();
  let worst = 0;
  for (let t = 0; t < 0.6; t += dt) { hf._substep(dt); worst = Math.max(worst, Math.abs(hf.energy() - e0) / e0); }
  assert.ok(worst < 0.03, `エネルギー変動 ${(worst * 100).toFixed(2)}%`);
});

test('減衰ありならエネルギーは(0.1 秒ごとに)減っていく', () => {
  // 1 ステップ単位では半陰的 Euler の微小な振動 (~0.06%) があるため、減衰による減少 (γ=0.8 で 0.1 秒に約 8%) が
  // 明確に見える間隔で比べる。
  const hf = new HeightField({ n: 96, speed: 1.2, damping: 0.8, sponge: 0 });
  bump(hf);
  const dt = 0.25 * hf.stableDt();
  hf._substep(dt);
  let prev = hf.energy();
  for (let k = 1; k <= 6; k++) {
    for (let t = 0; t < 0.1; t += dt) hf._substep(dt);
    const e = hf.energy();
    assert.ok(e < prev * 0.97, `${k * 0.1} s: エネルギーが十分に減っていない ${prev} → ${e}`);
    prev = e;
  }
});

test('傾きの計算: h = a·x なら ∂h/∂x = a, ∂h/∂y = 0', () => {
  const hf = new HeightField({ n: 33, size: 2.0 });
  const a = 0.3;
  for (let j = 0; j < hf.n; j++) for (let i = 0; i < hf.n; i++) hf.h[j * hf.n + i] = a * i * hf.dx;
  for (const [i, j] of [[5, 5], [0, 7], [32, 20]]) {
    const [gx, gy] = hf.gradient(i, j);
    assert.ok(Math.abs(gx - a) < 1e-5 && Math.abs(gy) < 1e-6, `(${i},${j}) → (${gx}, ${gy})`);
  }
});

test('平らな水面のコースティクスは一様 (平均 ≈ 1)', () => {
  const hf = new HeightField({ n: 128 });
  const c = hf.caustics(1.0);
  assert.ok(c.mean > 0.95 && c.mean < 1.05, `mean=${c.mean}`);
  assert.ok(c.max < 1.8, `一様のはずが max=${c.max}`);
});

test('波があると光が集まる (max が平均より大きく、光線は増えない)', () => {
  const hf = new HeightField({ n: 128, speed: 1.2, damping: 0.1 });
  bump(hf, 0.1);
  run(hf, 0.5, 1 / 60);
  const c = hf.caustics(1.5);
  assert.ok(c.max > 2.0 * c.mean, `max=${c.max}, mean=${c.mean}`);
  assert.ok(c.mean <= 1.0 + 1e-6, `光線数が増えた mean=${c.mean}`);       // 床の外へ出る分だけ減るのみ
});

test('同じ入力なら同じ結果 (決定的)', () => {
  const make = () => { const h = new HeightField({ n: 48 }); bump(h); run(h, 0.5, 1 / 60); return h; };
  const a = make(), b = make();
  assert.deepEqual(Array.from(a.h), Array.from(b.h));
});
