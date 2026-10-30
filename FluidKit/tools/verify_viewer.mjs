// verify_viewer.mjs — ヘッドレス Chrome + CDP で viewer/index.html を検証する (依存パッケージなし・Node 22+)
//
//   node FluidKit/tools/verify_viewer.mjs [--gpu] [data1.json data2.json ...]
//
//   既定のデータ: viewer/data/compare_{sph,flip,grid}.json
//     (無ければ先に `python FluidKit/tools/compare_solvers.py --export-viewer` で生成する)
//   Chrome の場所: 環境変数 CHROME、無ければ一般的なインストール先 / PATH を探す
//   --gpu: 実 GPU で描画する (既定はソフトウェア描画 SwiftShader。CI や GPU なしでも動く)
//
// 確認すること (「見た目」ではなく値で確認する):
//   1. コンソールエラー・例外が無い
//   2. 3 本読み込むと手法チップが有効になり、「比較」が使える
//   3. 比較モードで全ビューが同じ時刻を指す (fps や長さが違っても 1 フレーム以内)
//   4. シークしたフレームの位置が、データの該当フレームと一致する
//   5. 凡例の最小・最大が、実データの最小・最大と一致する
// 失敗があれば終了コード 1。
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const viewerDir = path.resolve(here, '..', 'viewer');
const args = process.argv.slice(2);
const useGpu = args.includes('--gpu');
let files = args.filter(a => !a.startsWith('--')).map(f => path.resolve(f));
if (!files.length) files = ['sph', 'flip', 'grid'].map(m => path.join(viewerDir, 'data', `compare_${m}.json`));

function findChrome() {
  const cands = [process.env.CHROME,
    'C:/Program Files/Google/Chrome/Application/chrome.exe', 'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe',
    'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
    '/usr/bin/google-chrome', '/usr/bin/google-chrome-stable', '/usr/bin/chromium', '/usr/bin/chromium-browser',
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'].filter(Boolean);
  return cands.find(p => fs.existsSync(p));
}
const chromePath = findChrome();
if (!chromePath) { console.error('Chrome が見つかりません。環境変数 CHROME で指定してください。'); process.exit(2); }
for (const f of files) if (!fs.existsSync(f)) {
  console.error(`データがありません: ${f}\n先に: python FluidKit/tools/compare_solvers.py --export-viewer`); process.exit(2);
}

const sleep = ms => new Promise(r => setTimeout(r, ms));
const profile = fs.mkdtempSync(path.join(os.tmpdir(), 'fluidkit-verify-'));
const port = 9300 + Math.floor(Math.random() * 400);
const chrome = spawn(chromePath, [
  '--headless=new', `--remote-debugging-port=${port}`, `--user-data-dir=${profile}`, '--window-size=1400,900',
  ...(useGpu ? ['--use-angle=d3d11'] : ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader']),
  '--ignore-gpu-blocklist', '--allow-file-access-from-files', 'about:blank'], { stdio: 'ignore' });

const results = [];
const check = (name, ok, detail = '') => { results.push({ name, ok }); console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? '  — ' + detail : ''}`); };
const cleanup = () => { try { chrome.kill(); } catch {} setTimeout(() => fs.rmSync(profile, { recursive: true, force: true, maxRetries: 5 }), 500); };

try {
  let targets;
  for (let i = 0; i < 60 && !targets?.length; i++) {
    try { targets = await (await fetch(`http://127.0.0.1:${port}/json`)).json(); } catch { await sleep(200); }
  }
  const page = targets.find(t => t.type === 'page');
  const ws = new WebSocket(page.webSocketDebuggerUrl);
  await new Promise(r => ws.addEventListener('open', r));
  let id = 0; const pending = new Map(); const errors = [];
  ws.addEventListener('message', e => {
    const m = JSON.parse(e.data);
    if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); }
    else if (m.method === 'Runtime.exceptionThrown') errors.push(m.params.exceptionDetails.exception?.description || m.params.exceptionDetails.text);
    else if (m.method === 'Runtime.consoleAPICalled' && m.params.type === 'error') errors.push(m.params.args.map(a => a.value ?? a.description).join(' '));
  });
  const send = (method, params = {}) => new Promise(res => { const i = ++id; pending.set(i, res); ws.send(JSON.stringify({ id: i, method, params })); });
  const ev = async expr => { const r = await send('Runtime.evaluate', { expression: expr, awaitPromise: true, returnByValue: true }); return r.result.result?.value; };

  await send('Runtime.enable'); await send('Page.enable');
  await send('Page.navigate', { url: pathToFileURL(path.join(viewerDir, 'index.html')).href });
  for (let i = 0; i < 60 && !(await ev('!!window.fluidViewer && fluidViewer.datasets.length > 0')); i++) await sleep(250);
  check('ビューアが起動し、デモが読み込まれる', await ev(`fluidViewer.active && fluidViewer.active.method === 'DEMO'`));

  // 実際のファイル入力経由で読み込む (ドロップと同じ経路)
  const doc = await send('DOM.getDocument');
  const inp = await send('DOM.querySelector', { nodeId: doc.result.root.nodeId, selector: '#file-input' });
  await send('DOM.setFileInputFiles', { nodeId: inp.result.nodeId, files });
  for (let i = 0; i < 80 && (await ev('fluidViewer.realDatasets.length')) < files.length; i++) await sleep(250);
  const n = await ev('fluidViewer.realDatasets.length');
  check(`${files.length} 本の JSON を読み込める`, n === files.length, `datasets=${n}`);
  if (n !== files.length) throw new Error('読み込み失敗のため以降を中止');

  check('手法チップが有効になり、比較が使える',
    await ev(`!document.getElementById('btn-compare').disabled && [...document.querySelectorAll('#seg-method button')].filter(b => !b.disabled).length === ${files.length}`));

  await ev(`document.getElementById('btn-compare').click()`);
  await ev(`fluidViewer.setPlaying(false); fluidViewer.seekFrame(Math.floor(fluidViewer.master.n * 0.6))`);
  await sleep(600);

  const sync = JSON.parse(await ev(`JSON.stringify(fluidViewer.visible.map(d => ({ m: d.method, fps: d.fps, t: fluidViewer.views.get(d.id).frame / d.fps })))`));
  const maxFrameDt = Math.max(...sync.map(s => 1 / s.fps));
  const spread = Math.max(...sync.map(s => s.t)) - Math.min(...sync.map(s => s.t));
  check('比較モードで全ビューが同じ時刻を指す', spread <= maxFrameDt + 1e-9, `時刻のばらつき ${spread.toFixed(3)} s (許容 ${maxFrameDt.toFixed(3)} s)`);

  const seek = JSON.parse(await ev(`(() => { fluidViewer.seekFrame(10); const m = fluidViewer.master, v = fluidViewer.views.get(m.id);
    const pos = v.points.geometry.getAttribute('position').array, f = m.frames[v.frame]; let ok = true;
    for (let i = 0; i < Math.min(60, f.length); i++) if (Math.abs(pos[i] - f[i]) > 1e-4) ok = false;
    return JSON.stringify({ frame: v.frame, ok }); })()`));
  check('シークしたフレームの位置がデータと一致する', seek.ok && seek.frame === 10, `frame=${seek.frame}`);

  const lg = JSON.parse(await ev(`(() => { fluidViewer.setColorMode(1); let mn = Infinity, mx = -Infinity;
    fluidViewer.visible.forEach(d => d.speeds.forEach(s => { for (const x of s) { if (x < mn) mn = x; if (x > mx) mx = x; } }));
    return JSON.stringify({ mn, mx, lmin: +document.getElementById('lg-min').textContent, lmax: +document.getElementById('lg-max').textContent }); })()`));
  check('凡例の最小・最大が実データと一致する',
    Math.abs(lg.lmin - lg.mn) < 0.006 && Math.abs(lg.lmax - lg.mx) < 0.006, `凡例 ${lg.lmin}..${lg.lmax} / データ ${lg.mn.toFixed(2)}..${lg.mx.toFixed(2)}`);

  check('コンソールエラー・例外が無い', errors.length === 0, errors.slice(0, 3).join(' | '));
  ws.close();
} catch (e) {
  check('検証を最後まで実行できる', false, String(e.message || e));
} finally {
  cleanup();
  const failed = results.filter(r => !r.ok).length;
  console.log(failed ? `\n${failed} 件失敗` : `\n全 ${results.length} 件 OK`);
  process.exit(failed ? 1 : 0);
}
