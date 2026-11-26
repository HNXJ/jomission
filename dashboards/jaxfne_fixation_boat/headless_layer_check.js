// Layer-readout checks for the Fixation Boat page. Loads the page's own script in a Node vm with a
// stubbed DOM, exposes build() and parseCfg() by a text injection (in memory only; the page file is
// not changed), and checks what the motorLayer parameter does and does not change.
// Usage: node headless_layer_check.js <path to index.html>
const fs = require('fs');
const vm = require('vm');
const assert = require('assert');

const file = process.argv[2];
const html = fs.readFileSync(file, 'utf8');
const m = html.match(/<script>([\s\S]*?)<\/script>/);
if (!m) throw new Error('no script block in ' + file);
let src = m[1];
const cut = src.lastIndexOf('})();');
if (cut < 0) throw new Error('could not find the page IIFE end');
src = src.slice(0, cut) + 'globalThis.__t = { build, parseCfg, DEFAULTS }; ' + src.slice(cut);

function stubEl(id) {
  return { id, textContent: '', value: '0', checked: true, width: 640, height: 520, className: '', innerHTML: '',
    addEventListener() {}, getContext() { return new Proxy({}, { get: () => () => ({ width: 10 }), set: () => true }); },
    getBoundingClientRect() { return { width: 600, height: 400 }; } };
}
globalThis.document = { getElementById: stubEl, documentElement: {}, querySelectorAll: () => [] };
globalThis.getComputedStyle = () => ({ getPropertyValue: () => '' });
globalThis.window = { devicePixelRatio: 1 };
globalThis.requestAnimationFrame = () => {};
globalThis.performance = { now: () => 0 };
vm.runInThisContext(src);
const { build, parseCfg, DEFAULTS } = globalThis.__t;

const net = (over) => build(parseCfg(JSON.stringify(over || {})));
const cellInfo = (idx, n) => ({
  count: idx.length,
  layers: [...new Set([...idx].map(i => n.layer[i]))],
  types: [...new Set([...idx].map(i => n.type[i]))],
  areas: [...new Set([...idx].map(i => n.area[i]))],
});
const results = [];
const check = (name, fn) => { try { fn(); results.push('PASS ' + name); } catch (e) { results.push('FAIL ' + name + ': ' + e.message); } };
const throwsWith = (over, text) => { let err = null; try { net(over); } catch (e) { err = e; } assert.ok(err, 'expected a refusal'); assert.ok(err.message.includes(text), 'message was: ' + (err && err.message)); };

check('default: readout and cue both on L5 (index 4) in the top area, 20 E cells', () => {
  const n = net();
  assert.deepStrictEqual(cellInfo(n.motor, n), { count: 20, layers: [4], types: [0], areas: [1] });
  assert.deepStrictEqual(Array.from(n.cueCells), Array.from(n.motor));
});
check('motorLayer 5: readout moves to L6 (index 5), same count and area', () => {
  const n = net({ motorLayer: 5 });
  assert.deepStrictEqual(cellInfo(n.motor, n), { count: 20, layers: [5], types: [0], areas: [1] });
});
check('motorLayer 5: cue pool unchanged (still L5, the same cells as default)', () => {
  const a = net(), b = net({ motorLayer: 5 });
  assert.deepStrictEqual(Array.from(b.cueCells), Array.from(a.cueCells));
  assert.deepStrictEqual(cellInfo(b.cueCells, b), { count: 20, layers: [4], types: [0], areas: [1] });
});
check('motorLayer changes the readout list, not the wiring', () => {
  const a = net(), b = net({ motorLayer: 5 });
  assert.notDeepStrictEqual(Array.from(a.motor), Array.from(b.motor));
  assert.deepStrictEqual(Array.from(a.pre), Array.from(b.pre));
  assert.deepStrictEqual(Array.from(a.w), Array.from(b.w));
  assert.deepStrictEqual(Array.from(a.type), Array.from(b.type));
  assert.deepStrictEqual(Array.from(a.bias), Array.from(b.bias));
});
check('explicit motorLayer 4 gives the same readout and cue as the default', () => {
  const a = net(), b = net({ motorLayer: 4 });
  assert.deepStrictEqual(Array.from(a.motor), Array.from(b.motor));
  assert.deepStrictEqual(Array.from(a.cueCells), Array.from(b.cueCells));
});
check('boundary: 120 neurons, motorPerDir 4, motorLayer 5 (readout 16 E cells, cue 17) is accepted', () => {
  const n = net({ neuronsPerColumn: 120, areas: 1, motorPerDir: 4, motorLayer: 5 });
  assert.deepStrictEqual(cellInfo(n.motor, n), { count: 16, layers: [5], types: [0], areas: [0] });
  assert.deepStrictEqual(cellInfo(n.cueCells, n), { count: 16, layers: [4], types: [0], areas: [0] });
});
check('negative: 120 neurons, motorPerDir 5 is refused: L5 has 17 cue cells, 20 needed', () => {
  throwsWith({ neuronsPerColumn: 120, areas: 1, motorPerDir: 5, motorLayer: 5 }, 'layer 5 of the top area (cue)');
});
check('negative: motorLayer 0 (L1 has no E cells) is refused, naming layer 1 as the readout', () => {
  throwsWith({ motorLayer: 0 }, 'layer 1 of the top area (readout)');
});
check('validation: motorLayer 6 is rejected by parseCfg, 5 is accepted', () => {
  assert.throws(() => parseCfg(JSON.stringify({ motorLayer: 6 })), /motorLayer/);
  assert.strictEqual(parseCfg(JSON.stringify({ motorLayer: 5 })).motorLayer, 5);
});
check('defaults object carries motorLayer 4', () => {
  assert.strictEqual(DEFAULTS.motorLayer, 4);
});

check('default at 120 neurons, motorPerDir 4: readout and cue are the same 16 L5 cells', () => {
  const n = net({ neuronsPerColumn: 120, areas: 1, motorPerDir: 4 });
  assert.deepStrictEqual(Array.from(n.cueCells), Array.from(n.motor));
  assert.deepStrictEqual(cellInfo(n.motor, n), { count: 16, layers: [4], types: [0], areas: [0] });
});
check('motorLayer 5: readout pool is disjoint from the cue pool (L6 vs L5)', () => {
  const n = net({ motorLayer: 5 });
  const cue = new Set(n.cueCells);
  assert.strictEqual(n.motor.filter(i => cue.has(i)).length, 0);
});
check('motorLayer 5: no cue cell reaches a readout cell by any directed path (edge pre[k] -> n)', () => {
  const n = net({ motorLayer: 5 });
  const out = Array.from({ length: n.type.length }, () => []);
  for (let t = 0; t < n.type.length; t++) for (let k = n.start[t]; k < n.start[t + 1]; k++) out[n.pre[k]].push(t);
  const reached = new Set(n.cueCells), queue = Array.from(n.cueCells);
  while (queue.length) { const s = queue.shift(); for (const t of out[s]) if (!reached.has(t)) { reached.add(t); queue.push(t); } }
  assert.strictEqual(n.motor.filter(i => reached.has(i)).length, 0, 'a readout cell is reached from the cue');
});

check('visible note states the L5 pool identity, the L6 limit, the evidence split and the seed and alpha limits', () => {
  const m = html.match(/<p class="note">([\s\S]*?)<\/p>/);
  assert.ok(m, 'no note paragraph in the page');
  const note = m[1].replace(/\s+/g, ' ');
  for (const phrase of [
    'the readout cells and the cue cells are the same 20 L5 E cells',
    '16 cells in total at 120 neurons per column with motorPerDir 4',
    'L6 is unsupported under the current connectivity',
    'identical at cue 0 and cue 0.2',
    'rE and rI differ',
    'the cue-only case (α 0) was not re-measured',
    'seeds 1–5',
  ]) assert.ok(note.includes(phrase), 'missing from the visible note: ' + phrase);
});

console.log(results.join('\n'));
const failed = results.filter(r => r.startsWith('FAIL')).length;
console.log(failed ? failed + ' FAILED' : 'all ' + results.length + ' layer checks passed');
process.exitCode = failed ? 1 : 0;
