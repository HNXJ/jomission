// Headless check for the v3 page: runs the page's own script with a stubbed DOM and reads its stats.
const fs = require('fs');
const vm = require('vm');
const src = fs.readFileSync(process.argv[2], 'utf8');
const seconds = parseInt(process.argv[3] || '30', 10);

const ctx = new Proxy({ measureText: () => ({ width: 10 }) }, {
  get: (t, p) => (p in t ? t[p] : () => {}),
  set: () => true,
});
const els = {};
function mkEl(id) {
  if (!els[id]) els[id] = { id, textContent: '', value: '0', checked: true, width: 640, height: 520,
    addEventListener() {}, getContext() { return ctx; }, getBoundingClientRect() { return { width: 600, height: 400 }; } };
  return els[id];
}
let frames = [];
let now = 0;
globalThis.document = { getElementById: mkEl, documentElement: {}, querySelectorAll: () => [] };
globalThis.getComputedStyle = () => ({ getPropertyValue: () => '' });
globalThis.window = { devicePixelRatio: 1 };
globalThis.requestAnimationFrame = (cb) => { frames.push(cb); };
globalThis.performance = { now: () => now };

const t0 = Date.now();
vm.runInThisContext(src);
const FR = 1000 / 60;
const errors = [];
for (let f = 1; f <= seconds * 60; f++) {
  now += FR;
  const cbs = frames; frames = [];
  for (const cb of cbs) {
    try { cb(now); } catch (e) { errors.push(String(e && e.stack || e).slice(0, 300)); }
  }
  if (errors.length > 2) break;
  if (f % 300 === 0) {
    const g = (id) => mkEl(id).textContent;
    console.log(`t=${f / 60}s e=${g('s_e')} fix=${g('s_fix')} f=${g('s_f')} rE=${g('s_rE')} rI=${g('s_rI')}`);
  }
}
const wall = (Date.now() - t0) / 1000;
console.log('run info: ' + mkEl('runinfo').textContent);
console.log(`wall ${wall.toFixed(1)} s for ${seconds} s of frames`);
console.log(errors.length ? 'ERRORS: ' + errors.join(' | ') : 'no runtime errors');
