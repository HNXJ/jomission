// Headless check with event capture: runs the page script with a stubbed DOM and fires its
// listeners on a schedule (apply JSON, invalid JSON, unknown key, bad fan-in, play/pause,
// network size, reset, preset, moved target/start).
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
  if (!els[id]) els[id] = {
    id, textContent: '', value: '0', checked: true, width: 640, height: 520, className: '', _l: {},
    addEventListener(type, cb) { (this._l[type] = this._l[type] || []).push(cb); },
    getContext() { return ctx; },
    getBoundingClientRect() { return { width: 600, height: 400 }; },
  };
  return els[id];
}
function fire(id, type = 'click') { const el = mkEl(id); (el._l[type] || []).forEach(cb => cb({ target: el })); }
const presetBtns = ['spec', 'haptic', 'cue', 'mixed'].map(name => ({
  name, getAttribute() { return name; }, _l: {},
  addEventListener(t, cb) { (this._l[t] = this._l[t] || []).push(cb); },
}));
const firePreset = (name) => { const b = presetBtns.find(x => x.name === name); (b._l.click || []).forEach(cb => cb({ target: b })); };
let frames = [];
let now = 0;
globalThis.document = {
  getElementById: mkEl, documentElement: {},
  querySelectorAll: (sel) => (sel === '[data-preset]' ? presetBtns : []),
};
globalThis.getComputedStyle = () => ({ getPropertyValue: () => '' });
globalThis.window = { devicePixelRatio: 1 };
globalThis.requestAnimationFrame = (cb) => { frames.push(cb); };
globalThis.performance = { now: () => now };

const setJSON = (obj) => { mkEl('cfgText').value = typeof obj === 'string' ? obj : JSON.stringify(obj); };
const msg = () => `msg="${mkEl('cfgMsg').textContent}" cls="${mkEl('cfgMsg').className}"`;
const snap = (label) => {
  const run = mkEl('runinfo').textContent;
  const cells = (run.match(/(\d+) cells total/) || [])[1];
  console.log(`[${label}] e=${mkEl('s_e').textContent} fix=${mkEl('s_fix').textContent} cells=${cells} ` +
    `label="${mkEl('l_fix').textContent}" alpha="${mkEl('o_alpha').textContent}" cue="${mkEl('o_cue').textContent}"`);
};

const actions = {
  5: () => snap('defaults, 5 s'),
  8: () => { setJSON({ alpha: 0, cue: 0 }); fire('applyCfg'); console.log('apply alpha 0, cue 0: ' + msg()); },
  12: () => snap('alpha 0 cue 0, 12 s (expect drift)'),
  13: () => { setJSON('{ "alpha": 0,'); fire('applyCfg'); console.log('invalid JSON: ' + msg()); },
  14: () => { setJSON({ foo: 1 }); fire('applyCfg'); console.log('unknown key: ' + msg()); },
  15: () => { setJSON({ proj: { EE: { k: -3, w: 0.01 } } }); fire('applyCfg'); console.log('negative fan-in: ' + msg()); },
  16: () => { setJSON({ alpha: 'x' }); fire('applyCfg'); console.log('non-numeric alpha: ' + msg()); },
  17: () => snap('after 4 rejected inputs (unchanged from 12 s)'),
  18: () => { fire('play'); console.log('paused: label ' + mkEl('play').textContent); },
  19: () => { fire('play'); console.log('resumed: label ' + mkEl('play').textContent); },
  20: () => { setJSON({ neuronsPerColumn: 120, areas: 1 }); fire('applyCfg'); console.log('120 neurons, 1 area: ' + msg()); },
  22: () => snap('120 neurons, 1 area (expect cells 120 x 1 area = 120)'),
  23: () => { fire('resetCfg'); console.log('reset: ' + msg()); },
  24: () => snap('after reset (expect defaults: 480 cells, alpha 0.50)'),
  25: () => { firePreset('cue'); console.log('cue preset clicked'); },
  27: () => snap('cue preset (alpha 0, cue 0.20)'),
  28: () => { setJSON({ target: [1, 0.5], start: [-2, 0], fixWindow: 300 }); fire('applyCfg'); console.log('moved target/start, fixWindow 300: ' + msg()); },
  31: () => snap('moved target (expect label 10 s window)'),
  32: () => { fire('reset'); snap('boat reset button'); },
};

const t0 = Date.now();
vm.runInThisContext(src);
const FR = 1000 / 60;
const errors = [];
for (let f = 1; f <= seconds * 60; f++) {
  now += FR;
  const sec = Math.round(f / 60);
  if (f % 60 === 0 && actions[sec]) actions[sec]();
  const cbs = frames; frames = [];
  for (const cb of cbs) {
    try { cb(now); } catch (e) { errors.push(String(e && e.stack || e).slice(0, 300)); }
  }
  if (errors.length > 2) break;
}
console.log(errors.length ? 'ERRORS: ' + errors.join(' | ') : 'no runtime errors');
console.log('wall ' + ((Date.now() - t0) / 1000).toFixed(1) + ' s');
