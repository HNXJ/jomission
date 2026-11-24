// Headless check with event capture: runs the page script with a stubbed DOM and fires its
// listeners on a schedule. Covers apply, rejected inputs (syntax, unknown keys, bounds that the
// review flagged), the motor-fit refusal, preset, reset, moved target/start, seed clamp, and
// that unapplied JSON text survives a slider move.
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
const reject = (label, obj) => { setJSON(obj); fire('applyCfg'); console.log(label + ': ' + msg()); };

const actions = {
  5: () => snap('defaults, 5 s'),
  8: () => { setJSON({ alpha: 0, cue: 0 }); fire('applyCfg'); console.log('apply alpha 0, cue 0: ' + msg()); },
  12: () => snap('alpha 0 cue 0, 12 s (expect drift, e about 3)'),
  13: () => reject('invalid JSON', '{ "alpha": 0,'),
  14: () => reject('unknown key', { foo: 1 }),
  15: () => reject('negative fan-in', { proj: { EE: { k: -3, w: 0.01 } } }),
  16: () => reject('non-numeric alpha', { alpha: 'x' }),
  17: () => reject('start 1e308 (review blocker)', { start: [1e308, 0] }),
  18: () => reject('layerFrac row sums to 1000 (review blocker)', { layerFrac: [[0, 1000, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]] }),
  19: () => reject('damping 5000 (review blocker)', { damping: 5000 }),
  20: () => reject('projection k 1000 (review major)', { proj: { EE: { k: 1000 } } }),
  21: () => reject('projection with string value (review minor)', { proj: { EE: 'abc' } }),
  22: () => reject('projection with extra field', { proj: { EE: { k: 6, kk: 1 } } }),
  23: () => reject('120 neurons with motorPerDir 5 (expect motor refusal)', { neuronsPerColumn: 120, areas: 1 }),
  24: () => snap('after 7 rejected inputs (unchanged, cells 480)'),
  25: () => { reject('120 neurons, 1 area, motorPerDir 3', { neuronsPerColumn: 120, areas: 1, motorPerDir: 3 }); },
  27: () => snap('120/1 network (expect cells 120, run line 1 area)'),
  28: () => { fire('resetCfg'); console.log('reset: ' + msg()); },
  29: () => snap('after reset (expect 480 cells)'),
  30: () => { firePreset('cue'); console.log('cue preset clicked'); },
  31: () => snap('cue preset (alpha 0, cue 0.20)'),
  32: () => { setJSON({ target: [1, 0.5], start: [-2, 0], fixWindow: 300 }); fire('applyCfg'); console.log('moved target/start, fixWindow 300: ' + msg()); },
  33: () => snap('moved target (expect label 6 s window)'),
  34: () => { mkEl('seed').value = '99999999'; fire('seed', 'change'); console.log('seed typed 99999999 -> JSON seed ' + (mkEl('cfgText').value.match(/"seed": (\d+)/) || [])[1]); },
  35: () => { fire('applyCfg'); console.log('apply after seed clamp: ' + msg()); },
  36: () => { fire('resetCfg'); setJSON('{"start":[1,1]}'); fire('cfgText', 'input'); mkEl('alpha').value = '0.3'; fire('alpha', 'input'); console.log('pending text kept after slider move: ' + (mkEl('cfgText').value.indexOf('"start":[1,1]') >= 0 ? 'YES' : 'NO')); },
  37: () => { fire('applyCfg'); console.log('apply pending text: ' + msg()); },
  38: () => { fire('reset'); snap('boat reset button'); },
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
