// Headless check with event capture: runs the page script with a stubbed DOM and fires its
// listeners on a schedule. Covers apply, rejected inputs (syntax, unknown keys, bounds, the
// old "proj" key, route validation and the fan-in budget), the motor-fit refusal, a route
// edit that changes the wiring, preset, reset, moved target/start, seed clamp, and that
// unapplied JSON text survives a slider move.
const fs = require('fs');
const vm = require('vm');
const src = fs.readFileSync(process.argv[2], 'utf8');
const seconds = parseInt(process.argv[3] || '50', 10);

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
const reject = (label, obj, expect) => { setJSON(obj); fire('applyCfg'); const ok = expect === undefined ? '' : (msg().includes(expect) ? ' [OK]' : ' [MISMATCH: expected "' + expect + '"]'); console.log(label + ': ' + msg() + ok); };

// default routes without the VIP->SST route (weight edit test: removing one connection)
const R = (name, from, to, k, w, fb) => Object.assign({ name, from, to, k, w }, fb !== undefined ? { fallbackLayer: fb } : {});
const DEF_ROUTES = [
  R('EE', { area: 0, layer: 'same', type: 0 }, { layer: '*', type: 0 }, 6, 0.01),
  R('PVE', { area: 0, layer: 'same', type: 1 }, { layer: '*', type: 0 }, 4, -0.05),
  R('E L5 -> E L2', { area: -1, layer: 4, type: 0 }, { layer: 1, type: 0 }, 6, 0.02),
  R('E L5 -> E L6', { area: 1, layer: 4, type: 0 }, { layer: 5, type: 0 }, 6, 0.02),
  R('EPV', { area: 0, layer: 'same', type: 0 }, { layer: '*', type: 1 }, 18, 0.04, 2),
  R('SSTPV', { area: 0, layer: 'same', type: 2 }, { layer: '*', type: 1 }, 4, -0.03),
  R('ESST', { area: 0, layer: 'same', type: 0 }, { layer: '*', type: 2 }, 18, 0.02, 2),
  R('VIPSST', { area: 0, layer: 'same', type: 3 }, { layer: '*', type: 2 }, 4, -0.05),
  R('EVIP', { area: 0, layer: 'same', type: 0 }, { layer: '*', type: 3 }, 12, 0.02, 2),
  R('E L5 -> VIP L1', { area: 1, layer: 4, type: 0 }, { layer: 0, type: 3 }, 6, 0.02),
];
const withoutVIPSST = DEF_ROUTES.filter(r => r.name !== 'VIPSST');

const actions = {
  5: () => snap('defaults, 5 s'),
  8: () => { setJSON({ alpha: 0, cue: 0 }); fire('applyCfg'); console.log('apply alpha 0, cue 0: ' + msg()); },
  12: () => snap('alpha 0 cue 0, 12 s (expect drift, e about 3)'),
  13: () => reject('invalid JSON', '{ "alpha": 0,'),
  14: () => reject('unknown key', { foo: 1 }),
  15: () => reject('old "proj" key', { proj: { EE: { k: 6 } } }),
  16: () => reject('non-numeric alpha', { alpha: 'x' }),
  17: () => reject('start 1e308', { start: [1e308, 0] }),
  18: () => reject('layerFrac row sums to 1000', { layerFrac: [[0, 1000, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]] }),
  19: () => reject('damping 5000', { damping: 5000 }),
  20: () => reject('route k 1000', { routes: [R('x', { area: 0, layer: 'same', type: 0 }, { layer: '*', type: 0 }, 1000, 0.01)] }),
  21: () => reject('route to.layer 6', { routes: [R('x', { area: 0, layer: 'same', type: 0 }, { layer: 6, type: 0 }, 6, 0.01)] }),
  22: () => reject('route from.area 2', { routes: [{ from: { area: 2, layer: 'same', type: 0 }, to: { layer: '*', type: 0 }, k: 6, w: 0.01 }] }),
  23: () => reject('route with extra field', { routes: [Object.assign(R('x', { area: 0, layer: 'same', type: 0 }, { layer: '*', type: 0 }, 6, 0.01), { bogus: 1 })] }),
  24: () => reject('fan-in budget 4 x 100 = 400', { routes: [0, 1, 2, 3].map(i => R('b' + i, { area: 0, layer: 'same', type: 0 }, { layer: '*', type: 0 }, 100, 0.01)) }, 'fan-in into type 0 in layer 1 is 400'),
  25: () => snap('after 9 rejected inputs (unchanged, cells 480)'),
  45: () => { const four = (t) => [0, 1, 2, 3].map(i => R('p' + i, { area: 0, layer: 'same', type: 0 }, { layer: '*', type: i }, 100, 0.01)); reject('POSITIVE per-target: four k=100 routes, one per target type (total 400, valid)', { routes: four() }, 'Applied.'); },
  48: () => reject('layerOffset 1.5 (out of range)', { layerOffset: [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,1.5],[0,0,0,0]] }),
  49: () => { setJSON({ layerOffset: [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0.05,0,0],[0,0,0,0]] }); fire('applyCfg'); console.log('layerOffset L5 PV +0.05: ' + msg()); },
  50: () => snap('L5 PV drive +0.05 (expect PV rate in area 1 to move)'),
  51: () => { fire('resetCfg'); console.log('reset: ' + msg()); },
  47: () => snap('after positive case (expect 480 cells, unchanged by routes)'),
  26: () => { setJSON({ routes: withoutVIPSST }); fire('applyCfg'); console.log('remove VIP->SST route: ' + msg()); },
  28: () => snap('VIP->SST removed (expect cells 480, state rebuilt)'),
  29: () => reject('120 neurons with motorPerDir 5 (expect motor refusal)', { neuronsPerColumn: 120, areas: 1 }),
  30: () => { setJSON({ neuronsPerColumn: 120, areas: 1, motorPerDir: 3, routes: DEF_ROUTES }); fire('applyCfg'); console.log('120 neurons, 1 area, motorPerDir 3: ' + msg()); },
  32: () => snap('120/1 network (expect cells 120)'),
  33: () => { fire('resetCfg'); console.log('reset: ' + msg()); },
  34: () => snap('after reset (expect 480 cells)'),
  35: () => { firePreset('cue'); console.log('cue preset clicked'); },
  37: () => snap('cue preset (alpha 0, cue 0.20)'),
  38: () => { setJSON({ target: [1, 0.5], start: [-2, 0], fixWindow: 300 }); fire('applyCfg'); console.log('moved target/start, fixWindow 300: ' + msg()); },
  39: () => snap('moved target (expect label 6 s window)'),
  40: () => { mkEl('seed').value = '99999999'; fire('seed', 'change'); console.log('seed typed 99999999 -> JSON seed ' + (mkEl('cfgText').value.match(/"seed": (\d+)/) || [])[1]); },
  41: () => { fire('applyCfg'); console.log('apply after seed clamp: ' + msg()); },
  42: () => { fire('resetCfg'); setJSON('{"start":[1,1]}'); fire('cfgText', 'input'); mkEl('alpha').value = '0.3'; fire('alpha', 'input'); console.log('pending text kept after slider move: ' + (mkEl('cfgText').value.indexOf('"start":[1,1]') >= 0 ? 'YES' : 'NO')); },
  43: () => { fire('applyCfg'); console.log('apply pending text: ' + msg()); },
  44: () => { fire('reset'); snap('boat reset button'); },
};

const t0 = Date.now();
const fired = new Set();
vm.runInThisContext(src);
const FR = 1000 / 60;
const errors = [];
for (let f = 1; f <= seconds * 60; f++) {
  now += FR;
  const sec = Math.round(f / 60);
  if (f % 60 === 0 && actions[sec]) { actions[sec](); fired.add(sec); }
  const cbs = frames; frames = [];
  for (const cb of cbs) {
    try { cb(now); } catch (e) { errors.push(String(e && e.stack || e).slice(0, 300)); }
  }
  if (errors.length > 2) break;
}
const missing = Object.keys(actions).map(Number).filter(k => k <= seconds && !fired.has(k));
console.log(missing.length ? 'MISSING actions (not run): ' + missing.join(', ') : 'all scheduled actions ran');
console.log(errors.length ? 'ERRORS: ' + errors.join(' | ') : 'no runtime errors');
console.log('wall ' + ((Date.now() - t0) / 1000).toFixed(1) + ' s');
