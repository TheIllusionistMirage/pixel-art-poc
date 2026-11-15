import { buildLibrary } from './assets/library';
import { DETAIL } from './assets/voxel';
import { BirdAudio } from './audio/birdAudio';
import { WeatherAudio } from './audio/weatherAudio';
import { FLOATS_PER_INSTANCE, PITCH_MAX, PITCH_MIN, Renderer, pixelSnapper, writeInstances, type View } from './engine/renderer';
import { Birds } from './sim/birds';
import { Villagers } from './sim/villagers';
import { dayLighting, sunHeight } from './world/daycycle';
import { generateVillage } from './world/village';
import { Weather, WEATHER } from './world/weather';

const DEBUG_VIEWS = ['final', 'albedo', 'normals', 'height', 'sun shadow', 'light only'];

const params = new URLSearchParams(location.search);
const num = (k: string, d: number) => (params.has(k) ? Number(params.get(k)) : d);

const canvas = document.getElementById('view') as HTMLCanvasElement;
const hud = document.getElementById('hud')!;
const loading = document.getElementById('loading')!;

function fail(msg: string): never {
  loading.textContent = msg;
  throw new Error(msg);
}

const gl = canvas.getContext('webgl2', { antialias: false, alpha: false, preserveDrawingBuffer: params.has('capture') }) ?? fail('WebGL2 is not available in this browser.');

// Let the loading message paint before the (synchronous) generation work.
await new Promise((r) => requestAnimationFrame(() => setTimeout(r, 0)));

const t0 = performance.now();
const lib = buildLibrary(1);
const t1 = performance.now();
const seed = num('seed', 7);
const world = generateVillage(lib, seed);
const t2 = performance.now();
// Crowd density: the `villagers` param is the "normal" count; the pool is sized for the busiest level.
const CROWD = [
  { label: 'none', k: 0 },
  { label: 'sparse', k: 0.35 },
  { label: 'normal', k: 1 },
  { label: 'busy', k: 2.5 },
  { label: 'crowded', k: 5 },
];
const baseVillagers = Math.max(0, num('villagers', 180));
const villagers = new Villagers(world.roads, world.doors, Math.round(baseVillagers * CROWD[CROWD.length - 1].k), lib.villagers.length);
const birds = new Birds(world.birds, [world.size / 2, world.size / 2], seed);
const renderer = new Renderer(gl, lib.atlas, world.size, world.ground, world.statics, world.chimneys);
const t3 = performance.now();
loading.remove();
console.info(
  `assets ${(t1 - t0).toFixed(0)}ms (atlas ${lib.atlas.width}×${lib.atlas.height}), world ${(t2 - t1).toFixed(0)}ms, gpu upload ${(t3 - t2).toFixed(0)}ms`,
  world.counts,
);

// ── State ───────────────────────────────────────────────────────────────
const DEG = Math.PI / 180;
/** Default camera elevation in degrees (45 = classic sprite stacking; lower = more side-on). */
const DEFAULT_PITCH = 35;
const clampPitch = (p: number) => Math.max(PITCH_MIN, Math.min(PITCH_MAX, p));
const view: View = {
  x: num('x', world.size / 2),
  y: num('y', world.size / 2 + 20),
  angle: num('rot', 0),
  pitch: clampPitch(num('pitch', DEFAULT_PITCH) * DEG),
};
const state = {
  hour: num('t', 9),
  daySpeed: num('speed', 1 / 8), // game hours per real second (full day ≈ 3 min)
  paused: params.has('paused'),
  zoom: num('zoom', 0), // device pixels per art pixel; 0 = auto
  debug: num('debug', 0),
  shadows: !params.has('noshadows'),
  crowd: Math.max(0, Math.min(CROWD.length - 1, Math.round(num('crowd', 2)) || 0)),
};

/** Auto zoom tracks the window size until the user picks a zoom level themselves. */
let autoZoom = !state.zoom;
const setZoom = (z: number) => {
  autoZoom = false;
  state.zoom = Math.max(1, Math.min(8, z));
  resize();
};
function resize() {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.floor(canvas.clientWidth * dpr);
  canvas.height = Math.floor(canvas.clientHeight * dpr);
  // Auto zoom frames ~420 world units along the longer side, i.e. ~420·DETAIL art pixels
  // (the longer side, so a portrait phone gets the same pixel size as a landscape one).
  if (autoZoom) state.zoom = Math.max(1, Math.round(Math.max(canvas.width, canvas.height) / (420 * DETAIL)));
  renderer.setSize(canvas.width, canvas.height, state.zoom);
}
// ResizeObserver also catches mobile browser bars sliding in and out, which don't always fire `resize`.
new ResizeObserver(resize).observe(canvas);
window.addEventListener('resize', resize);
resize();

const isTouch = matchMedia('(pointer: coarse)').matches;

// ── Weather + sound ─────────────────────────────────────────────────────
const weather = new Weather();
if (params.has('weather')) weather.set(params.get('weather')!, true);
const audio = new WeatherAudio();
const birdAudio = new BirdAudio(audio);
const weatherSelect = document.getElementById('weather') as HTMLSelectElement;
const soundButton = document.getElementById('sound') as HTMLButtonElement;
const crowdButton = document.getElementById('crowd') as HTMLButtonElement;
for (const w of WEATHER) weatherSelect.add(new Option(w.label, w.id, w.id === weather.preset.id, w.id === weather.preset.id));
weatherSelect.addEventListener('change', () => {
  weather.set(weatherSelect.value);
  audio.unlock();
  weatherSelect.blur(); // give the keyboard back to the camera
});
const showSound = () => (soundButton.textContent = `sound: ${audio.isMuted ? 'off' : 'on'}`);
soundButton.addEventListener('click', () => {
  audio.unlock();
  audio.setMuted(!audio.isMuted);
  showSound();
  soundButton.blur();
});
showSound();
const setCrowd = (level: number) => {
  state.crowd = (level + CROWD.length) % CROWD.length;
  villagers.active = baseVillagers * CROWD[state.crowd].k;
  crowdButton.textContent = `crowd: ${CROWD[state.crowd].label}`;
};
crowdButton.addEventListener('click', () => {
  setCrowd(state.crowd + 1);
  crowdButton.blur();
});
setCrowd(state.crowd);
// Audio may only start after a user gesture. iOS only accepts some gesture events (touchend/click,
// not touchstart) and suspends the context again when the page is backgrounded, so keep listening;
// unlock() is cheap once the context is running.
for (const ev of ['pointerdown', 'pointerup', 'touchend', 'click', 'keydown']) window.addEventListener(ev, () => audio.unlock(), { passive: true });

// Touch-only buttons for actions that otherwise need a keyboard.
const pauseButton = document.getElementById('pause') as HTMLButtonElement;
const showPause = () => (pauseButton.textContent = state.paused ? 'play' : 'pause');
const togglePause = () => {
  state.paused = !state.paused;
  showPause();
};
showPause();
const resetView = () => Object.assign(view, { x: world.size / 2, y: world.size / 2 + 20, angle: 0, pitch: DEFAULT_PITCH * DEG });
const touchButtons: Record<string, () => void> = {
  'hour-back': () => (state.hour = (state.hour + 23) % 24),
  pause: togglePause,
  'hour-fwd': () => (state.hour = (state.hour + 1) % 24),
  reset: resetView,
};
for (const [id, action] of Object.entries(touchButtons)) {
  const b = document.getElementById(id) as HTMLButtonElement;
  b.addEventListener('click', () => {
    action();
    b.blur();
  });
}
// The HUD is long; on touch screens it starts collapsed and a tap toggles the details.
let hudExpanded = !isTouch;
hud.addEventListener('click', () => {
  hudExpanded = !hudExpanded;
  hudTimer = 0;
});
// Stop iOS page pinch-zoom (it ignores user-scalable=no) so pinches reach the canvas.
document.addEventListener('gesturestart', (e) => e.preventDefault());

// ── Input ───────────────────────────────────────────────────────────────
const keys = new Set<string>();
window.addEventListener('keydown', (e) => {
  if (e.target instanceof HTMLSelectElement || e.target instanceof HTMLButtonElement) return;
  keys.add(e.code);
  switch (e.code) {
    case 'Space': togglePause(); break;
    case 'BracketRight': state.daySpeed *= 2; break;
    case 'BracketLeft': state.daySpeed /= 2; break;
    case 'Period': state.hour = (state.hour + 1) % 24; break;
    case 'Comma': state.hour = (state.hour + 23) % 24; break;
    case 'KeyG': state.debug = (state.debug + 1) % DEBUG_VIEWS.length; break;
    case 'KeyH': state.shadows = !state.shadows; break;
    case 'KeyC': setCrowd(state.crowd + (e.shiftKey ? -1 : 1)); break;
    case 'Equal': setZoom(state.zoom + 1); break;
    case 'Minus': setZoom(state.zoom - 1); break;
    case 'KeyR': resetView(); break;
  }
});
window.addEventListener('keyup', (e) => keys.delete(e.code));
window.addEventListener('blur', () => keys.clear());

/** Move the camera by a screen-space delta (in world units along the screen axes). */
function panScreen(dx: number, dy: number) {
  const c = Math.cos(view.angle), s = Math.sin(view.angle);
  dy /= Math.tan(view.pitch); // ground is squashed vertically on screen
  view.x += c * dx - s * dy;
  view.y += s * dx + c * dy;
  view.x = Math.max(0, Math.min(world.size, view.x));
  view.y = Math.max(0, Math.min(world.size, view.y));
}

/** Rotate (horizontal) and tilt (vertical) by a CSS-pixel drag delta. */
function orbit(dx: number, dy: number) {
  view.angle += dx * 0.006;
  view.pitch = clampPitch(view.pitch - dy * 0.004); // drag up = camera higher
}
/** Pan so the ground follows a CSS-pixel drag delta. */
function dragPan(dx: number, dy: number) {
  const k = (window.devicePixelRatio || 1) / state.zoom / DETAIL;
  panScreen(-dx * k, -dy * k);
}

// Pointers currently down on the canvas. Mouse: left-drag pans, right/shift-drag orbits.
// Touch: one finger pans; two fingers pinch to zoom, twist to rotate and drag up/down to tilt.
const pointers = new Map<number, { x: number; y: number; orbit: boolean }>();
/** Two-finger gesture: the finger span at the last zoom step, and last frame's twist angle and midpoint. */
let pinch: { span0: number; angle: number; midY: number } | null = null;
function twoFingers() {
  const [a, b] = [...pointers.values()];
  return { span: Math.hypot(b.x - a.x, b.y - a.y), angle: Math.atan2(b.y - a.y, b.x - a.x), midY: (a.y + b.y) / 2 };
}
function startPinch() {
  if (pointers.size !== 2) return void (pinch = null);
  const g = twoFingers();
  pinch = { span0: g.span, angle: g.angle, midY: g.midY };
}

canvas.addEventListener('contextmenu', (e) => e.preventDefault());
canvas.addEventListener('pointerdown', (e) => {
  pointers.set(e.pointerId, { x: e.clientX, y: e.clientY, orbit: e.pointerType === 'mouse' && (e.shiftKey || e.button === 2) });
  canvas.setPointerCapture(e.pointerId);
  startPinch();
});
const pointerEnd = (e: PointerEvent) => {
  pointers.delete(e.pointerId);
  startPinch(); // two fingers left (e.g. a third one lifted): restart cleanly; otherwise stop
};
canvas.addEventListener('pointerup', pointerEnd);
canvas.addEventListener('pointercancel', pointerEnd);
canvas.addEventListener('pointermove', (e) => {
  const p = pointers.get(e.pointerId);
  if (!p) return;
  const dx = e.clientX - p.x, dy = e.clientY - p.y;
  p.x = e.clientX;
  p.y = e.clientY;
  if (pointers.size === 1) {
    if (p.orbit) orbit(dx, dy);
    else dragPan(dx, dy);
    return;
  }
  if (!pinch) return;
  const g = twoFingers();
  // Twist: turn the world with the fingers. Screen angles grow clockwise; view.angle turns the world anticlockwise.
  let da = g.angle - pinch.angle;
  da -= Math.round(da / (2 * Math.PI)) * 2 * Math.PI;
  view.angle -= da;
  pinch.angle = g.angle;
  // Both fingers moving up or down together: tilt.
  view.pitch = clampPitch(view.pitch - (g.midY - pinch.midY) * 0.004);
  pinch.midY = g.midY;
  // Pinch: zoom is in whole steps, so step each time the span changes by ~30%.
  const ratio = g.span / pinch.span0;
  if (ratio > 1.3 || ratio < 1 / 1.3) {
    setZoom(state.zoom + (ratio > 1 ? 1 : -1));
    pinch.span0 = g.span;
  }
});
canvas.addEventListener('wheel', (e) => {
  e.preventDefault();
  setZoom(state.zoom + (e.deltaY < 0 ? 1 : -1));
}, { passive: false });

// ── Loop ────────────────────────────────────────────────────────────────
const villagerSlices = Math.max(...lib.villagers.flat().map((n) => lib.atlas.byName.get(n)!.slices.length));
const rookModels = lib.rooks.map((n) => lib.atlas.byName.get(n)!);
const rookSlices = Math.max(...rookModels.map((m) => m.slices.length));
let dynData = new Float32Array(1024 * FLOATS_PER_INSTANCE);
let last = performance.now();
let time = 0;
let fpsAcc = 0, fpsFrames = 0, fps = 0, cpuMs = 0, hudTimer = 0;

function frame(now: number) {
  const dt = Math.min(0.1, (now - last) / 1000);
  last = now;
  time += dt;
  const cpuStart = performance.now();

  const pan = 220 * dt; // world units per second
  if (keys.has('KeyW') || keys.has('ArrowUp')) panScreen(0, -pan);
  if (keys.has('KeyS') || keys.has('ArrowDown')) panScreen(0, pan);
  if (keys.has('KeyA') || keys.has('ArrowLeft')) panScreen(-pan, 0);
  if (keys.has('KeyD') || keys.has('ArrowRight')) panScreen(pan, 0);
  if (keys.has('KeyQ')) view.angle -= 1.4 * dt;
  if (keys.has('KeyE')) view.angle += 1.4 * dt;
  if (keys.has('KeyZ')) view.pitch = clampPitch(view.pitch - 0.6 * dt);
  if (keys.has('KeyX')) view.pitch = clampPitch(view.pitch + 0.6 * dt);
  if (!state.paused) state.hour = (state.hour + state.daySpeed * dt) % 24;

  weather.update(dt, time, view.x, view.y);
  const strikes = weather.takeStrikes();
  for (const s of strikes) audio.thunder(s);
  const W = weather.cur;
  villagers.update(dt, W.shelter, Math.max(W.rain, W.snow * W.wind));
  birds.update(dt, { hour: state.hour, sun: sunHeight(state.hour), shelter: W.shelter, wind: weather.wind, time, strikes });
  birdAudio.play(birds.takeSounds(), view);

  // Moving entities → instance data (culled to the heightmap window, which contains the screen).
  const hmb = renderer.heightmapBounds(view);
  const need = (villagers.active * villagerSlices + birds.count * rookSlices) * FLOATS_PER_INSTANCE;
  if (dynData.length < need) dynData = new Float32Array(need);
  const snap = pixelSnapper(view);
  let o = 0;
  for (let i = 0; i < villagers.active; i++) {
    if (villagers.hidden[i]) continue;
    const x = villagers.x[i], y = villagers.y[i];
    if (x < hmb.minX || y < hmb.minY || x > hmb.maxX || y > hmb.maxY) continue;
    const m = lib.atlas.byName.get(lib.villagers[villagers.look[i]][villagers.frame[i]])!;
    // Whole-pixel steps: sub-pixel positions make slice rows flicker as they walk.
    const p = snap(x, y);
    o = writeInstances(dynData, o, m, p.x, p.y, villagers.angle[i]);
  }
  // Birds: the ones on the ground or perched first, then everything in the air
  // (the renderer puts the last `flyers` instances in the heightmap's airborne layer).
  const lift = 1 / Math.tan(view.pitch);
  let flyerStart = o;
  for (let pass = 0; pass < 2; pass++) {
    if (pass === 1) flyerStart = o;
    for (let i = 0; i < birds.count; i++) {
      if (!birds.visible(i) || birds.airborne(i) !== (pass === 1)) continue;
      const x = birds.x[i], y = birds.y[i];
      // Something high up is drawn well above its ground point: widen the cull by that much.
      const mz = birds.z[i] * lift + 4;
      if (x < hmb.minX - mz || y < hmb.minY - mz || x > hmb.maxX + mz || y > hmb.maxY + mz) continue;
      const p = snap(x, y);
      o = writeInstances(dynData, o, rookModels[birds.frame[i]], p.x, p.y, birds.angle[i], Math.round(birds.z[i] * DETAIL));
    }
  }
  const flyers = (o - flyerStart) / FLOATS_PER_INSTANCE;

  const lighting = weather.light(dayLighting(state.hour));
  audio.update({ rain: W.rain, snow: W.snow, wind: W.wind, gust: weather.gust });
  const instances = o / FLOATS_PER_INSTANCE;
  const stats = renderer.render({
    view,
    lighting,
    lights: world.lights,
    time,
    debug: state.debug,
    shadows: state.shadows,
    dynamic: { data: dynData, count: instances, flyers },
    weather: weather.frame(lighting),
  });
  cpuMs = cpuMs * 0.9 + (performance.now() - cpuStart) * 0.1;

  fpsAcc += dt;
  fpsFrames++;
  if (fpsAcc >= 0.5) {
    fps = fpsFrames / fpsAcc;
    fpsAcc = 0;
    fpsFrames = 0;
  }
  hudTimer -= dt;
  if (hudTimer <= 0) {
    hudTimer = 0.2;
    const hh = Math.floor(state.hour), mm = Math.floor((state.hour % 1) * 60);
    const [rw, rh] = renderer.internalResolution;
    const summary = [
      `${String(hh).padStart(2, '0')}:${String(mm).padStart(2, '0')}  ${lighting.phase}${state.paused ? '  (paused)' : `  ×${(state.daySpeed * 8).toFixed(2)}`}`,
      `${fps.toFixed(0)} fps   cpu ${cpuMs.toFixed(1)} ms   ${rw}×${rh} @${state.zoom}x`,
    ];
    if (!hudExpanded) {
      hud.textContent = [...summary, `tap for details`].join('\n');
    }
    else hud.textContent = [
      ...summary,
      `objects ${world.statics.length}   villagers ${villagers.active} (${CROWD[state.crowd].label})   lights ${stats.lights}`,
      `birds ${birds.count} in ${birds.flockCount} flocks (${birds.flying} flying${birds.hiddenCount ? `, ${birds.hiddenCount} sheltering` : ''})`,
      `slices drawn ${stats.instances.toLocaleString()} / ${(renderer.staticInstanceCount + instances).toLocaleString()}   draws ${stats.drawCalls}`,
      `weather ${weather.preset.label}${weather.transitioning ? ' (changing)' : ''}   wind ${Math.hypot(...weather.wind).toFixed(2)}   wet ${(weather.wetness * 100).toFixed(0)}%   snow ${(weather.snowCover * 100).toFixed(0)}%   indoors ${villagers.indoors()}`,
      `view: ${DEBUG_VIEWS[state.debug]}${state.shadows ? '' : '  (shadows off)'}   pitch ${(view.pitch / DEG).toFixed(0)}°`,
      ``,
      ...(isTouch
        ? [`drag pan · pinch zoom`, `two-finger twist rotate · two-finger drag tilt`, `tap here to collapse`]
        : [
          `drag pan · shift/right-drag rotate & tilt · wheel zoom`,
          `WASD pan · Q/E rotate · Z/X tilt · , . hour · [ ] speed`,
          `space pause · G debug view · H shadows · C crowd · R reset`,
        ]),
    ].join('\n');
  }
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
