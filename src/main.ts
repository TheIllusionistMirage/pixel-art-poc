import { buildLibrary } from './assets/library';
import { DETAIL } from './assets/voxel';
import { WeatherAudio } from './audio/weatherAudio';
import { FLOATS_PER_INSTANCE, PITCH_MAX, PITCH_MIN, Renderer, pixelSnapper, writeInstances, type View } from './engine/renderer';
import { Villagers } from './sim/villagers';
import { dayLighting } from './world/daycycle';
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
const world = generateVillage(lib, num('seed', 7));
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

function resize() {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.floor(canvas.clientWidth * dpr);
  canvas.height = Math.floor(canvas.clientHeight * dpr);
  // Auto zoom frames ~420 world units across, i.e. ~420·DETAIL art pixels.
  if (!state.zoom) state.zoom = Math.max(1, Math.round(canvas.width / (420 * DETAIL)));
  renderer.setSize(canvas.width, canvas.height, state.zoom);
}
window.addEventListener('resize', resize);
resize();

// ── Weather + sound ─────────────────────────────────────────────────────
const weather = new Weather();
if (params.has('weather')) weather.set(params.get('weather')!, true);
const audio = new WeatherAudio();
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
// Audio may only start after a user gesture.
window.addEventListener('pointerdown', () => audio.unlock(), { once: true });
window.addEventListener('keydown', () => audio.unlock(), { once: true });

// ── Input ───────────────────────────────────────────────────────────────
const keys = new Set<string>();
window.addEventListener('keydown', (e) => {
  if (e.target instanceof HTMLSelectElement || e.target instanceof HTMLButtonElement) return;
  keys.add(e.code);
  switch (e.code) {
    case 'Space': state.paused = !state.paused; break;
    case 'BracketRight': state.daySpeed *= 2; break;
    case 'BracketLeft': state.daySpeed /= 2; break;
    case 'Period': state.hour = (state.hour + 1) % 24; break;
    case 'Comma': state.hour = (state.hour + 23) % 24; break;
    case 'KeyG': state.debug = (state.debug + 1) % DEBUG_VIEWS.length; break;
    case 'KeyH': state.shadows = !state.shadows; break;
    case 'KeyC': setCrowd(state.crowd + (e.shiftKey ? -1 : 1)); break;
    case 'Equal': state.zoom = Math.min(8, state.zoom + 1); resize(); break;
    case 'Minus': state.zoom = Math.max(1, state.zoom - 1); resize(); break;
    case 'KeyR': Object.assign(view, { x: world.size / 2, y: world.size / 2 + 20, angle: 0, pitch: DEFAULT_PITCH * DEG }); break;
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

let drag: { button: number; x: number; y: number } | null = null;
canvas.addEventListener('contextmenu', (e) => e.preventDefault());
canvas.addEventListener('pointerdown', (e) => {
  drag = { button: e.shiftKey ? 2 : e.button, x: e.clientX, y: e.clientY };
  canvas.setPointerCapture(e.pointerId);
});
canvas.addEventListener('pointerup', () => (drag = null));
canvas.addEventListener('pointermove', (e) => {
  if (!drag) return;
  const dpr = window.devicePixelRatio || 1;
  const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
  drag.x = e.clientX;
  drag.y = e.clientY;
  if (drag.button === 2) {
    view.angle += dx * 0.006;
    view.pitch = clampPitch(view.pitch - dy * 0.004); // drag up = camera higher
  }
  else panScreen((-dx * dpr) / state.zoom / DETAIL, (-dy * dpr) / state.zoom / DETAIL);
});
canvas.addEventListener('wheel', (e) => {
  e.preventDefault();
  state.zoom = Math.max(1, Math.min(8, state.zoom + (e.deltaY < 0 ? 1 : -1)));
  resize();
}, { passive: false });

// ── Loop ────────────────────────────────────────────────────────────────
const villagerSlices = Math.max(...lib.villagers.flat().map((n) => lib.atlas.byName.get(n)!.slices.length));
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
  for (const s of weather.takeStrikes()) audio.thunder(s);
  const W = weather.cur;
  villagers.update(dt, W.shelter, Math.max(W.rain, W.snow * W.wind));

  // Moving entities → instance data (culled to the heightmap window, which contains the screen).
  const hmb = renderer.heightmapBounds(view);
  const need = villagers.active * villagerSlices * FLOATS_PER_INSTANCE;
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
    dynamic: { data: dynData, count: instances },
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
    hud.textContent = [
      `${String(hh).padStart(2, '0')}:${String(mm).padStart(2, '0')}  ${lighting.phase}${state.paused ? '  (paused)' : `  ×${(state.daySpeed * 8).toFixed(2)}`}`,
      `${fps.toFixed(0)} fps   cpu ${cpuMs.toFixed(1)} ms   ${rw}×${rh} @${state.zoom}x`,
      `objects ${world.statics.length}   villagers ${villagers.active} (${CROWD[state.crowd].label})   lights ${stats.lights}`,
      `slices drawn ${stats.instances.toLocaleString()} / ${(renderer.staticInstanceCount + instances).toLocaleString()}   draws ${stats.drawCalls}`,
      `weather ${weather.preset.label}${weather.transitioning ? ' (changing)' : ''}   wind ${Math.hypot(...weather.wind).toFixed(2)}   wet ${(weather.wetness * 100).toFixed(0)}%   snow ${(weather.snowCover * 100).toFixed(0)}%   indoors ${villagers.indoors()}`,
      `view: ${DEBUG_VIEWS[state.debug]}${state.shadows ? '' : '  (shadows off)'}   pitch ${(view.pitch / DEG).toFixed(0)}°`,
      ``,
      `drag pan · shift/right-drag rotate & tilt · wheel zoom`,
      `WASD pan · Q/E rotate · Z/X tilt · , . hour · [ ] speed`,
      `space pause · G debug view · H shadows · C crowd · R reset`,
    ].join('\n');
  }
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
