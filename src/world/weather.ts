// Weather: a handful of presets, each a set of intensities in [0, 1]. The live
// state eases towards the selected preset (a crossfade of a few seconds), and a
// few slower quantities integrate on top of it: how wet surfaces are, how much
// snow has settled, the wind (direction + gusts) and lightning strikes.
// Nothing here touches the GPU; `frame()` packs what the renderer needs and
// `light()` folds the weather into the day/night lighting.

import { clamp, lerp, mulberry32, smoothstep } from '../engine/rng';
import type { DayLighting, Vec3 } from './daycycle';

export interface WeatherParams {
  /** Cloud cover: 0 = clear sky, 1 = solid overcast (no direct sun/moon). */
  cloud: number;
  rain: number;
  snow: number;
  fog: number;
  wind: number;
  /** Thunderstorm activity: strike rate and darkness. */
  lightning: number;
  /** Share of villagers that go indoors. */
  shelter: number;
}

export interface WeatherPreset {
  id: string;
  label: string;
  p: WeatherParams;
}

const P = (cloud: number, rain: number, snow: number, fog: number, wind: number, lightning: number, shelter: number): WeatherParams =>
  ({ cloud, rain, snow, fog, wind, lightning, shelter });

//                                                      cloud rain snow fog  wind  bolt shelter
export const WEATHER: WeatherPreset[] = [
  { id: 'clear', label: 'Clear', p: P(0, 0, 0, 0, 0.12, 0, 0) },
  { id: 'cloudy', label: 'Cloudy', p: P(0.5, 0, 0, 0.02, 0.24, 0, 0) },
  { id: 'overcast', label: 'Overcast', p: P(0.78, 0, 0, 0.05, 0.3, 0, 0.05) },
  { id: 'fog', label: 'Fog', p: P(0.6, 0, 0, 0.85, 0.04, 0, 0.1) },
  { id: 'lightrain', label: 'Light rain', p: P(0.75, 0.35, 0, 0.06, 0.22, 0, 0.3) },
  { id: 'rain', label: 'Rain', p: P(0.9, 0.72, 0, 0.1, 0.42, 0, 0.65) },
  { id: 'storm', label: 'Thunderstorm', p: P(1, 1, 0, 0.14, 0.85, 1, 0.95) },
  { id: 'lightsnow', label: 'Light snow', p: P(0.7, 0, 0.35, 0.1, 0.12, 0, 0.15) },
  { id: 'snow', label: 'Snow', p: P(0.85, 0, 0.72, 0.22, 0.3, 0, 0.45) },
  { id: 'blizzard', label: 'Blizzard', p: P(1, 0, 1, 0.62, 1, 0, 0.92) },
];

/** Time constant of the crossfade between presets, seconds (~95% after 3τ). */
const FADE_TAU = 2.2;
/** How far rain and snow are carried sideways per unit of wind per second. */
export const RAIN_CARRY = 45;
export const SNOW_CARRY = 26;

/** Lifetime of each rain-sheet layer, seconds (two layers, half a cycle apart). */
const MIST_CYCLE = 10;

/** Prevailing wind direction (radians): wanders slowly, roughly from the west. */
const windAngle = (time: number): number => 0.35 + 0.5 * Math.sin(time * 0.031) + 0.25 * Math.sin(time * 0.083 + 1.3);

/**
 * Rain-sheet pattern layer. Its offset integrates the wind velocity (so the
 * pattern only ever moves forwards), and its stretch direction is frozen for
 * its lifetime and only picks up the current wind while its weight is zero,
 * so a turning wind never swings the pattern around.
 */
interface MistLayer {
  dir: [number, number];
  off: [number, number];
  ph: number;
}

/** How long a lightning flash lasts (bolt visible while bright), seconds. */
const FLASH_LEN = 0.6;

export interface Strike {
  /** Distance from the camera, world units (drives flash strength and thunder delay). */
  distance: number;
  strength: number;
  /** Where it hit, world units (birds nearby scatter). */
  x: number;
  y: number;
}

/** Everything the renderer needs from the weather for one frame. */
export interface WeatherFrame {
  cloud: number;
  rain: number;
  snow: number;
  fog: number;
  /** Rain mist: spray off the ground and blown sheets of rain, from the rain intensity. */
  mist: number;
  wetness: number;
  snowCover: number;
  /** Wind vector (direction × strength × gust), ~[0, 1.3] long. */
  wind: [number, number];
  /** Rain-sheet layers: (dir x, dir y, offset x, offset y) each, and their weights. */
  mistA: [number, number, number, number];
  mistB: [number, number, number, number];
  mistW: [number, number];
  /** Accumulated sideways drift of falling rain / snow (wind integrated over time), world units. */
  rainDrift: [number, number];
  snowDrift: [number, number];
  /** Accumulated cloud drift, world units. */
  cloudOffset: [number, number];
  smoke: number;
  flash: number;
  /** Lightning bolt as world-space line pairs (x, y, z, x, y, z, …), or null. */
  bolt: Float32Array | null;
  fogColor: Vec3;
}

export class Weather {
  preset: WeatherPreset = WEATHER[0];
  readonly cur: WeatherParams = { ...WEATHER[0].p };
  wetness = 0;
  snowCover = 0;
  wind: [number, number] = [0, 0];
  gust = 1;
  private cloudOffset: [number, number] = [0, 0];
  private rainDrift: [number, number] = [0, 0];
  private snowDrift: [number, number] = [0, 0];
  private readonly mist: MistLayer[] = [0, 0.5].map((ph) => ({ dir: [Math.cos(windAngle(0)), Math.sin(windAngle(0))], off: [0, 0], ph }));
  private flashT = Infinity;
  private flashAmp = 0;
  private pulses: [number, number][] = [];
  private bolt: Float32Array | null = null;
  private readonly rng = mulberry32(911);
  /** Strikes since the last call to `takeStrikes` (for thunder). */
  private strikes: Strike[] = [];

  /** Select a preset; `instant` skips the crossfade (for starting in it). */
  set(id: string, instant = false): void {
    this.preset = WEATHER.find((w) => w.id === id) ?? this.preset;
    if (instant) Object.assign(this.cur, this.preset.p);
  }

  /** True while the live state is still easing towards the selected preset. */
  get transitioning(): boolean {
    const t = this.preset.p;
    return (Object.keys(t) as (keyof WeatherParams)[]).some((k) => Math.abs(t[k] - this.cur[k]) > 0.02);
  }

  takeStrikes(): Strike[] {
    const s = this.strikes;
    this.strikes = [];
    return s;
  }

  /** dt and time in real seconds; (cx, cy) = camera position, where strikes happen around. */
  update(dt: number, time: number, cx: number, cy: number): void {
    const c = this.cur, t = this.preset.p;
    const f = 1 - Math.exp(-dt / FADE_TAU);
    for (const k of Object.keys(c) as (keyof WeatherParams)[]) c[k] += (t[k] - c[k]) * f;

    // Wind: a slowly wandering direction (roughly from the west) and gusts that get
    // more violent as the wind picks up.
    const ang = windAngle(time);
    const gv = 0.18 + 0.35 * c.wind;
    this.gust = 1 + gv * (0.6 * Math.sin(time * 0.53) + 0.3 * Math.sin(time * 1.37 + 2.1) + 0.25 * Math.sin(time * 3.1 + 0.4));
    const ws = c.wind * this.gust;
    this.wind = [Math.cos(ang) * ws, Math.sin(ang) * ws];
    // Clouds drift with the wind, plus a little even on a still day.
    this.cloudOffset[0] -= (this.wind[0] * 26 + 2.5) * dt;
    this.cloudOffset[1] -= (this.wind[1] * 26 + 1.2) * dt;
    // Falling rain and snow are carried by the wind as it is now: integrating it
    // (instead of wind × fall time per particle) keeps a gust from shoving the
    // whole field sideways at once.
    this.rainDrift[0] += this.wind[0] * RAIN_CARRY * dt;
    this.rainDrift[1] += this.wind[1] * RAIN_CARRY * dt;
    this.snowDrift[0] += this.wind[0] * SNOW_CARRY * dt;
    this.snowDrift[1] += this.wind[1] * SNOW_CARRY * dt;
    // Rain sheets travel downwind: a base speed plus the (gusting) wind.
    const vx = Math.cos(ang) * 14 + this.wind[0] * 50, vy = Math.sin(ang) * 14 + this.wind[1] * 50;
    for (const m of this.mist) {
      m.off[0] += vx * dt;
      m.off[1] += vy * dt;
      m.ph += dt / MIST_CYCLE;
      if (m.ph >= 1) {
        m.ph -= Math.floor(m.ph);
        m.dir = [Math.cos(ang), Math.sin(ang)];
      }
    }

    // Surfaces: rain soaks them in a few seconds, they dry over about a minute.
    // Snow settles while it falls and melts (into wetness) once it stops; rain melts it faster.
    if (c.rain > 0.02) this.wetness += dt * c.rain * 0.14;
    else this.wetness -= dt * 0.016;
    if (c.snow > 0.02) {
      const cap = 0.55 + 0.45 * c.snow;
      if (this.snowCover < cap) this.snowCover = Math.min(cap, this.snowCover + dt * c.snow * 0.04);
    }
    const melt = this.snowCover > 0 ? (c.snow < 0.02 ? 0.018 : 0) + c.rain * 0.09 : 0;
    if (melt > 0) {
      const m = Math.min(this.snowCover, melt * dt);
      this.snowCover -= m;
      this.wetness += m * 0.9;
    }
    this.wetness = clamp(this.wetness, 0, 1);
    this.snowCover = clamp(this.snowCover, 0, 1);

    // Lightning: a Poisson process, roughly one strike every 6 s in a full storm.
    this.flashT += dt;
    if (c.lightning > 0.3 && this.rng() < dt * c.lightning * 0.17) this.strike(cx, cy);
    if (this.flashT > FLASH_LEN) this.bolt = null;
  }

  private strike(cx: number, cy: number): void {
    const r = this.rng;
    // Mostly away from the camera; now and then one lands on screen.
    const d = r() < 0.3 ? 20 + r() * 160 : 200 + r() * 500;
    const a = r() * Math.PI * 2;
    const x = cx + Math.cos(a) * d, y = cy + Math.sin(a) * d;
    const strength = clamp(1.15 - d / 700, 0.25, 1);
    this.flashT = 0;
    this.flashAmp = strength;
    // A main flash and a couple of re-strokes.
    this.pulses = [[0, 1], [0.08 + r() * 0.06, 0.45 + r() * 0.3]];
    if (r() < 0.6) this.pulses.push([0.22 + r() * 0.12, 0.5 + r() * 0.4]);
    this.bolt = d < 240 ? makeBolt(x, y, r) : null;
    this.strikes.push({ distance: d, strength, x, y });
  }

  private flash(): number {
    let v = 0;
    for (const [t0, amp] of this.pulses) if (this.flashT >= t0) v = Math.max(v, amp * Math.exp(-(this.flashT - t0) / 0.055));
    return v * this.flashAmp;
  }

  /** Fold the weather into the day/night lighting. */
  light(day: DayLighting): DayLighting {
    const c = this.cur;
    const flash = this.flash();
    const gray = (v: Vec3, tint: Vec3): Vec3 => {
      const l = 0.3 * v[0] + 0.59 * v[1] + 0.11 * v[2];
      return [l * tint[0], l * tint[1], l * tint[2]];
    };
    const mix3 = (a: Vec3, b: Vec3, t: number): Vec3 => [lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t)];
    // Under cloud, direct light is scattered into a flat, slightly cool ambient.
    // Rain, storms and blizzards also take light away overall.
    const dark = clamp(c.cloud * 0.12 + c.rain * 0.2 + c.lightning * 0.16 + c.snow * c.wind * 0.12 + c.fog * 0.06, 0, 0.6);
    let ambient = mix3(day.ambient, gray(day.ambient, [0.97, 1, 1.06]), 0.6 * c.cloud);
    ambient = ambient.map((v, i) => (v + day.lightColor[i] * 0.42 * c.cloud) * (1 - dark) + 0.07 * this.snowCover * (1 - day.night)) as Vec3;
    let sky = mix3(day.sky, gray(day.sky, [0.95, 1, 1.05]), 0.75 * c.cloud).map((v) => v * (1 - dark)) as Vec3;
    const fl: Vec3 = [0.8 * flash, 0.84 * flash, 1.05 * flash];
    ambient = ambient.map((v, i) => v + fl[i]) as Vec3;
    sky = sky.map((v, i) => v + fl[i]) as Vec3;
    // Gloomy weather brings the lamps and windows on early.
    const night = Math.max(day.night, clamp(dark * 1.1 + c.fog * 0.25, 0, 0.75));
    return { ...day, ambient, sky, night };
  }

  frame(lit: DayLighting): WeatherFrame {
    const c = this.cur;
    // Fog takes the colour of the light around it: grey-white by day, deep blue at night.
    const base = lit.ambient.map((v, i) => v * 1.08 + lit.lightColor[i] * 0.2 * (1 - c.cloud)) as Vec3;
    const l = 0.3 * base[0] + 0.59 * base[1] + 0.11 * base[2];
    const fogColor = base.map((v, i) => lerp(v, l * [0.95, 1, 1.06][i], 0.45) + 0.04 * this.snowCover) as Vec3;
    const flashT = this.flashT;
    const boltOn = this.bolt && this.pulses.some(([t0]) => flashT >= t0 && flashT - t0 < 0.07);
    return {
      cloud: c.cloud,
      rain: c.rain,
      snow: c.snow,
      fog: c.fog,
      mist: 0.55 * Math.pow(c.rain, 1.3),
      wetness: smoothstep(0, 1, this.wetness),
      snowCover: this.snowCover,
      wind: this.wind,
      mistA: [...this.mist[0].dir, ...this.mist[0].off],
      mistB: [...this.mist[1].dir, ...this.mist[1].off],
      mistW: [1 - Math.abs(2 * this.mist[0].ph - 1), 1 - Math.abs(2 * this.mist[1].ph - 1)],
      rainDrift: this.rainDrift,
      snowDrift: this.snowDrift,
      cloudOffset: this.cloudOffset,
      smoke: clamp(0.85 + 0.2 * c.snow - 0.35 * c.rain, 0, 1),
      flash: this.flash(),
      bolt: boltOn ? this.bolt : null,
      fogColor,
    };
  }
}

/** A jagged bolt from high above down to the ground at (x, y), with a branch or two. */
function makeBolt(x: number, y: number, r: () => number): Float32Array {
  const pts: number[] = [];
  const zig = (x0: number, y0: number, z0: number, x1: number, y1: number, z1: number, steps: number, jag: number) => {
    let px = x0, py = y0, pz = z0;
    for (let i = 1; i <= steps; i++) {
      const t = i / steps;
      const j = i === steps ? 0 : jag;
      const nx = lerp(x0, x1, t) + (r() - 0.5) * j, ny = lerp(y0, y1, t) + (r() - 0.5) * j, nz = lerp(z0, z1, t);
      pts.push(px, py, pz, nx, ny, nz);
      px = nx; py = ny; pz = nz;
    }
  };
  const top = 190;
  const sx = x + (r() - 0.5) * 40, sy = y - 20 - r() * 20;
  zig(sx, sy, top, x, y, 0, 16, 9);
  for (let b = 0, n = 1 + Math.floor(r() * 2); b < n; b++) {
    const i = 2 + Math.floor(r() * 8); // branch from the i-th joint of the main bolt
    const bx = pts[i * 6 + 3], by = pts[i * 6 + 4], bz = pts[i * 6 + 5];
    const a = r() * Math.PI * 2, len = 20 + r() * 30;
    zig(bx, by, bz, bx + Math.cos(a) * len, by + Math.sin(a) * len, bz - 30 - r() * 40, 6, 7);
  }
  return new Float32Array(pts);
}
