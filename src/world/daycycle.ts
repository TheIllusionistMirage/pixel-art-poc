// Time of day → sun/moon direction, light colours, ambient and sky.
// World axes: +x east, +y south (screen-down at rotation 0), +z up.

import { lerp, smoothstep } from '../engine/rng';

export type Vec3 = [number, number, number];

export interface DayLighting {
  /** Direction towards the dominant light (sun by day, moon by night). */
  lightDir: Vec3;
  lightColor: Vec3;
  ambient: Vec3;
  sky: Vec3;
  /** 0 = full day, 1 = full night; drives lamps and windows. */
  night: number;
  phase: 'night' | 'dawn' | 'day' | 'dusk';
}

type Key = [hour: number, ambient: Vec3, sun: Vec3, sky: Vec3];

// Soft, mostly-ambient light (overcast pixel-art look): shadows read as gentle
// tone shifts rather than black holes.
const NIGHT_AMB: Vec3 = [0.17, 0.21, 0.33];
const NIGHT_SKY: Vec3 = [0.04, 0.05, 0.09];
const KEYS: Key[] = [
  [0, NIGHT_AMB, [0, 0, 0], NIGHT_SKY],
  [4.6, NIGHT_AMB, [0, 0, 0], NIGHT_SKY],
  [5.6, [0.36, 0.32, 0.4], [0.8, 0.44, 0.32], [0.36, 0.28, 0.36]],
  [7.0, [0.56, 0.56, 0.62], [0.62, 0.52, 0.42], [0.5, 0.58, 0.66]],
  [12, [0.66, 0.7, 0.74], [0.46, 0.44, 0.4], [0.52, 0.62, 0.68]],
  [16.5, [0.62, 0.6, 0.64], [0.58, 0.5, 0.42], [0.56, 0.6, 0.66]],
  [18.0, [0.42, 0.34, 0.4], [0.8, 0.44, 0.3], [0.5, 0.34, 0.34]],
  [19.3, NIGHT_AMB, [0, 0, 0], NIGHT_SKY],
  [24, NIGHT_AMB, [0, 0, 0], NIGHT_SKY],
];

const MOON: Vec3 = [0.18, 0.24, 0.38];
const SUN_MAX_EL = (62 * Math.PI) / 180;
const MOON_MAX_EL = (48 * Math.PI) / 180;

const lerp3 = (a: Vec3, b: Vec3, t: number): Vec3 => [lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t)];
const scale3 = (a: Vec3, s: number): Vec3 => [a[0] * s, a[1] * s, a[2] * s];

/**
 * Direction for a body travelling east → north → west. (Lighting from the far
 * side of the scene is the pixel-art convention: highlights on the tops of
 * forms, shadows falling towards the viewer where they read best.)
 */
function arc(phi: number, maxEl: number): { dir: Vec3; sinEl: number } {
  const sinEl = Math.sin(phi) * Math.sin(maxEl);
  const el = Math.asin(sinEl);
  const ce = Math.cos(el);
  return { dir: [Math.cos(phi) * ce, -Math.sin(phi) * ce, sinEl], sinEl };
}

export function dayLighting(hour: number): DayLighting {
  hour = ((hour % 24) + 24) % 24;
  let k = 0;
  while (KEYS[k + 1][0] < hour) k++;
  const [h0, a0, s0, k0] = KEYS[k], [h1, a1, s1, k1] = KEYS[k + 1];
  const t = (hour - h0) / (h1 - h0);
  const ambient = lerp3(a0, a1, t), sunKey = lerp3(s0, s1, t), sky = lerp3(k0, k1, t);

  const sun = arc(((hour - 6) / 12) * Math.PI, SUN_MAX_EL);
  const moon = arc(((hour - 18) / 12) * Math.PI, MOON_MAX_EL);
  const sunI = smoothstep(-0.01, 0.14, sun.sinEl);
  const moonI = smoothstep(0.0, 0.2, moon.sinEl) * (1 - smoothstep(-0.12, 0.0, sun.sinEl));
  const night = 1 - smoothstep(-0.1, 0.14, sun.sinEl);

  const useSun = sun.sinEl > -0.05;
  const body = useSun ? sun : moon;
  const dir: Vec3 = [body.dir[0], body.dir[1], Math.max(body.dir[2], 0.04)];
  const len = Math.hypot(...dir);
  const lightColor = useSun ? scale3(sunKey, sunI) : scale3(MOON, moonI);

  const phase = night > 0.9 ? 'night' : night < 0.1 ? 'day' : hour < 12 ? 'dawn' : 'dusk';
  return { lightDir: [dir[0] / len, dir[1] / len, dir[2] / len], lightColor, ambient, sky, night, phase };
}
