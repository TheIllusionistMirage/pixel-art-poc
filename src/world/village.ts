// Procedural village: road network → plaza, church, houses along roads,
// farm fields, forest around the outskirts, props and light sources.
// Produces plain data (placements, lights, ground texture, road graph);
// nothing here touches the GPU.

import type { ModelInfo } from '../assets/atlas';
import type { Library } from '../assets/library';
import { DETAIL, type RGB } from '../assets/voxel';
import { clamp, fbm, hash3, irange, mulberry32, pick, range, smoothstep } from '../engine/rng';

export interface Placed {
  model: ModelInfo;
  x: number;
  y: number;
  angle: number;
}

export interface PointLight {
  x: number;
  y: number;
  z: number;
  radius: number;
  color: [number, number, number];
  /** 0 = steady, >0 = flicker amount. */
  flicker: number;
  phase: number;
  /**
   * Optional: occluders taller than this (world units) are treated as overhangs
   * (tree canopies) the light shines under, instead of solid heightfield columns.
   */
  overhang?: number;
}

export interface RoadGraph {
  nodes: { x: number; y: number }[];
  adj: number[][];
  /** Nodes [0, ringCount) form the plaza ring road. */
  ringCount: number;
}

export interface World {
  size: number;
  /** RGBA8 ground albedo, DETAIL texels per world unit ((size·DETAIL)² texels); alpha = material. */
  ground: Uint8Array;
  statics: Placed[];
  lights: PointLight[];
  roads: RoadGraph;
  counts: Record<string, number>;
}

const WARM: [number, number, number] = [1.0, 0.64, 0.3];
const FIRE: [number, number, number] = [1.0, 0.55, 0.22];

/** Rotation used everywhere: local (lx, ly) → world, matching the shaders. */
const rot = (lx: number, ly: number, a: number): [number, number] => {
  const c = Math.cos(a), s = Math.sin(a);
  return [c * lx - s * ly, s * lx + c * ly];
};
/** Angle that turns a model's front (+y) to face direction (fx, fy). */
export const facing = (fx: number, fy: number) => Math.atan2(-fx, fy);

export function generateVillage(lib: Library, seed = 7): World {
  const S = 1024, CX = S / 2, CY = S / 2;
  const rng = mulberry32(seed);
  const model = (name: string) => {
    const m = lib.atlas.byName.get(name);
    if (!m) throw new Error(`unknown model ${name}`);
    return m;
  };

  const roadMask = new Float32Array(S * S);
  const plazaMask = new Uint8Array(S * S);
  const blocked = new Uint8Array(S * S);
  const fieldKind = new Uint8Array(S * S);
  /** Index+1 into `fields` per world cell; the fine ground texture re-evaluates stripes analytically. */
  const fieldId = new Uint8Array(S * S);
  const fields: { cx: number; cy: number; fw: number; fh: number; ca: number; sa: number }[] = [];

  const statics: Placed[] = [];
  const lights: PointLight[] = [];
  const footprints: { x: number; y: number; r: number }[] = [];
  const counts: Record<string, number> = {};

  const inWorld = (x: number, y: number, margin = 0) => x >= margin && y >= margin && x < S - margin && y < S - margin;
  const at = (x: number, y: number) => (Math.floor(y) * S + Math.floor(x)) | 0;

  const disc = (cx: number, cy: number, r: number, fn: (i: number, d: number) => void) => {
    for (let y = Math.floor(cy - r); y <= Math.ceil(cy + r); y++)
      for (let x = Math.floor(cx - r); x <= Math.ceil(cx + r); x++) {
        if (!inWorld(x, y)) continue;
        const d = Math.hypot(x + 0.5 - cx, y + 0.5 - cy);
        if (d <= r) fn(y * S + x, d);
      }
  };

  const place = (name: string, x: number, y: number, angle: number, block = true, category = name.replace(/\d+(_\d)?$/, '')) => {
    const m = model(name);
    statics.push({ model: m, x, y, angle });
    counts[category] = (counts[category] ?? 0) + 1;
    if (block) {
      const r = Math.min(m.w, m.d) / 2 + 1;
      footprints.push({ x, y, r: m.radius * 0.85 });
      disc(x, y, r, (i) => (blocked[i] = 1));
      disc(x, y, m.radius * 0.85, (i) => (blocked[i] = Math.max(blocked[i], 1)));
    }
  };

  const collides = (x: number, y: number, r: number) => footprints.some((f) => Math.hypot(f.x - x, f.y - y) < f.r + r);

  /** True if a rotated rectangle (half extents hw, hd, plus margin) touches road or blocked area. */
  const areaBusy = (x: number, y: number, hw: number, hd: number, a: number, margin = 2) => {
    for (let ly = -hd - margin; ly <= hd + margin; ly += 2)
      for (let lx = -hw - margin; lx <= hw + margin; lx += 2) {
        const [ox, oy] = rot(lx, ly, a);
        const px = x + ox, py = y + oy;
        if (!inWorld(px, py, 4)) return true;
        const i = at(px, py);
        if (roadMask[i] > 0.05 || blocked[i]) return true;
      }
    return false;
  };

  // ── Road network ──────────────────────────────────────────────────────
  const nodes: { x: number; y: number }[] = [];
  const edges: { a: number; b: number; w: number }[] = [];
  const addNode = (x: number, y: number) => nodes.push({ x, y }) - 1;
  const PLAZA_R = 50, RING_R = 34, RING_N = 12;

  for (let i = 0; i < RING_N; i++) {
    const a = (i / RING_N) * Math.PI * 2;
    addNode(CX + Math.cos(a) * RING_R, CY + Math.sin(a) * RING_R);
    if (i > 0) edges.push({ a: i - 1, b: i, w: 2.5 });
  }
  edges.push({ a: RING_N - 1, b: 0, w: 2.5 });

  const N_MAIN = 5;
  const base = rng() * Math.PI * 2;
  const mainAngles: number[] = [];
  const deadEnds: number[] = [];
  for (let r = 0; r < N_MAIN; r++) {
    let a = base + (r / N_MAIN) * Math.PI * 2 + range(rng, -0.25, 0.25);
    mainAngles.push(a);
    let px = CX + Math.cos(a) * PLAZA_R, py = CY + Math.sin(a) * PLAZA_R;
    let prev = addNode(px, py);
    const ring = ((Math.round((a / (Math.PI * 2)) * RING_N) % RING_N) + RING_N) % RING_N;
    edges.push({ a: ring, b: prev, w: 4.5 });
    const along: { node: number; angle: number }[] = [{ node: prev, angle: a }];
    for (let s = 0; s < 10; s++) {
      a += range(rng, -0.2, 0.2);
      const nx = px + Math.cos(a) * 42, ny = py + Math.sin(a) * 42;
      if (!inWorld(nx, ny, 30)) break;
      const n = addNode(nx, ny);
      edges.push({ a: prev, b: n, w: 4.5 });
      along.push({ node: n, angle: a });
      prev = n;
      px = nx;
      py = ny;
    }
    deadEnds.push(prev);
    for (let j = 2; j < Math.min(along.length - 1, 7); j++) {
      if (rng() > 0.55) continue;
      const side = rng() < 0.5 ? 1 : -1;
      let ba = along[j].angle + side * (Math.PI / 2 + range(rng, -0.3, 0.3));
      let bp = along[j].node;
      const steps = irange(rng, 1, 3);
      for (let k = 0; k < steps; k++) {
        ba += range(rng, -0.2, 0.2);
        const bx = nodes[bp].x + Math.cos(ba) * 38, by = nodes[bp].y + Math.sin(ba) * 38;
        if (!inWorld(bx, by, 30)) break;
        const n = addNode(bx, by);
        edges.push({ a: bp, b: n, w: 3 });
        bp = n;
      }
      deadEnds.push(bp);
    }
  }

  for (const e of edges) {
    const A = nodes[e.a], B = nodes[e.b];
    const len = Math.hypot(B.x - A.x, B.y - A.y);
    const R = e.w + 1.5;
    for (let t = 0; t <= len; t += 0.75) {
      const x = A.x + ((B.x - A.x) * t) / len, y = A.y + ((B.y - A.y) * t) / len;
      disc(x, y, R, (i, d) => {
        const v = clamp((R - d) / 1.5, 0, 1);
        if (v > roadMask[i]) roadMask[i] = v;
      });
    }
  }
  disc(CX, CY, PLAZA_R + 1.5, (i, d) => {
    roadMask[i] = Math.max(roadMask[i], clamp((PLAZA_R + 1.5 - d) / 1.5, 0, 1));
    if (d <= PLAZA_R) plazaMask[i] = 1;
  });

  const adj: number[][] = nodes.map(() => []);
  for (const e of edges) {
    adj[e.a].push(e.b);
    adj[e.b].push(e.a);
  }

  // ── Plaza: well, market stalls, torches ───────────────────────────────
  place('well', CX, CY, 0);
  const sorted = [...mainAngles].map((a) => ((a % (Math.PI * 2)) + Math.PI * 2) % (Math.PI * 2)).sort((a, b) => a - b);
  const gaps = sorted.map((a, i) => {
    const b = i + 1 < sorted.length ? sorted[i + 1] : sorted[0] + Math.PI * 2;
    return { mid: (a + b) / 2, size: b - a };
  });
  gaps.forEach((g, i) => {
    const dx = Math.cos(g.mid), dy = Math.sin(g.mid);
    place(lib.stalls[i % lib.stalls.length], CX + dx * 20, CY + dy * 20, facing(dx, dy));
    const [tx, ty] = [-dy, dx];
    place(rng() < 0.5 ? 'barrel' : 'crate', CX + dx * 22 + tx * 12, CY + dy * 22 + ty * 12, rng() * 6);
    for (const off of [-0.28, 0.28]) {
      const a = g.mid + off;
      const x = CX + Math.cos(a) * (PLAZA_R - 4), y = CY + Math.sin(a) * (PLAZA_R - 4);
      place('torch', x, y, 0);
      lights.push({ x, y, z: 15, radius: 72, color: FIRE, flicker: 0.22, phase: rng() * 100 });
    }
  });

  // ── Church in the widest gap between main roads ───────────────────────
  {
    const g = gaps.reduce((a, b) => (b.size > a.size ? b : a));
    const dx = Math.cos(g.mid), dy = Math.sin(g.mid);
    const m = model('church');
    for (let r = PLAZA_R + 20; r < PLAZA_R + 90; r += 5) {
      const x = CX + dx * r, y = CY + dy * r, a = facing(-dx, -dy);
      if (areaBusy(x, y, m.w / 2, m.d / 2, a, 3)) continue;
      place('church', x, y, a);
      for (const lx of [-22, -5]) {
        const [ox, oy] = rot(lx, m.d / 2 + 2, a);
        lights.push({ x: x + ox, y: y + oy, z: 12, radius: 52, color: WARM, flicker: 0.05, phase: rng() * 100 });
      }
      break;
    }
  }

  // ── Houses along roads ────────────────────────────────────────────────
  for (const e of edges) {
    if (e.w < 3) continue;
    const A = nodes[e.a], B = nodes[e.b];
    const len = Math.hypot(B.x - A.x, B.y - A.y);
    const ux = (B.x - A.x) / len, uy = (B.y - A.y) / len;
    for (let t = 8; t < len - 4; t += range(rng, 20, 28)) {
      for (const side of [-1, 1]) {
        const px = A.x + ux * t, py = A.y + uy * t;
        const dc = Math.hypot(px - CX, py - CY);
        const p = dc < 260 ? 0.9 : dc < 330 ? 0.45 : 0.08;
        if (rng() > p) continue;
        const name = pick(rng, lib.houses);
        const m = model(name);
        const nx = -uy * side, ny = ux * side;
        const dist = e.w + m.d / 2 + range(rng, 4.5, 7);
        const x = px + nx * dist, y = py + ny * dist;
        const a = facing(-nx, -ny) + range(rng, -0.05, 0.05);
        if (areaBusy(x, y, m.w / 2, m.d / 2, a) || collides(x, y, m.radius * 0.8)) continue;
        place(name, x, y, a);
        if (rng() < 0.75) {
          const [ox, oy] = rot(4, m.d / 2 + 1, a);
          lights.push({ x: x + ox, y: y + oy, z: 10, radius: 46, color: WARM, flicker: 0.04, phase: rng() * 100 });
        }
        const pr = rng();
        if (pr < 0.55) {
          const sx = rng() < 0.5 ? -1 : 1;
          const [ox, oy] = rot(sx * (m.w / 2 + 3), m.d / 2 - 5, a);
          if (!collides(x + ox, y + oy, 3)) place(rng() < 0.6 ? 'barrel' : 'crate', x + ox, y + oy, rng() * 6);
        } else if (pr < 0.7) {
          const [ox, oy] = rot(m.w / 2 + 11, -2, a);
          if (!areaBusy(x + ox, y + oy, 4, 8, a + Math.PI / 2, 0) && !collides(x + ox, y + oy, 8))
            place('cart', x + ox, y + oy, a + Math.PI / 2 + range(rng, -0.3, 0.3));
        }
      }
    }
  }

  // ── Campfires at a couple of road ends ────────────────────────────────
  let fires = 0;
  for (const n of deadEnds.sort(() => rng() - 0.5)) {
    if (fires >= 2) break;
    const N = nodes[n];
    const d = Math.hypot(N.x - CX, N.y - CY);
    if (d < 140 || d > 360) continue;
    const x = N.x + range(rng, -6, 6), y = N.y + range(rng, -6, 6);
    if (collides(x, y, 14)) continue;
    place('campfire', x, y, 0);
    lights.push({ x, y, z: 6, radius: 96, color: FIRE, flicker: 0.32, phase: rng() * 100 });
    for (let k = 0; k < 3; k++) {
      const a = rng() * Math.PI * 2;
      place(pick(rng, lib.rocks), x + Math.cos(a) * 12, y + Math.sin(a) * 12, rng() * 6);
    }
    fires++;
  }

  // ── Farm fields ───────────────────────────────────────────────────────
  for (let attempt = 0, made = 0; attempt < 300 && made < 18; attempt++) {
    const ang = rng() * Math.PI * 2, dist = range(rng, 190, 400);
    const cx = CX + Math.cos(ang) * dist, cy = CY + Math.sin(ang) * dist;
    const fw = range(rng, 50, 90), fh = range(rng, 36, 60);
    const nearest = mainAngles.reduce((a, b) => (Math.abs(Math.sin(b - ang)) < Math.abs(Math.sin(a - ang)) ? b : a));
    const fa = nearest + range(rng, -0.1, 0.1);
    if (areaBusy(cx, cy, fw / 2, fh / 2, fa, 4)) continue;
    const kind = irange(rng, 1, 3);
    const R = Math.hypot(fw, fh) / 2 + 1;
    const ca = Math.cos(fa), sa = Math.sin(fa);
    fields.push({ cx, cy, fw, fh, ca, sa });
    disc(cx, cy, R, (i) => {
      const x = (i % S) + 0.5 - cx, y = Math.floor(i / S) + 0.5 - cy;
      const u = ca * x + sa * y, v = -sa * x + ca * y;
      if (Math.abs(u) > fw / 2 || Math.abs(v) > fh / 2) return;
      fieldKind[i] = kind;
      fieldId[i] = fields.length;
      blocked[i] = 2;
    });
    if (kind === 1)
      for (let k = irange(rng, 1, 3); k > 0; k--) {
        const [ox, oy] = rot(range(rng, -fw / 2 + 8, fw / 2 - 8), fh / 2 - 7, fa);
        place('haystack', cx + ox, cy + oy, fa + range(rng, -0.4, 0.4), false);
      }
    if (rng() < 0.65)
      for (let u = -fw / 2 + 8; u < fw / 2 - 4; u += 16) {
        const [ox, oy] = rot(u, -fh / 2 - 2, fa);
        place('fence', cx + ox, cy + oy, fa, false);
      }
    made++;
  }

  // ── Ponds: noisy blobs with a muddy rim, reeds, and someone fishing ────
  const water = new Float32Array(S * S); // 0 = land, else the pond shape value (<1 water, <rim mud)
  /** Pond shape functions, kept so the fine ground texture can trace smooth shorelines. */
  const ponds: { shape: (x: number, y: number) => number; rim: number; x0: number; y0: number; x1: number; y1: number }[] = [];
  for (let attempt = 0, made = 0; attempt < 400 && made < 3; attempt++) {
    const ang = rng() * Math.PI * 2, dist = range(rng, 120, 330);
    const cx = CX + Math.cos(ang) * dist, cy = CY + Math.sin(ang) * dist;
    const r = range(rng, 24, 38), ax = range(rng, 0.75, 1.3);
    const ext = r * Math.max(ax, 1 / ax) * 1.45;
    if (areaBusy(cx, cy, ext, ext, 0, 6)) continue;
    const ps = seed + 50 + made * 7;
    const shape = (x: number, y: number) =>
      Math.hypot((x - cx) / (r * ax), (y - cy) / (r / ax)) + (fbm(x / 22, y / 22, ps, 3) - 0.5) * 0.7;
    const rim = 1 + 2.5 / r;
    const pe = ext * 1.15;
    ponds.push({ shape, rim, x0: cx - pe, y0: cy - pe, x1: cx + pe, y1: cy + pe });
    disc(cx, cy, ext * 1.15, (i, _d) => {
      const v = shape((i % S) + 0.5, Math.floor(i / S) + 0.5);
      if (v < rim) water[i] = Math.max(v, 1e-3);
      if (v < 1 + 8 / r) blocked[i] = 2;
    });
    /** Point on the shoreline in direction a (shape value ≈ target). */
    const shore = (a: number, target: number): [number, number] => {
      let t = 0;
      while (t < ext * 1.5 && shape(cx + Math.cos(a) * t, cy + Math.sin(a) * t) < target) t += 0.5;
      return [cx + Math.cos(a) * t, cy + Math.sin(a) * t];
    };
    if (made === 0) {
      const a = rng() * Math.PI * 2;
      const [sx, sy] = shore(a, 1.0);
      const fa = facing(-Math.cos(a), -Math.sin(a));
      place('fisher', sx, sy, fa, true);
      // A lantern on the bank at the fisher's right hand (model-local offset, world units).
      // The light sits just above the 7.5-unit lantern so the heightfield shadow test doesn't block it,
      // and it shines under tree canopies (the fisher sits beneath an oak).
      const [lx, ly] = rot(5, -8.5, fa);
      place('lantern', sx + lx, sy + ly, fa + 0.3, false);
      lights.push({ x: sx + lx, y: sy + ly, z: 8.5, radius: 40, color: WARM, flicker: 0.08, phase: sx, overhang: 24 }); // no rng(): keeps the rest of the layout unchanged
    }
    const nReeds = Math.round(r * 0.55);
    for (let k = 0; k < nReeds; k++) {
      const a = rng() * Math.PI * 2;
      const [x, y] = shore(a, range(rng, 0.9, 1.04));
      if (collides(x, y, 5)) continue;
      place(pick(rng, lib.reeds), x, y, rng() * Math.PI * 2, false, 'reeds');
    }
    for (let k = 0; k < 3; k++) {
      const a = rng() * Math.PI * 2;
      const [x, y] = shore(a, 1.25);
      if (!collides(x, y, 6)) place(rng() < 0.5 ? pick(rng, lib.rocks) : pick(rng, lib.bushes), x, y, rng() * 6, false);
    }
    made++;
  }

  // ── Forest ────────────────────────────────────────────────────────────
  const forestDensity = (x: number, y: number) => {
    const dc = Math.hypot(x - CX, y - CY);
    const clump = smoothstep(0.32, 0.62, fbm(x / 150, y / 150, seed + 11, 3));
    return clamp(0.03 + smoothstep(210, 380, dc) * (0.25 + 0.75 * clump), 0, 0.92);
  };
  const clearOf = (x: number, y: number, r: number) => {
    for (const [ox, oy] of [[0, 0], [r, 0], [-r, 0], [0, r], [0, -r]]) {
      if (!inWorld(x + ox, y + oy, 2)) return false;
      const i = at(x + ox, y + oy);
      if (roadMask[i] > 0.02 || blocked[i]) return false;
    }
    return true;
  };
  for (let gy = 6; gy < S - 6; gy += 8)
    for (let gx = 6; gx < S - 6; gx += 8) {
      const x = gx + range(rng, -3.5, 3.5), y = gy + range(rng, -3.5, 3.5);
      if (rng() > forestDensity(x, y)) continue;
      if (!clearOf(x, y, 6)) continue;
      const dc = Math.hypot(x - CX, y - CY);
      const r = rng();
      const pineP = smoothstep(300, 480, dc) * 0.65;
      const name = r < 0.1 ? pick(rng, lib.bushes) : r < 0.1 + pineP ? pick(rng, lib.pines) : pick(rng, lib.oaks);
      place(name, x, y, rng() * Math.PI * 2, false);
    }
  for (let k = 0; k < 90; k++) {
    const x = range(rng, 20, S - 20), y = range(rng, 20, S - 20);
    if (clearOf(x, y, 5)) place(pick(rng, lib.rocks), x, y, rng() * 6, false);
  }
  // Clumps of tall meadow grass, gathered in drifts.
  for (let k = 0; k < 4000; k++) {
    const x = range(rng, 10, S - 10), y = range(rng, 10, S - 10);
    if (fbm(x / 70, y / 70, seed + 41, 2) < 0.58 || !clearOf(x, y, 4)) continue;
    place(pick(rng, lib.grasses), x, y, rng() * Math.PI * 2, false, 'tallgrass');
  }

  // ── Ground texture ────────────────────────────────────────────────────
  // Painted like pixel art at DETAIL texels per world unit: a few flat tones per
  // surface, transitions done with an ordered (Bayer) dither instead of
  // gradients, and small hand-drawn-style stamps (grass tufts, flowers) on top.
  // Layout masks are per world unit; they are sampled bilinearly (roads) or
  // re-evaluated analytically (ponds) so edges stay smooth at the finer grid.
  const D = DETAIL, GS = S * D;
  const ground = new Uint8Array(GS * GS * 4);
  const mat = new Uint8Array(GS * GS).fill(255);
  const BAYER = [0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5].map((v) => (v + 0.5) / 16);
  const dith = (x: number, y: number) => BAYER[(y & 3) * 4 + (x & 3)];
  const P = {
    grass0: [70, 102, 64], grass1: [74, 107, 66], grassLight: [90, 116, 64], grassLight2: [104, 126, 66],
    grassDark: [62, 92, 60], forest0: [64, 94, 62], forest1: [58, 88, 60],
    dirt: [128, 110, 84], dirtEdge: [108, 94, 72], dirtDark: [110, 94, 72], pebble: [146, 130, 104], rut: [116, 99, 76],
    soil: [104, 86, 64], soilDark: [88, 72, 56], soilFurrow: [80, 66, 52], wheat: [180, 160, 96], wheatDark: [158, 138, 84], wheatLight: [196, 178, 110],
    cabbage: [82, 120, 68], cabbageLight: [102, 138, 74],
    cobble: [134, 130, 120], cobbleDark: [120, 117, 108], cobbleLight: [148, 144, 132], cobbleHi: [162, 158, 144], mortar: [104, 102, 96],
    water: [62, 100, 104], waterDeep: [56, 92, 98], mud: [90, 70, 56], mudDark: [74, 60, 50], bankShade: [58, 88, 58],
    tuft: [46, 74, 58], tuftMid: [58, 90, 60], tuftLight: [84, 112, 64], flower: [216, 208, 186], flowerEye: [222, 196, 96],
    flowerRed: [196, 92, 80], flowerBlue: [128, 132, 186],
  } satisfies Record<string, RGB>;

  /** Bilinear sample of a per-world-unit mask at world position (wx, wy). */
  const bilinear = (a: Float32Array, wx: number, wy: number) => {
    const x = clamp(wx - 0.5, 0, S - 1.001), y = clamp(wy - 0.5, 0, S - 1.001);
    const x0 = Math.floor(x), y0 = Math.floor(y), fx = x - x0, fy = y - y0;
    const i = y0 * S + x0;
    return (a[i] * (1 - fx) + a[i + 1] * fx) * (1 - fy) + (a[i + S] * (1 - fx) + a[i + S + 1] * fx) * fy;
  };

  // Fine water: the pond shape evaluated per texel (0 = land).
  const fineWater = new Float32Array(GS * GS);
  for (const pd of ponds)
    for (let fy = Math.max(0, Math.floor(pd.y0 * D)); fy < Math.min(GS, Math.ceil(pd.y1 * D)); fy++)
      for (let fx = Math.max(0, Math.floor(pd.x0 * D)); fx < Math.min(GS, Math.ceil(pd.x1 * D)); fx++) {
        const v = pd.shape((fx + 0.5) / D, (fy + 0.5) / D);
        if (v < pd.rim) fineWater[fy * GS + fx] = Math.max(v, 1e-3);
      }

  for (let y = 0; y < GS; y++)
    for (let x = 0; x < GS; x++) {
      const wx = (x + 0.5) / D, wy = (y + 0.5) / D;
      const i = Math.floor(wy) * S + Math.floor(wx); // world cell
      const fi = y * GS + x; // texel
      const d = dith(x, y);
      const h = hash3(x, y, 0, seed + 9);
      // Grass: two broad tones, lighter dithered patches, darker hollows.
      let c: RGB = fbm(wx / 90, wy / 90, seed + 3, 2) + (d - 0.5) * 0.1 > 0.5 ? P.grass1 : P.grass0;
      const lp = fbm(wx / 16, wy / 16, seed + 21, 2);
      if (lp > 0.6 && d < (lp - 0.6) * 22) c = lp > 0.67 && d < (lp - 0.67) * 22 ? P.grassLight2 : P.grassLight;
      const dp = fbm(wx / 28, wy / 28, seed + 31, 2);
      if (dp < 0.33 && d < (0.33 - dp) * 20) c = P.grassDark;
      const fd = smoothstep(0.15, 0.75, forestDensity(wx, wy));
      if (fd + (d - 0.5) * 0.6 > 0.5) c = fd + (d - 0.5) * 0.6 > 0.8 ? P.forest1 : P.forest0;

      const fk = fieldKind[i];
      const fl = fk ? fields[fieldId[i] - 1] : undefined;
      const fu = fl ? fl.ca * (wx - fl.cx) + fl.sa * (wy - fl.cy) : 0;
      const fv = fl ? -fl.sa * (wx - fl.cx) + fl.ca * (wy - fl.cy) : 0;
      if (fl && Math.abs(fu) <= fl.fw / 2 && Math.abs(fv) <= fl.fh / 2) {
        // Crop rows 2.5 world units apart; a touch of dither breaks up the row edges.
        const s = Math.floor((fv + fl.fh) / 2.5 + (d - 0.5) * 0.12) % 2;
        if (fk === 1) c = s ? (h > 0.8 ? P.wheatDark : h < 0.1 ? P.wheatLight : P.wheat) : h > 0.6 ? P.soilFurrow : P.soil;
        else if (fk === 2) c = s ? (hash3(x >> 1, y >> 1, 4, seed) > 0.62 ? P.cabbageLight : P.cabbage) : P.soil;
        else c = s ? P.soil : h > 0.8 ? P.soilFurrow : P.soilDark;
      }
      const rm = bilinear(roadMask, wx, wy) + (d - 0.5) * 0.5;
      if (rm > 0.5) {
        c = rm < 0.72 ? P.dirtEdge : h > 0.97 ? P.dirtDark : P.dirt;
        const pb = hash3(x >> 1, y >> 1, 2, seed);
        if (pb > 0.95) c = (x & 1) === 0 && (y & 1) === 0 && pb > 0.985 ? P.pebble : P.dirtDark;
        else if (rm > 0.85 && rm < 0.95) c = P.rut;
      }
      if (plazaMask[i]) {
        // Cobbles: 6×4-texel stones in offset rows, a lit top-left edge, varied tones.
        const row = Math.floor(y / 4);
        const ox = x + (row % 2) * 3 + Math.floor(hash3(row, 0, 5, seed) * 3);
        const sx = Math.floor(ox / 6), u = ox % 6, v = y % 4;
        const mortar = v === 0 || u === 0;
        const tone = hash3(sx, row, 3, seed);
        if (mortar) c = P.mortar;
        else if (v === 1 && u < 4) c = tone > 0.5 ? P.cobbleHi : P.cobbleLight;
        else if (v === 3 && u === 5) c = P.cobbleDark;
        else c = tone > 0.78 ? P.cobbleLight : tone < 0.2 ? P.cobbleDark : P.cobble;
      }
      const wv = fineWater[fi];
      if (wv > 0) {
        if (wv < 1) {
          c = wv + (d - 0.5) * 0.25 < 0.62 ? P.waterDeep : P.water;
          mat[fi] = 50;
        } else c = wv < 1 + 1.2 / 30 ? P.mudDark : h > 0.9 ? P.mudDark : P.mud;
      } else if (x > 0 && y > 0 && x < GS - 1 && y < GS - 1 && (fineWater[fi - 1] > 0 || fineWater[fi + 1] > 0 || fineWater[fi - GS] > 0 || fineWater[fi + GS] > 0)) {
        c = P.bankShade;
      }
      ground[fi * 4] = c[0];
      ground[fi * 4 + 1] = c[1];
      ground[fi * 4 + 2] = c[2];
    }

  // Stamps (in texels): dark grass tufts and small flowers, only on open grass.
  const TUFTS = [
    ['x.......x', '.x..x..x.', '.x.x.x.x.', '..xx.xx..', '..xxmxx..'],
    ['x.....x', '.x.x.x.', '.x.x.x.', '..xxx..', '..xmx..'],
    ['..x...x..', 'x..x.x..x', '.x.x.x.x.', '.xx.m.xx.', '..xxxxx..'],
    ['.x..x', 'x.x.x', '.xmx.', '.xxx.'],
    ['l.....l', '.x.l.x.', 'x.x.x.x', '.xxmxx.', '..xxx..'],
  ];
  const FLOWERS = [
    ['..o..', '.oeo.', '..o..', '..x..', '.x.x.'],
    ['.o.o.', 'o.e.o', '.o.o.', '..x..', '.xxx.'],
    ['o...', '.x.o', '.xx.'],
  ];
  const canStamp = (x: number, y: number) => {
    if (x < 0 || y < 0 || x >= GS || y >= GS) return false;
    const i = at(x / D, y / D);
    return !fieldKind[i] && !plazaMask[i] && roadMask[i] < 0.2 && fineWater[y * GS + x] === 0 && blocked[i] !== 1;
  };
  const stamp = (art: string[], x0: number, y0: number, colors: Record<string, RGB>) => {
    for (let r = 0; r < art.length; r++)
      for (let k = 0; k < art[r].length; k++) {
        const col = colors[art[r][k]];
        const x = x0 + k, y = y0 + r;
        if (!col || !canStamp(x, y)) continue;
        const i = (y * GS + x) * 4;
        ground[i] = col[0];
        ground[i + 1] = col[1];
        ground[i + 2] = col[2];
      }
  };
  for (let k = 0; k < 2600 * D * D; k++) {
    const x = irange(rng, 4, GS - 12), y = irange(rng, 4, GS - 12);
    if (!canStamp(x + 4, y + 4) || fbm(x / D / 40, y / D / 40, seed + 61, 2) < 0.4) continue;
    if (rng() < 0.16) {
      const petal = pick(rng, [P.flower, P.flower, P.flower, P.flowerRed, P.flowerBlue]);
      stamp(pick(rng, FLOWERS), x, y, { o: petal, e: P.flowerEye, x: P.tuftMid });
    } else stamp(pick(rng, TUFTS), x, y, { x: rng() < 0.7 ? P.tuft : P.tuftMid, m: P.tuftMid, l: P.tuftLight });
  }
  for (let i = 0; i < GS * GS; i++) ground[i * 4 + 3] = mat[i];

  return { size: S, ground, statics, lights, roads: { nodes, adj, ringCount: RING_N }, counts };
}
