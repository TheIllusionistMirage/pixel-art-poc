// Procedural voxel models, authored on the DETAIL grid (2 voxels per world unit).
// Conventions:
//   x → right, y → "south" (towards the camera at rotation 0), z → up.
//   The model's front faces +y; placement rotates it to face roads etc.
//   All coordinates below are voxels; world size = voxels / DETAIL.

import { hash3, mulberry32, range, irange, pick, type Rng } from '../engine/rng';
import { DETAIL, MAT, VoxelModel, jitter, mix, shade, type RGB } from './voxel';

/** Voxels per world unit, for converting world-unit options (house sizes) to voxels. */
const S = DETAIL;

const C = {
  plaster: [208, 196, 166],
  timber: [88, 66, 52],
  stone: [136, 138, 130],
  stoneDark: [106, 110, 106],
  stoneLight: [162, 160, 148],
  mortar: [96, 96, 92],
  thatch: [178, 152, 94],
  thatchDark: [146, 122, 78],
  tile: [152, 90, 70],
  tileDark: [128, 76, 62],
  slate: [92, 100, 112],
  slateLight: [112, 120, 132],
  door: [110, 80, 58],
  glass: [44, 52, 62],
  lead: [46, 44, 50],
  wood: [130, 100, 72],
  woodDark: [90, 68, 52],
  plank: [142, 114, 82],
  bark: [104, 70, 56],
  iron: [66, 68, 70],
  brass: [176, 150, 84],
  leather: [74, 56, 42],
  rope: [150, 128, 94],
  straw: [188, 164, 102],
  water: [62, 100, 104],
  fire1: [255, 214, 110],
  fire2: [255, 140, 48],
  ember: [190, 72, 40],
  gold: [212, 182, 92],
  moss: [84, 104, 66],
  mossLight: [104, 122, 72],
} as const satisfies Record<string, RGB>;

type Fill = (x: number, y: number, z: number) => RGB | null;
const clamp01 = (v: number) => (v < 0 ? 0 : v > 1 ? 1 : v);

/** Thick line (sphere brush of radius r; r < 0.6 draws single voxels). */
function line(m: VoxelModel, a: readonly number[], b: readonly number[], r: number, fill: Fill, mat: number = MAT.SOLID) {
  const len = Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]);
  const n = Math.max(1, Math.ceil(len * 2));
  for (let i = 0; i <= n; i++) {
    const t = i / n;
    const cx = a[0] + (b[0] - a[0]) * t, cy = a[1] + (b[1] - a[1]) * t, cz = a[2] + (b[2] - a[2]) * t;
    if (r < 0.6) {
      const x = Math.floor(cx), y = Math.floor(cy), z = Math.floor(cz);
      const c = fill(x, y, z);
      if (c) m.set(x, y, z, c, mat);
      continue;
    }
    for (let z = Math.floor(cz - r); z <= cz + r; z++)
      for (let y = Math.floor(cy - r); y <= cy + r; y++)
        for (let x = Math.floor(cx - r); x <= cx + r; x++) {
          if (Math.hypot(x + 0.5 - cx, y + 0.5 - cy, z + 0.5 - cz) > r) continue;
          const c = fill(x, y, z);
          if (c) m.set(x, y, z, c, mat);
        }
  }
}

/** Filled disc (one layer) centred on (cx, cy). */
function disc(m: VoxelModel, cx: number, cy: number, z: number, r: number, fill: Fill, mat: number = MAT.SOLID) {
  for (let y = Math.floor(cy - r); y <= cy + r; y++)
    for (let x = Math.floor(cx - r); x <= cx + r; x++) {
      if (Math.hypot(x + 0.5 - cx, y + 0.5 - cy) >= r) continue;
      const c = fill(x, y, z);
      if (c) m.set(x, y, z, c, mat);
    }
}

// ── Roofs ────────────────────────────────────────────────────────────────

/** (x, y, z, k = layer above the eaves, kTop = ridge layer, end = outermost gable column). */
type RoofFill = (x: number, y: number, z: number, k: number, kTop: number, end: boolean) => RGB;

/**
 * Gable roof whose ridge runs along x, over walls occupying [x0,x1) × [y0,y1).
 * Starts at zBase, overhangs `ov` voxels on the long sides and `endOv` at the
 * gable ends; the roof skin is three voxels thick. Interior of each layer is
 * filled with the gable wall.
 */
function gableRoof(m: VoxelModel, x0: number, x1: number, y0: number, y1: number, zBase: number, ov: number, endOv: number, roof: RoofFill, gable: Fill): number {
  const ya0 = y0 - ov, yb0 = y1 + ov;
  const kTop = Math.ceil((yb0 - ya0) / 2) - 1;
  let k = 0;
  for (; ; k++) {
    const ya = ya0 + k, yb = yb0 - k, z = zBase + k;
    if (ya >= yb || z >= m.h) break;
    for (let x = x0 - endOv; x < x1 + endOv; x++)
      for (let y = ya; y < yb; y++) {
        const edge = y < ya + 3 || y >= yb - 3; // 3-thick skin: rotated slices never sample through to the gable fill
        const end = x === x0 - endOv || x === x1 + endOv - 1;
        const c = edge ? roof(x, y, z, k, kTop, end) : x >= x0 && x < x1 ? gable(x, y, z) : null;
        if (c) m.set(x, y, z, c);
      }
  }
  return zBase + k; // first empty layer above the ridge
}

function roofFill(kind: 'thatch' | 'tile' | 'slate', seed: number): RoofFill {
  return (x, y, z, k, kTop, end) => {
    const n = hash3(x, y, z, seed);
    const ridge = k >= kTop - 1;
    if (kind === 'thatch') {
      // Thick bundled courses with straw streaks, a darker pegged ridge cap.
      if (ridge) return jitter(shade(C.thatchDark, x % 6 === 0 ? 0.84 : 1), 0.04, n);
      const course = Math.floor(k / 3);
      const streak = 0.93 + 0.12 * hash3(x, course, 1, seed);
      let f = k % 3 === 0 ? 0.84 : streak;
      if (k < 2) f *= 0.9;
      if (end) f *= 0.86;
      return jitter(shade(C.thatch, f), 0.035, n);
    }
    if (kind === 'tile') {
      if (ridge) return jitter(shade(C.tileDark, x % 5 === 0 ? 0.82 : 0.96), 0.03, n);
      if (end) return jitter(C.woodDark, 0.04, n); // bargeboard
      const course = Math.floor(k / 2);
      const tu = x + (course % 2) * 2;
      if (tu % 4 === 0) return jitter(shade(C.tileDark, 0.8), 0.03, n);
      const tone = hash3(tu >> 2, course, 2, seed);
      return jitter(shade(mix(C.tile, C.tileDark, tone * 0.6), k % 2 === 0 ? 0.88 : 1.02), 0.025, n);
    }
    if (ridge) return jitter(shade(C.slate, 0.8), 0.03, n);
    if (end) return jitter(shade(C.slate, 0.84), 0.03, n);
    const course = Math.floor(k / 2);
    const su = x + (course % 2) * 3;
    if (su % 5 === 0) return shade(C.slate, 0.78);
    const tone = hash3(Math.floor(su / 5), course, 3, seed);
    return jitter(shade(mix(C.slate, C.slateLight, tone * 0.6), k % 2 === 0 ? 0.88 : 1), 0.02, n);
  };
}

// ── Houses ───────────────────────────────────────────────────────────────

export interface HouseOpts {
  /** Footprint and wall height in world units. */
  w: number;
  d: number;
  wallH: number;
  roof: 'thatch' | 'tile' | 'slate';
  walls: 'timber' | 'stone' | 'plank';
  chimney: boolean;
  seed: number;
}

const SHUTTERS: RGB[] = [[92, 108, 96], [128, 78, 62], [84, 96, 118], [96, 74, 56]];
const FLOWERS: RGB[] = [[196, 84, 76], [222, 196, 96], [222, 214, 196], [150, 104, 168]];

export function house(o: HouseOpts): VoxelModel {
  const rng = mulberry32(o.seed);
  const w = o.w * S, d = o.d * S, wallH = o.wallH * S;
  const OV = 3, END = 2;
  const W = w + 2 * OV, D = d + 2 * OV;
  const H = wallH + Math.ceil(D / 2) + 6;
  const m = new VoxelModel(W, D, H);
  const x0 = OV, x1 = OV + w, y0 = OV, y1 = OV + d;
  const midRail = wallH >= 24 ? Math.floor(wallH / 2) : -1;

  /** Position along the wall face: front/back faces run along x, side faces along y. */
  const along = (x: number, y: number): [number, number, boolean] =>
    (x === x0 || x === x1 - 1) && y > y0 && y < y1 - 1 ? [y - y0, d, true] : [x - x0, w, false];
  const postsFor = (len: number) => {
    const n = Math.max(2, Math.round(len / 11));
    return Array.from({ length: n + 1 }, (_, i) => Math.min(len - 2, Math.max(0, Math.round((i * len) / n) - 1)));
  };
  const postsW = postsFor(w), postsD = postsFor(d);

  const timber: Fill = (x, y, z) => {
    const n = hash3(x, y, z, o.seed);
    const [u, len, side] = along(x, y);
    if (z >= wallH) {
      // Gable triangle: a king post, otherwise plaster.
      return Math.abs(u + 0.5 - len / 2) < 1 ? jitter(C.timber, 0.04, n) : jitter(C.plaster, 0.015, n);
    }
    const ps = side ? postsD : postsW;
    const post = ps.some((p) => u >= p && u < p + 2);
    const beam = (z >= 2 && z < 4) || z >= wallH - 2 || (midRail > 0 && z >= midRail && z < midRail + 2);
    if (post || beam) return jitter(C.timber, 0.04, n);
    if (side && ps.length > 2) {
      // Diagonal braces in the two end panels of each side wall.
      const zb = 4, zt = midRail > 0 ? midRail : wallH - 2;
      const p0 = ps[0] + 2, p1 = ps[1], q0 = ps[ps.length - 2] + 2, q1 = ps[ps.length - 1];
      if (u >= p0 && u < p1 && Math.abs(z + 0.5 - (zb + ((u - p0 + 0.5) / (p1 - p0)) * (zt - zb))) < 1.2) return jitter(C.timber, 0.04, n);
      if (u >= q0 && u < q1 && Math.abs(z + 0.5 - (zb + ((q1 - u - 0.5) / (q1 - q0)) * (zt - zb))) < 1.2) return jitter(C.timber, 0.04, n);
    }
    const grime = z < 9 ? (9 - z) * 0.012 : 0;
    return jitter(shade(C.plaster, 1 - grime), 0.015, n);
  };

  const stone: Fill = (x, y, z) => {
    const n = hash3(x, y, z, o.seed);
    const [u] = along(x, y);
    const row = Math.floor(z / 3);
    const ou = u + (row % 2) * 3 + Math.floor(hash3(row, 0, 9, o.seed) * 3);
    if (z % 3 === 0 || ou % 6 === 0) return jitter(C.mortar, 0.04, n);
    return jitter(mix(C.stone, C.stoneDark, hash3(Math.floor(ou / 6), row, 7, o.seed) * 0.7), 0.03, n);
  };

  const plank: Fill = (x, y, z) => {
    const n = hash3(x, y, z, o.seed);
    const [u, len] = along(x, y);
    if (z < wallH && (u < 2 || u >= len - 2)) return jitter(C.woodDark, 0.04, n); // corner boards
    const board = Math.floor(z / 3);
    if (z % 3 === 0) return jitter(shade(C.plank, 0.76), 0.03, n); // shadow line under each board
    const off = Math.floor(hash3(board, 0, 4, o.seed) * 14);
    if ((u + off) % 14 === 0) return shade(C.plank, 0.8); // butt joint
    return jitter(shade(C.plank, 0.94 + 0.1 * hash3(board, Math.floor((u + off) / 14), 5, o.seed)), 0.03, n);
  };

  const body = o.walls === 'stone' ? stone : o.walls === 'plank' ? plank : timber;
  const wall: Fill = (x, y, z) => {
    if (z >= 2) return body(x, y, z);
    const [u] = along(x, y);
    const n = hash3(x, y, z, o.seed);
    return (u + z * 2) % 5 === 0 ? jitter(C.mortar, 0.04, n) : jitter(C.stoneDark, 0.05, n);
  };
  m.box(x0, y0, 0, x1, y1, wallH, wall);

  // ── Door: vertical planks, iron strap hinges, a ring handle, frame and doorstep.
  const front = y1 - 1, back = y0;
  const doorW = 6, doorH = Math.min(15, wallH - 8);
  const dx0 = x0 + Math.floor((w - doorW) / 2);
  const frameCol: RGB = o.walls === 'stone' ? C.stoneLight : C.timber;
  for (let z = 1; z < doorH + 2; z++)
    for (let x = dx0 - 1; x <= dx0 + doorW; x++) {
      let c: RGB;
      if (z >= doorH || x === dx0 - 1 || x === dx0 + doorW) c = jitter(frameCol, 0.04, hash3(x, 1, z, o.seed));
      else if ((z === 3 || z === doorH - 3) && x < dx0 + doorW - 1) c = C.iron;
      else if (x === dx0 + doorW - 2 && z === (doorH >> 1)) c = C.brass;
      else c = jitter(shade(C.door, (x - dx0) % 2 === 0 ? 0.86 : 1), 0.05, hash3(x, 0, z, o.seed));
      m.set(x, front, z, c);
    }
  m.box(dx0 - 1, front + 1, 0, dx0 + doorW + 1, front + 3, 1, (x, y, z) => jitter(C.stoneLight, 0.06, hash3(x, y, z, o.seed)));
  m.anchors.door = [dx0 + doorW / 2, front + 2, 0];

  // ── Windows: mullion cross, frame, a sill; optional shutters and flower box.
  const shutterCol = pick(rng, SHUTTERS);
  const flowerBox = rng() < 0.45;
  const winFrame: RGB = o.walls === 'stone' ? C.stoneLight : o.walls === 'plank' ? C.woodDark : C.timber;
  const WIN_W = 5;
  const win = (x: number, y: number, dx: number, dy: number, ox: number, oy: number, wz: number, wh: number, extras: boolean) => {
    for (let i = -1; i <= WIN_W; i++)
      for (let z = wz - 1; z <= wz + wh; z++) {
        const wx = x + dx * i, wy = y + dy * i;
        const frame = i === -1 || i === WIN_W || z === wz - 1 || z === wz + wh;
        const mullion = i === 2 || z === wz + (wh >> 1);
        if (frame) m.set(wx, wy, z, jitter(winFrame, 0.04, hash3(wx, wy, z, o.seed)));
        else if (mullion) m.set(wx, wy, z, [70, 56, 44]);
        else m.set(wx, wy, z, z === wz + wh - 1 ? shade(C.glass, 1.25) : C.glass, MAT.WINDOW);
      }
    for (let i = -1; i <= WIN_W; i++) m.set(x + dx * i + ox, y + dy * i + oy, wz - 1, shade(winFrame, 1.12));
    if (!extras) return;
    for (const i0 of [-3, WIN_W + 1])
      for (let i = i0; i < i0 + 2; i++)
        for (let z = wz; z < wz + wh; z++)
          m.set(x + dx * i, y + dy * i, z, z === wz + (wh >> 1) ? shade(shutterCol, 0.75) : shade(shutterCol, i % 2 ? 0.9 : 1));
    if (flowerBox)
      for (let i = -1; i <= WIN_W; i++) {
        const fx = x + dx * i + ox, fy = y + dy * i + oy;
        m.set(fx, fy, wz - 2, jitter(C.woodDark, 0.05, hash3(fx, fy, 0, o.seed)));
        const h = hash3(fx, fy, 1, o.seed);
        m.set(fx, fy, wz - 1, h < 0.45 ? pick(rng, FLOWERS) : h < 0.8 ? [76, 112, 64] : [96, 130, 70], MAT.FOLIAGE);
      }
  };
  const wz0 = Math.max(7, Math.floor(wallH * 0.38)), wh = 6;
  win(x0 + 4, front, 1, 0, 0, 1, wz0, wh, true);
  win(x1 - 9, front, 1, 0, 0, 1, wz0, wh, true);
  win(x0 + Math.floor(w / 2) - 2, back, 1, 0, 0, -1, wz0, wh, false);
  const sy = y0 + Math.floor(d / 2) - 2;
  win(x0, sy, 0, 1, -1, 0, wz0, wh, false);
  win(x1 - 1, sy, 0, 1, 1, 0, wz0, wh, false);
  if (wallH >= 26) {
    const uz = wallH - 9;
    if (uz > wz0 + wh + 2) {
      win(x0 + 5, front, 1, 0, 0, 1, uz, 4, false);
      win(x1 - 10, front, 1, 0, 0, 1, uz, 4, false);
    }
  }

  const top = gableRoof(m, x0, x1, y0, y1, wallH, OV, END, roofFill(o.roof, o.seed), wall);

  if (o.chimney) {
    const cx = x1 - 10, cy = Math.floor(D / 2) - 3;
    const zTop = Math.min(H - 1, top + 4);
    for (let z = wallH - 4; z <= zTop; z++) {
      const cap = z >= zTop - 1;
      for (let y = cy - (cap ? 1 : 0); y < cy + 5 + (cap ? 1 : 0); y++)
        for (let x = cx - (cap ? 1 : 0); x < cx + 5 + (cap ? 1 : 0); x++) {
          const n = hash3(x, y, z, o.seed + 5);
          const hole = z === zTop && x > cx && x < cx + 4 && y > cy && y < cy + 4;
          const mortar = z % 2 === 0 || (x + y + ((z >> 1) % 2) * 2) % 4 === 0;
          m.set(x, y, z, hole ? [30, 26, 24] : cap ? jitter(C.stoneLight, 0.05, n) : jitter(mortar ? C.mortar : C.stoneDark, 0.08, n));
        }
    }
    m.anchors.chimney = [cx + 2.5, cy + 2.5, zTop + 1];
  }
  return m;
}

// ── Church ───────────────────────────────────────────────────────────────

export function church(seed: number): VoxelModel {
  const m = new VoxelModel(124, 64, 138);
  const stone: Fill = (x, y, z) => {
    const n = hash3(x, y, z, seed);
    const row = Math.floor(z / 3);
    const ou = x + y + (row % 2) * 4;
    if (z % 3 === 0 || ou % 8 === 0) return jitter(C.mortar, 0.04, n);
    return jitter(mix(C.stone, C.stoneDark, hash3(Math.floor(ou / 8), row, 3, seed) * 0.7), 0.03, n);
  };
  const dressed: Fill = (x, y, z) => jitter(C.stoneLight, 0.04, hash3(x, y, z, seed + 2));

  // Nave with buttresses between tall lancet windows.
  const nx0 = 34, nx1 = 118, ny0 = 12, ny1 = 52, nH = 44;
  m.box(nx0, ny0, 0, nx1, ny1, nH, stone);
  const bays = 6, bayW = (nx1 - nx0) / bays;
  for (let b = 0; b <= bays; b++) {
    const bx = Math.min(nx1 - 3, Math.max(nx0, Math.round(nx0 + b * bayW) - 1));
    for (const [z0, z1, depth] of [[0, 30, 3], [30, 36, 2], [36, 40, 1]]) {
      m.box(bx, ny0 - depth, z0, bx + 3, ny0, z1, z1 === 40 ? dressed : stone);
      m.box(bx, ny1, z0, bx + 3, ny1 + depth, z1, z1 === 40 ? dressed : stone);
    }
  }
  const stained: RGB[] = [[80, 98, 146], [146, 76, 70], [82, 122, 92], [168, 144, 76]];
  const lancet = (lx: number, y: number, oy: number, b: number) => {
    for (let z = 9; z <= 35; z++)
      for (let i = -1; i <= 5; i++) {
        const half = z < 31 ? 2.5 : z < 33 ? 1.8 : z < 35 ? 1 : 0;
        const di = Math.abs(i - 2);
        if (z === 9) {
          m.set(lx + i, y, z, C.stoneLight);
          m.set(lx + i, y + oy, z, C.stoneLight);
        } else if (di < half) {
          const lead = i === 2 || (z - 10) % 5 === 0;
          if (lead) m.set(lx + i, y, z, C.lead);
          else m.set(lx + i, y, z, stained[(Math.floor((z - 10) / 5) + (i > 2 ? 1 : 0) + b) % 4], MAT.WINDOW);
        } else if (di < half + 1) m.set(lx + i, y, z, C.stoneLight);
      }
  };
  for (let b = 0; b < bays; b++) {
    const lx = Math.round(nx0 + (b + 0.5) * bayW) - 2;
    lancet(lx, ny1 - 1, 1, b);
    lancet(lx, ny0, -1, b + 2);
  }
  gableRoof(m, nx0, nx1, ny0, ny1, nH, 4, 2, roofFill('slate', seed), stone);

  // Bell tower: quoins, arched door, clock, louvred belfry, cornice, spire.
  const tx0 = 4, tx1 = 36, ty0 = 16, ty1 = 48, tH = 88, tcx = (tx0 + tx1) / 2;
  m.box(tx0, ty0, 0, tx1, ty1, tH, (x, y, z) => {
    const corner = (x < tx0 + 2 || x >= tx1 - 2) && (y < ty0 + 2 || y >= ty1 - 2);
    return corner && Math.floor(z / 3) % 2 === 0 && z % 3 !== 0 ? dressed(x, y, z) : stone(x, y, z);
  });
  // Arched double door.
  for (let z = 1; z <= 24; z++)
    for (let x = tcx - 8; x < tcx + 8; x++) {
      const dx = Math.abs(x + 0.5 - tcx);
      const hw = z < 17 ? 5 : Math.sqrt(Math.max(0, 25 - (z - 17) ** 2));
      if (dx < hw) {
        const stud = (x % 3 === 1 && z % 4 === 2) || z === 5 || z === 13;
        const seam = Math.abs(x + 0.5 - tcx) < 0.6;
        m.set(x, ty1 - 1, z, seam ? [52, 38, 30] : stud ? C.iron : jitter(shade(C.door, x % 2 ? 0.88 : 1), 0.05, hash3(x, 0, z, seed)));
      } else if (dx < hw + 2) {
        const a = Math.atan2(z - 17, x + 0.5 - tcx);
        m.set(x, ty1 - 1, z, Math.floor(a * 4) % 2 === 0 || z < 17 ? C.stoneLight : shade(C.stoneLight, 0.88));
      }
    }
  m.box(tcx - 8, ty1, 0, tcx + 8, ty1 + 3, 1, dressed);
  // Clock face.
  for (let z = 58; z <= 72; z++)
    for (let x = tcx - 7; x < tcx + 7; x++) {
      const dd = Math.hypot(x + 0.5 - tcx, z + 0.5 - 65);
      if (dd >= 6.2) continue;
      let c: RGB = dd >= 5 ? C.iron : [214, 206, 182];
      const tick = dd >= 4 && dd < 5 && (Math.abs(x + 0.5 - tcx) < 0.6 || Math.abs(z + 0.5 - 65) < 0.6);
      const hourHand = Math.abs(x + 0.5 - tcx) < 0.6 && z >= 65 && z < 68;
      const minHand = Math.abs(z + 0.5 - 65) < 0.6 && x >= tcx && x < tcx + 4;
      if (tick || hourHand || minHand) c = C.iron;
      m.set(x, ty1 - 1, z, c);
    }
  // Belfry: two louvred, pointed openings per face.
  const belfry = (set: (u: number, z: number, c: RGB) => void) => {
    for (const u0 of [7, 21])
      for (let z = 72; z <= 85; z++)
        for (let i = -1; i <= 4; i++) {
          const half = z < 82 ? 2 : z < 84 ? 1.2 : 0.4;
          const di = Math.abs(i + 0.5 - 2);
          if (di < half) set(u0 + i, z, z % 2 ? [86, 68, 52] : [28, 24, 22]);
          else if (di < half + 1) set(u0 + i, z, C.stoneLight);
        }
  };
  belfry((u, z, c) => m.set(tx0 + u, ty1 - 1, z, c));
  belfry((u, z, c) => m.set(tx0 + u, ty0, z, c));
  belfry((u, z, c) => m.set(tx0, ty0 + u, z, c));
  belfry((u, z, c) => m.set(tx1 - 1, ty0 + u, z, c));
  m.box(tx0 - 1, ty0 - 1, tH - 2, tx1 + 1, ty1 + 1, tH, dressed);
  let z = tH;
  for (let inset = 0; tx0 + inset < tx1 - inset; inset++)
    for (let r = 0; r < 2; r++, z++)
      m.box(tx0 + inset, ty0 + inset, z, tx1 - inset, ty1 - inset, z + 1, (x, y, zz) => {
        const hip = Math.abs(Math.abs(x + 0.5 - tcx) - Math.abs(y + 0.5 - (ty0 + ty1) / 2)) < 1;
        return jitter(shade(hip ? C.slateLight : C.slate, zz % 3 === 0 ? 0.86 : 1), 0.04, hash3(x, y, zz, seed));
      });
  for (let k = 0; k < 12; k++) m.set(tcx - 1, (ty0 + ty1) / 2 - 1, z + k, C.gold);
  for (let i = -3; i <= 3; i++) m.set(tcx - 1 + i, (ty0 + ty1) / 2 - 1, z + 8, C.gold);
  return m;
}

// ── Organic shapes ───────────────────────────────────────────────────────

/**
 * Ellipsoid lump. Overlapping lumps compete per voxel (the one whose surface is
 * nearest wins), and each lump writes its own analytic normal so it shades as a
 * distinct rounded clump with a lit top and darker underside.
 */
function blob(
  m: VoxelModel, cx: number, cy: number, cz: number, rx: number, ry: number, rz: number,
  seed: number, fill: Fill, mat: number = MAT.SOLID, rough = 0.22, upBias = 0.25,
) {
  for (let z = Math.max(0, Math.floor(cz - rz)); z <= cz + rz; z++)
    for (let y = Math.floor(cy - ry); y <= cy + ry; y++)
      for (let x = Math.floor(cx - rx); x <= cx + rx; x++) {
        const ex = (x + 0.5 - cx) / rx, ey = (y + 0.5 - cy) / ry, ez = (z + 0.5 - cz) / rz;
        const d = Math.hypot(ex, ey, ez);
        if (d >= 1 - rough * hash3(x, y, z, seed)) continue;
        if (!m.claim(x, y, z, d)) continue;
        const c = fill(x, y, z);
        if (!c) continue;
        m.set(x, y, z, c, mat);
        m.setNormal(x, y, z, ex / rx, ey / ry, ez / rz + upBias / Math.max(rx, rz));
      }
}

/** Leaf colour for a clump: height picks the tone band; small clusters, glints and gaps add texture. */
function leafFill(dark: RGB, light: RGB, t: number, seed: number): Fill {
  return (x, y, z) => {
    let c = mix(dark, light, clamp01(t + (hash3(x >> 1, y >> 1, z >> 1, seed) - 0.5) * 0.12));
    const f = hash3(x, y, z, seed + 7);
    if (f > 0.94) c = shade(c, 1.1);
    else if (f < 0.05) c = shade(c, 0.86);
    return jitter(c, 0.02, hash3(x, y, z, seed + 1));
  };
}

const barkFill = (seed: number, base: RGB = C.bark): Fill => (x, y, z) => {
  const groove = hash3(x, y, 0, seed) > 0.68;
  const knot = hash3(x >> 1, y >> 1, z >> 2, seed + 3) > 0.96;
  return jitter(shade(base, groove ? 0.78 : knot ? 0.86 : 1), 0.03, hash3(x, y, z, seed));
};

export function oak(seed: number): VoxelModel {
  const rng = mulberry32(seed);
  const m = new VoxelModel(76, 76, 88);
  const c = 38;
  const bark = barkFill(seed);
  for (let z = 0; z < 52; z++) disc(m, c, c, z, 3.3 + (z < 6 ? (6 - z) * 0.5 : 0) - z * 0.012, bark);
  for (let i = 0, n = irange(rng, 4, 5); i < n; i++) {
    const a = (i / n) * Math.PI * 2 + range(rng, -0.4, 0.4), len = range(rng, 6, 9);
    line(m, [c, c, 2.5], [c + Math.cos(a) * len, c + Math.sin(a) * len, 0.5], 1.3, bark);
  }
  for (let i = 0, n = irange(rng, 3, 4); i < n; i++) {
    const a = rng() * Math.PI * 2;
    line(m, [c, c, range(rng, 28, 36)], [c + Math.cos(a) * 15, c + Math.sin(a) * 15, 47], 1.4, bark);
  }
  const dark: RGB = [50, 82, 58], light: RGB = [112, 140, 68];
  const lump = (x: number, y: number, z: number, r: number, rz: number, i: number) =>
    blob(m, x, y, z, r, r * range(rng, 0.9, 1.05), rz, seed + i, leafFill(dark, light, (z / S - 21) / 14 + range(rng, -0.06, 0.06), seed + i), MAT.FOLIAGE, 0.16);
  const n = irange(rng, 6, 8);
  for (let i = 0; i < n; i++) {
    const a = (i / n) * Math.PI * 2 + range(rng, -0.3, 0.3), d = range(rng, 14, 20);
    lump(c + Math.cos(a) * d, c + Math.sin(a) * d, range(rng, 44, 54), range(rng, 12, 15), range(rng, 10, 12), i);
  }
  lump(c, c, 60, range(rng, 17, 20), 13, 20);
  for (let i = 0; i < 3; i++) {
    const a = rng() * Math.PI * 2;
    lump(c + Math.cos(a) * 10, c + Math.sin(a) * 10, range(rng, 66, 72), range(rng, 10, 13), 9, 30 + i);
  }
  return m;
}

export function pine(seed: number): VoxelModel {
  const m = new VoxelModel(44, 44, 96);
  const c = 22;
  m.box(c - 2, c - 2, 0, c + 2, c + 2, 84, barkFill(seed));
  const dark: RGB = [46, 82, 62], light: RGB = [84, 122, 74];
  for (let z = 12; z < 92; z++) {
    const t = (z - 12) / 80;
    const tier = ((z - 12) % 18) / 18;
    const r = (1 - t) * 20 * (1 - 0.4 * tier) + 1.2;
    for (let y = 0; y < 44; y++)
      for (let x = 0; x < 44; x++) {
        const dx = x + 0.5 - c, dy = y + 0.5 - c;
        const dist = Math.hypot(dx, dy);
        if (dist >= r + hash3(x, y, z, seed) * 1.6 - 0.8) continue;
        // Each tier is a skirt: darker underneath, lighter at the top edge; radial needle streaks.
        const streak = 0.92 + 0.12 * hash3(Math.floor((Math.atan2(dy, dx) + Math.PI) * 5), Math.floor(z / 6), 2, seed);
        const col = shade(mix(dark, light, tier * 0.7 + t * 0.3), streak);
        m.set(x, y, z, jitter(col, 0.03, hash3(x >> 1, y >> 1, z >> 1, seed + 1)), MAT.FOLIAGE);
        m.setNormal(x, y, z, dx, dy, (dist + 2) * 0.9);
      }
  }
  return m;
}

export function bush(seed: number): VoxelModel {
  const rng = mulberry32(seed);
  const m = new VoxelModel(32, 32, 20);
  const dark: RGB = [64, 98, 60], light: RGB = [110, 140, 70];
  const flowers = seed % 3 === 0;
  const n = irange(rng, 2, 4);
  for (let i = 0; i < n; i++) {
    const leaves = leafFill(dark, light, 0.35 + i * 0.12, seed + i);
    blob(m, 16 + range(rng, -6, 6), 16 + range(rng, -6, 6), range(rng, 4, 8), range(rng, 7, 10), range(rng, 7, 10), range(rng, 7, 10), seed + i,
      (x, y, z) => (flowers && hash3(x, y, z, seed + 9) > 0.965 ? [214, 206, 188] : leaves(x, y, z)), MAT.FOLIAGE, 0.18);
  }
  return m;
}

export function rock(seed: number): VoxelModel {
  const m = new VoxelModel(22, 20, 14);
  const light: RGB = [140, 144, 140], dark: RGB = [104, 110, 112];
  blob(m, 11, 10, 1, 9.6, 8, 11, seed, (x, y, z) => {
    if (z >= 8 && hash3(x >> 1, y >> 1, z, seed + 5) > 0.5) return jitter(mix(C.moss, C.mossLight, hash3(x, y, z, seed + 6)), 0.04, hash3(x, y, z, seed));
    if (hash3(x, y, 0, seed + 4) > 0.94) return shade(dark, 0.82); // cracks
    return jitter(mix(dark, light, hash3(x >> 2, y >> 2, z >> 1, seed) * 0.5 + z * 0.04), 0.03, hash3(x, y, z, seed));
  }, MAT.SOLID, 0.1, 0.4);
  return m;
}

/** A clump of blades: reeds (some with cattails) at the water's edge, or tall meadow grass. */
export function blades(seed: number, kind: 'reed' | 'grass'): VoxelModel {
  const rng = mulberry32(seed);
  const reed = kind === 'reed';
  const m = new VoxelModel(28, 28, reed ? 32 : 18);
  const base: RGB = reed ? [86, 104, 54] : [52, 84, 60];
  const tip: RGB = reed ? [170, 172, 86] : [78, 112, 66];
  const n = reed ? irange(rng, 28, 40) : irange(rng, 9, 14);
  for (let i = 0; i < n; i++) {
    // Reeds stand in a loose clump; meadow tufts fan out from one root, like a painted "W".
    const a = reed ? rng() * Math.PI * 2 : (i / n) * Math.PI * 2 + range(rng, -0.3, 0.3);
    const r = reed ? Math.sqrt(rng()) * 8.4 : rng() * 1.6;
    const bx = 14 + Math.cos(a) * r, by = 14 + Math.sin(a) * r;
    const h = reed ? irange(rng, 14, 30) : irange(rng, 6, 14);
    const lean = reed ? range(rng, 1.6, 4.4) : range(rng, 4, 7);
    const lx = Math.cos(a) * lean, ly = Math.sin(a) * lean;
    const cattail = reed && h > 22 && rng() < 0.35;
    const seedHead = !reed && rng() < 0.3;
    for (let z = 0; z < h; z++) {
      const f = reed ? (z / h) ** 2 : z / Math.max(1, h - 1);
      const x = Math.floor(bx + lx * f), y = Math.floor(by + ly * f);
      let c = mix(base, tip, (z / (h - 1)) * (0.7 + 0.3 * hash3(i, 0, 0, seed)));
      if (cattail && z >= h - 9 && z < h - 3) {
        c = z % 2 ? [110, 76, 50] : [94, 64, 44];
        m.set(x + 1, y, z, c, MAT.FOLIAGE);
        m.setNormal(x + 1, y, z, 1, 0.2, 0.6);
      }
      if (seedHead && z === h - 1) c = [156, 150, 98];
      m.set(x, y, z, c, MAT.FOLIAGE);
      m.setNormal(x, y, z, lx * 0.3, ly * 0.3 + 0.2, 1);
    }
  }
  return m;
}

export function haystack(seed: number): VoxelModel {
  const m = new VoxelModel(24, 20, 16);
  blob(m, 12, 10, 0, 11.2, 9.2, 15.2, seed, (x, y, z) => {
    const a = Math.atan2(y + 0.5 - 10, x + 0.5 - 12);
    const strand = hash3(Math.floor((a + Math.PI) * 9), Math.floor(z / 4), 0, seed);
    if (z === 7 && hash3(x, y, 1, seed) > 0.2) return C.rope;
    return jitter(shade(C.straw, (z % 3 === 0 ? 0.93 : 1) * (0.92 + 0.12 * strand)), 0.04, hash3(x, y, z, seed));
  }, MAT.SOLID, 0.1, 0.3);
  return m;
}

// ── Props ────────────────────────────────────────────────────────────────

export function well(seed: number): VoxelModel {
  const m = new VoxelModel(26, 26, 40);
  const c = 13;
  for (let z = 0; z < 10; z++)
    for (let y = 0; y < 26; y++)
      for (let x = 0; x < 26; x++) {
        const d = Math.hypot(x + 0.5 - c, y + 0.5 - c);
        const n = hash3(x, y, z, seed);
        if (d < 12.4 && d >= 7.2) {
          const row = Math.floor(z / 3);
          const seg = ((Math.atan2(y + 0.5 - c, x + 0.5 - c) + Math.PI) / (Math.PI * 2)) * 18 + (row % 2) * 0.5;
          if (z === 9) m.set(x, y, z, jitter(C.stoneLight, 0.04, n));
          else if (z % 3 === 2 || seg % 1 < 0.1) m.set(x, y, z, jitter(C.mortar, 0.04, n));
          else m.set(x, y, z, jitter(mix(C.stone, C.stoneDark, hash3(Math.floor(seg), row, 1, seed) * 0.7), 0.04, n));
        } else if (d < 7.2 && z === 4) m.set(x, y, z, jitter(C.water, 0.04, n), MAT.WATER);
      }
  const wood = (x: number, y: number, z: number) => jitter(C.wood, 0.06, hash3(x, y >> 2, z >> 2, seed));
  m.box(2, 12, 10, 4, 14, 30, wood);
  m.box(22, 12, 10, 24, 14, 30, wood);
  m.box(2, 12, 24, 24, 14, 26, C.woodDark);
  m.box(24, 12, 25, 26, 13, 26, C.iron);
  m.box(25, 12, 21, 26, 13, 25, C.iron);
  for (let z = 15; z < 24; z++) m.set(12, 12, z, C.rope);
  for (let z = 11; z < 16; z++)
    disc(m, 12.5, 12.5, z, 2.7, (x, y, zz) => {
      if (zz === 15 && Math.hypot(x + 0.5 - 12.5, y + 0.5 - 12.5) < 1.6) return null;
      return zz === 13 ? C.iron : shade(C.wood, (x + y) % 2 ? 0.9 : 1);
    });
  const tile = roofFill('tile', seed);
  for (let k = 0; k < 11; k++)
    m.box(0, 2 + k, 29 + k, 26, 24 - k, 30 + k, (x, y, z) => tile(x, y, z, k, 10, x === 0 || x === 25));
  return m;
}

export function stall(seed: number, cloth: RGB): VoxelModel {
  const rng = mulberry32(seed);
  const m = new VoxelModel(36, 24, 30);
  for (const [x, y] of [[2, 2], [32, 2], [2, 20], [32, 20]]) m.box(x, y, 0, x + 2, y + 2, 27, (xx, yy, z) => jitter(C.wood, 0.05, hash3(xx, yy, z >> 2, seed)));
  m.box(3, 12, 0, 33, 22, 10, (x, y, z) => {
    const n = hash3(x, y, z, seed);
    if (z === 9) return jitter(shade(C.plank, y % 3 === 0 ? 0.86 : 1.08), 0.04, n);
    return jitter(shade(C.plank, x % 4 === 0 ? 0.78 : z === 8 ? 1.0 : 0.94), 0.05, n);
  });
  // Wares along the counter.
  for (let sx = 5; sx < 30; sx += 5) {
    const kind = irange(rng, 0, 4), cx = sx + 1.5, cy = 16.5;
    if (kind === 0) {
      const apple: RGB = pick(rng, [[168, 70, 60], [176, 130, 60], [120, 140, 64]]);
      for (let z = 10; z < 13; z++) disc(m, cx, cy, z, 2.8 - (z - 10) * 0.9, (x, y) => shade(apple, hash3(x, y, z, seed) > 0.7 ? 1.15 : 1));
    } else if (kind === 1) {
      for (const oy of [-1.5, 1.5]) m.box(sx, Math.floor(cy + oy) - 1, 10, sx + 4, Math.floor(cy + oy) + 1, 12, (x, _y, z) => (z === 11 && x % 2 ? [214, 176, 116] : [190, 142, 90]));
    } else if (kind === 2) {
      blob(m, cx, cy, 10.5, 2.2, 2.2, 2, seed + sx, (x, y, z) => (hash3(x, y, z, seed) > 0.6 ? [124, 154, 82] : [100, 138, 72]), MAT.FOLIAGE, 0.1);
    } else if (kind === 3) {
      for (let z = 10; z < 15; z++) disc(m, cx, cy, z, z === 14 ? 1.6 : 2.1, (x, y) => (z === 14 && Math.hypot(x + 0.5 - cx, y + 0.5 - cy) < 1 ? [52, 34, 28] : z === 12 ? [128, 76, 54] : [150, 92, 64]));
    } else {
      for (let z = 10; z < 12; z++) disc(m, cx, cy, z, 2.3, () => (z === 11 ? [222, 194, 110] : [204, 174, 96]));
    }
  }
  // Striped awning sloping towards the front, with a scalloped valance.
  for (let y = 0; y < 24; y++)
    for (let x = 0; x < 36; x++) {
      const z = 28 - Math.floor(y / 8);
      const c = Math.floor(x / 6) % 2 ? cloth : ([236, 228, 210] as RGB);
      m.set(x, y, z, jitter(shade(c, x % 6 === 5 ? 0.9 : 1), 0.03, hash3(x, y, 0, seed)));
      if (y === 23) {
        const p = x % 6;
        if (p >= 1 && p <= 4) m.set(x, y, z - 1, shade(c, 0.86));
        if (p >= 2 && p <= 3) m.set(x, y, z - 2, shade(c, 0.8));
      }
    }
  return m;
}

export function barrel(seed: number): VoxelModel {
  const m = new VoxelModel(12, 12, 14);
  for (let z = 0; z < 14; z++) {
    const r = 4.6 + 0.9 * Math.sin((Math.PI * (z + 0.5)) / 14);
    for (let y = 0; y < 12; y++)
      for (let x = 0; x < 12; x++) {
        const dx = x + 0.5 - 6, dy = y + 0.5 - 6, d = Math.hypot(dx, dy);
        if (d >= r) continue;
        let c: RGB;
        if (z === 13) c = d > r - 1.3 ? C.woodDark : shade(C.wood, Math.floor(x / 2) % 2 ? 0.88 : 0.96);
        else if (z === 2 || z === 11) c = z === 2 ? C.iron : shade(C.iron, 1.15);
        else {
          const st = ((Math.atan2(dy, dx) + Math.PI) / (Math.PI * 2)) * 14;
          c = st % 1 < 0.12 ? shade(C.wood, 0.76) : jitter(shade(C.wood, Math.floor(st) % 2 ? 0.92 : 1), 0.04, hash3(Math.floor(st), z >> 2, 0, seed));
        }
        m.set(x, y, z, c);
      }
  }
  return m;
}

export function crate(seed: number): VoxelModel {
  const m = new VoxelModel(12, 12, 12);
  m.box(0, 0, 0, 12, 12, 12, (x, y, z) => {
    const n = hash3(x, y, z, seed);
    const ex = x < 2 || x > 9, ey = y < 2 || y > 9, ez = z < 2 || z > 9;
    if (+ex + +ey + +ez >= 2) return jitter(C.woodDark, 0.05, n);
    if (z === 11) return x % 3 === 0 ? shade(C.plank, 0.8) : jitter(C.plank, 0.06, n);
    const u = x === 0 || x === 11 ? y : x;
    if (Math.abs(u - z) <= 1) return jitter(shade(C.woodDark, 1.12), 0.05, n);
    if (z % 3 === 2) return shade(C.plank, 0.8);
    return jitter(C.plank, 0.06, n);
  });
  return m;
}

export function cart(seed: number): VoxelModel {
  const m = new VoxelModel(34, 18, 18);
  const plank = (x: number, y: number, z: number) => jitter(shade(C.plank, x % 6 === 0 ? 0.8 : 1), 0.05, hash3(x, y, z, seed));
  m.box(4, 2, 6, 28, 16, 9, plank);
  m.box(4, 2, 9, 28, 3, 13, plank);
  m.box(4, 15, 9, 28, 16, 13, plank);
  m.box(4, 2, 9, 5, 16, 13, C.wood);
  m.box(27, 2, 9, 28, 16, 13, C.wood);
  m.box(4, 2, 12, 28, 3, 13, C.woodDark);
  m.box(4, 15, 12, 28, 16, 13, C.woodDark);
  for (const y of [0, 1, 16, 17])
    for (let z = 0; z < 16; z++)
      for (let x = 8; x < 24; x++) {
        const dx = x + 0.5 - 16, dz = z + 0.5 - 8, d = Math.hypot(dx, dz);
        if (d >= 7.8) continue;
        if (d >= 7.0) m.set(x, y, z, C.iron);
        else if (d >= 5.8) m.set(x, y, z, C.woodDark);
        else if (d < 1.6) m.set(x, y, z, d < 0.9 ? C.iron : C.woodDark);
        else {
          const a = Math.atan2(dz, dx), seg = Math.PI / 4;
          if (Math.abs(d * Math.sin(a - Math.round(a / seg) * seg)) < 0.7) m.set(x, y, z, C.wood);
        }
      }
  m.box(28, 4, 8, 34, 5, 9, C.wood);
  m.box(28, 13, 8, 34, 14, 9, C.wood);
  const sack: Fill = (x, y, z) => jitter(shade([176, 156, 120], (x + z) % 3 === 0 ? 0.9 : 1), 0.04, hash3(x, y, z, seed));
  blob(m, 10, 7, 10, 3.6, 3.2, 3.4, seed, sack, MAT.SOLID, 0.05, 0.3);
  blob(m, 10, 12, 10, 3.4, 3, 3.2, seed + 1, sack, MAT.SOLID, 0.05, 0.3);
  blob(m, 20, 9, 9, 6.5, 5.5, 4.5, seed + 2, (x, y, z) => jitter(shade(C.straw, hash3(x, y, 0, seed) > 0.6 ? 0.9 : 1), 0.08, hash3(x, y, z, seed)), MAT.SOLID, 0.25, 0.3);
  return m;
}

export function fence(seed: number): VoxelModel {
  const m = new VoxelModel(32, 4, 12);
  const grain = (x: number, y: number, z: number) => jitter(shade(C.wood, hash3(x >> 2, 0, z, seed) > 0.7 ? 0.88 : 1), 0.06, hash3(x, y, z, seed));
  for (const x of [0, 30]) {
    m.box(x, 1, 0, x + 2, 3, 11, (xx, yy, z) => jitter(C.woodDark, 0.05, hash3(xx, yy, z, seed)));
    m.set(x, 1, 11, C.woodDark);
    m.set(x + 1, 2, 11, C.woodDark);
  }
  for (const z of [4, 8]) m.box(1, 2, z, 31, 3, z + 2, grain);
  return m;
}

function flame(m: VoxelModel, cx: number, cy: number, z0: number, height: number, radius: number, seed: number) {
  for (let z = z0; z < z0 + height; z++) {
    const r = radius * (1 - (z - z0) / height) + 0.4;
    for (let y = 0; y < m.d; y++)
      for (let x = 0; x < m.w; x++) {
        const d = Math.hypot(x + 0.5 - cx, y + 0.5 - cy);
        if (d < r) m.set(x, y, z, mix(C.fire1, C.fire2, Math.min(1, d / r + hash3(x, y, z, seed) * 0.3)), MAT.FIRE);
      }
  }
}

export function torchPost(seed: number): VoxelModel {
  const m = new VoxelModel(8, 8, 32);
  m.box(3, 3, 0, 5, 5, 23, (x, y, z) => jitter(C.wood, 0.08, hash3(x, y, z >> 1, seed)));
  for (let z = 22; z < 26; z++)
    disc(m, 4, 4, z, 3.4, (x, y) => (z === 22 || Math.hypot(x + 0.5 - 4, y + 0.5 - 4) > 2.3 ? C.iron : null));
  flame(m, 4, 4, 23, 9, 2.6, seed);
  return m;
}

/** Small iron lantern standing on the ground: amber panes (glow at night), a candle, a carry ring. */
export function lantern(seed: number): VoxelModel {
  const m = new VoxelModel(8, 8, 15);
  const iron = (x: number, y: number, z: number) => jitter(C.iron, 0.05, hash3(x, y, z, seed));
  m.box(1, 1, 0, 7, 7, 2, iron); // base
  for (let z = 2; z < 9; z++)
    for (let y = 1; y < 7; y++)
      for (let x = 1; x < 7; x++) {
        const shell = x === 1 || x === 6 || y === 1 || y === 6;
        if (!shell) continue;
        const corner = (x === 1 || x === 6) && (y === 1 || y === 6);
        const bar = z === 5 && !corner;
        if (corner || bar) m.set(x, y, z, iron(x, y, z));
        else m.set(x, y, z, z === 8 ? [236, 196, 120] : [214, 160, 84], MAT.WINDOW);
      }
  m.box(3, 3, 2, 5, 5, 4, [226, 214, 186]); // candle
  m.set(3, 3, 4, C.fire1, MAT.FIRE);
  m.set(4, 4, 5, C.fire2, MAT.FIRE);
  for (let z = 9; z < 12; z++) disc(m, 4, 4, z, 3.6 - (z - 9) * 1.1, iron); // pyramid hood
  m.set(3, 3, 12, C.iron);
  for (const [x, z] of [[2, 13], [5, 13], [3, 14], [4, 14]]) m.set(x, 3, z, C.iron); // ring
  return m;
}

export function campfire(seed: number): VoxelModel {
  const rng = mulberry32(seed);
  const m = new VoxelModel(22, 22, 14);
  for (let k = 0; k < 11; k++) {
    const a = (k / 11) * Math.PI * 2 + range(rng, -0.1, 0.1);
    blob(m, 11 + Math.cos(a) * 8.6, 11 + Math.sin(a) * 8.6, 0.5, range(rng, 1.5, 2.1), range(rng, 1.5, 2.1), 1.9, seed + k,
      (x, y, z) => jitter(mix(C.stone, C.stoneDark, hash3(x, y, z, seed)), 0.05, hash3(x, y, z, seed + 1)), MAT.SOLID, 0.1, 0.4);
  }
  for (let y = 4; y < 18; y++)
    for (let x = 4; x < 18; x++)
      if (Math.hypot(x + 0.5 - 11, y + 0.5 - 11) < 5.5) m.set(x, y, 0, hash3(x, y, 0, seed) > 0.55 ? C.ember : [52, 40, 36], hash3(x, y, 0, seed) > 0.55 ? MAT.FIRE : MAT.SOLID);
  const bark = barkFill(seed);
  for (let k = 0; k < 4; k++) {
    const a = (k / 4) * Math.PI * 2 + 0.4;
    line(m, [11 + Math.cos(a) * 6, 11 + Math.sin(a) * 6, 1], [11 + Math.cos(a) * 1.5, 11 + Math.sin(a) * 1.5, 7], 1.0, bark);
  }
  flame(m, 11, 11, 2, 12, 4.6, seed);
  return m;
}

// ── People ───────────────────────────────────────────────────────────────

export interface VillagerLook {
  tunic: RGB;
  pants: RGB;
  hair: RGB;
  skin: RGB;
  dress: boolean;
  hat?: boolean;
}

const HAT: RGB = [196, 168, 108], HAT_BAND: RGB = [118, 82, 60];
const EYE: RGB = [40, 32, 30];

/** Wide-brimmed straw hat centred on (cx, cy), brim at layer z0. */
function strawHat(m: VoxelModel, cx: number, cy: number, z0: number) {
  for (let y = Math.floor(cy - 6); y <= cy + 6; y++)
    for (let x = Math.floor(cx - 8); x <= cx + 8; x++) {
      const e = ((x + 0.5 - cx) / 7) ** 2 + ((y + 0.5 - cy) / 5.6) ** 2;
      if (e < 1) m.set(x, y, z0, shade(HAT, e > 0.7 ? 0.9 : (x + y) % 2 ? 0.95 : 1));
    }
  disc(m, cx, cy, z0 + 1, 3.3, () => HAT_BAND);
  disc(m, cx, cy, z0 + 2, 3.2, (x, y) => shade(HAT, (x + y) % 2 ? 0.95 : 1));
  disc(m, cx, cy, z0 + 3, 2.6, () => HAT);
}

/** frame: 0 = neutral, 1 = left leg forward, 2 = right leg forward. Faces +y. */
export function villager(look: VillagerLook, frame: 0 | 1 | 2): VoxelModel {
  // Body x 6..13, y 3..8; face on y = 8. Margins leave room for a hat brim.
  const m = new VoxelModel(20, 12, 34);
  const swing = frame === 0 ? 0 : frame === 1 ? 2 : -2;
  const boots: RGB = [64, 50, 40];
  const tunic = look.tunic;

  if (look.dress) {
    for (let z = 0; z < 11; z++) {
      const rx = 4.2 + (10 - z) * 0.12, ry = 3.1 + (10 - z) * 0.08;
      for (let y = 0; y < 12; y++)
        for (let x = 0; x < 20; x++) {
          const ex = Math.abs(x + 0.5 - 10) / rx, ey = Math.abs(y + 0.5 - 6) / ry;
          if (ex > 1 || ey > 1 || ex + ey > 1.75) continue;
          m.set(x, y, z, z <= 1 ? shade(tunic, 0.74) : shade(tunic, x % 3 === 0 ? 0.9 : 1));
        }
    }
    m.set(8, 9, 0, boots);
    m.set(11, 9, 0, boots);
  } else {
    for (const [lx, s] of [[7, swing], [10, -swing]] as const)
      for (let z = 0; z < 10; z++) {
        const off = z < 6 ? s : z < 8 ? Math.trunc(s / 2) : 0;
        m.box(lx, 4 + off, z, lx + 3, 8 + off, z + 1, z < 3 ? boots : z === 9 ? shade(look.pants, 0.85) : look.pants);
        if (z < 2) m.box(lx, 8 + off, z, lx + 3, 9 + off, z + 1, boots); // toe
      }
  }

  // Torso with belt and buckle; laced bodice for dresses, V neck for tunics.
  m.box(6, 3, 10, 14, 9, 21, (x, y, z) => {
    if ((x === 6 || x === 13) && (y === 3 || y === 8)) return null;
    if (z === 20 && (x === 6 || x === 13)) return null;
    if (z < 12) return y === 8 && (x === 9 || x === 10) ? C.brass : C.leather;
    if (y === 8 && look.dress && (x === 9 || x === 10) && z < 19 && z % 2 === 0) return shade(tunic, 0.6);
    if (y === 8 && !look.dress && z >= 18 && Math.abs(x + 0.5 - 10) < z - 17) return look.skin;
    return shade(tunic, x % 4 === 0 ? 0.93 : 1);
  });
  // Arms: sleeve, cuff, hand; they swing opposite to the legs.
  for (const [ax, s] of [[4, -swing], [14, swing]] as const)
    for (let z = 10; z < 21; z++) {
      const off = z < 18 ? Math.round((s * (18 - z)) / 8) : 0;
      const c = z < 12 ? look.skin : z === 12 ? shade(tunic, 0.75) : shade(tunic, 0.92);
      m.box(ax, 4 + off, z, ax + 2, 7 + off, z + 1, c);
    }
  // Neck and head with eyes, mouth shading and hair.
  m.box(8, 5, 21, 12, 8, 22, shade(look.skin, 0.9));
  m.box(7, 3, 22, 13, 9, 29, (x, y, z) => {
    if ((x === 7 || x === 12) && (y === 3 || y === 8)) return null;
    if (z === 22 && (x === 7 || x === 12 || y === 3)) return null;
    return look.skin;
  });
  m.set(8, 8, 25, EYE);
  m.set(11, 8, 25, EYE);
  m.set(9, 8, 23, shade(look.skin, 0.86));
  m.set(10, 8, 23, shade(look.skin, 0.86));
  const hair = look.hair;
  m.box(7, 3, 28, 13, 9, 30, (x, y, z) => (z === 29 && (x === 7 || x === 12) && (y === 3 || y === 8) ? null : hair));
  m.box(7, 3, look.dress ? 17 : 23, 13, 4, 28, hair);
  for (const x of [7, 12]) m.box(x, 3, look.dress ? 21 : 25, x + 1, 7, 28, hair);
  m.box(8, 8, 27, 12, 9, 28, (x) => (x === 9 ? null : hair)); // fringe with a part
  if (look.hat) strawHat(m, 10, 6, 30);
  return m;
}

/** A villager sitting on the bank with a rod, line and bobber, plus a basket. Faces +y (the water). */
export function fisher(seed: number): VoxelModel {
  const m = new VoxelModel(32, 52, 34);
  const tunic: RGB = [96, 70, 56], pants: RGB = [82, 78, 66], skin: RGB = [214, 172, 132], hair: RGB = [70, 50, 40];
  const bx = 10, by = 4;
  m.box(bx, by + 2, 0, bx + 8, by + 8, 5, pants);
  m.box(bx + 1, by + 6, 2, bx + 7, by + 14, 6, (x) => shade(pants, x === bx + 3 || x === bx + 4 ? 0.85 : 1));
  m.box(bx + 1, by + 13, 0, bx + 7, by + 16, 3, (_x, _y, z) => (z < 2 ? [64, 50, 40] : pants));
  m.box(bx, by + 2, 5, bx + 8, by + 8, 16, (x, y, z) => {
    if ((x === bx || x === bx + 7) && (y === by + 2 || y === by + 7)) return null;
    return z < 7 ? C.leather : shade(tunic, x % 4 === 0 ? 0.92 : 1);
  });
  for (const ax of [bx - 1, bx + 8]) {
    line(m, [ax + 0.5, by + 5, 14], [ax < bx ? bx + 2 : bx + 6, by + 12, 10], 1.0, () => shade(tunic, 0.92));
  }
  m.box(bx + 2, by + 11, 9, bx + 7, by + 14, 11, skin);
  m.box(bx + 2, by + 4, 16, bx + 6, by + 7, 17, shade(skin, 0.9));
  m.box(bx + 1, by + 2, 17, bx + 7, by + 8, 24, (x, y) => ((x === bx + 1 || x === bx + 6) && (y === by + 2 || y === by + 7) ? null : skin));
  m.set(bx + 2, by + 7, 20, EYE);
  m.set(bx + 5, by + 7, 20, EYE);
  m.box(bx + 1, by + 2, 18, bx + 7, by + 3, 24, hair);
  m.box(bx + 1, by + 2, 23, bx + 7, by + 8, 24, hair);
  strawHat(m, bx + 4, by + 5, 24);
  // Rod from the hands out over the water, line dropping to a bobber.
  const rod: RGB = [120, 92, 64];
  const tipY = 45, tipZ = 21;
  line(m, [bx + 6.5, by + 11.5, 10.5], [bx + 6.5, tipY + 0.5, tipZ + 0.5], 0, () => rod);
  for (let z = 3; z < tipZ; z++) m.set(bx + 6, tipY, z, [150, 152, 144]);
  m.set(bx + 6, tipY, 2, [220, 214, 196]);
  m.set(bx + 6, tipY, 1, [190, 70, 56]);
  m.set(bx + 6, tipY, 0, [190, 70, 56]);
  // Wicker basket with the catch.
  for (let z = 0; z < 6; z++)
    for (let y = 0; y < 8; y++)
      for (let x = 0; x < 8; x++) {
        const wall = x === 0 || x === 7 || y === 0 || y === 7;
        if (z === 0 || wall) m.set(x + 1, y + 12, z, shade([140, 104, 70], z === 5 ? 0.8 : (x + y + z) % 2 ? 0.84 : 1));
        else if (z === 4 && hash3(x, y, 0, seed) > 0.4) m.set(x + 1, y + 12, z, (x + y) % 3 ? [150, 160, 160] : [118, 130, 134]);
      }
  return m;
}

export const VILLAGER_LOOKS: VillagerLook[] = [
  { tunic: [138, 76, 64], pants: [92, 78, 62], hair: [76, 54, 40], skin: [218, 176, 138], dress: false },
  { tunic: [78, 102, 128], pants: [84, 76, 66], hair: [48, 40, 36], skin: [196, 150, 112], dress: false },
  { tunic: [108, 124, 78], pants: [96, 84, 64], hair: [146, 112, 70], skin: [222, 184, 146], dress: false, hat: true },
  { tunic: [150, 120, 82], pants: [76, 68, 58], hair: [40, 34, 32], skin: [172, 126, 94], dress: false },
  { tunic: [124, 80, 106], pants: [0, 0, 0], hair: [118, 76, 50], skin: [224, 186, 152], dress: true },
  { tunic: [170, 154, 120], pants: [0, 0, 0], hair: [186, 160, 112], skin: [214, 172, 134], dress: true, hat: true },
  { tunic: [84, 108, 100], pants: [0, 0, 0], hair: [66, 48, 38], skin: [186, 140, 104], dress: true },
  { tunic: [112, 92, 72], pants: [70, 66, 58], hair: [90, 66, 48], skin: [206, 162, 124], dress: false, hat: true },
];

export function randomHouseOpts(rng: Rng, seed: number): HouseOpts {
  const walls = pick(rng, ['timber', 'timber', 'timber', 'stone', 'plank'] as const);
  return {
    w: irange(rng, 16, 26),
    d: irange(rng, 12, 17),
    wallH: irange(rng, 10, 15),
    roof: walls === 'stone' ? pick(rng, ['slate', 'tile'] as const) : pick(rng, ['thatch', 'thatch', 'tile'] as const),
    walls,
    chimney: rng() < 0.7,
    seed,
  };
}
