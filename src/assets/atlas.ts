// Converts voxel models into the runtime sprite-stack format: every non-empty
// layer becomes one slice image packed into a shared atlas. Two atlases are
// produced in lockstep:
//   albedo: rgb = colour, a = material id (0 = empty)
//   normal: rgb = model-space surface normal (encoded), derived from occupancy
// The runtime never sees voxels — only slices — so hand-drawn or exported
// slice strips can be dropped in later by producing the same two images.

import { DETAIL, VoxelModel } from './voxel';

export interface SliceRect {
  u: number;
  v: number;
  /** Height of the slice's top surface in voxels (layer index + 1); world z = z / DETAIL. */
  z: number;
}

export interface ModelInfo {
  id: number;
  name: string;
  /** Size in world units. */
  w: number;
  d: number;
  h: number;
  /** Slice size in atlas texels (= voxels). */
  tw: number;
  td: number;
  slices: SliceRect[];
  /** Footprint radius in world units, for culling and placement. */
  radius: number;
  /** Named points (see VoxelModel.anchors) in model-local world units, before placement rotation. */
  anchors: Record<string, [number, number, number]>;
  /**
   * Wind response: slice z (voxels) is displaced by wind · sway · z², so the top
   * slice moves by wind · flex. 0 = rigid.
   */
  sway: number;
}

export interface Atlas {
  width: number;
  height: number;
  albedo: Uint8Array;
  normal: Uint8Array;
  models: ModelInfo[];
  byName: Map<string, ModelInfo>;
}

const PAD = 1;

function sliceEmpty(m: VoxelModel, z: number): boolean {
  for (let y = 0; y < m.d; y++)
    for (let x = 0; x < m.w; x++) if (m.data[m.index(x, y, z) + 3] !== 0) return false;
  return true;
}

/** Occupancy used for normal estimation: below the model counts as solid ground. */
function occ(m: VoxelModel, x: number, y: number, z: number): boolean {
  if (z < 0) return true;
  return m.solid(x, y, z);
}

function normalAt(m: VoxelModel, x: number, y: number, z: number): [number, number, number] {
  // Sum directions towards empty neighbours in a 3×3×3 neighbourhood.
  let nx = 0, ny = 0, nz = 0;
  for (let dz = -1; dz <= 1; dz++)
    for (let dy = -1; dy <= 1; dy++)
      for (let dx = -1; dx <= 1; dx++) {
        if (!dx && !dy && !dz) continue;
        if (occ(m, x + dx, y + dy, z + dz)) continue;
        const w = 1 / Math.hypot(dx, dy, dz);
        nx += dx * w;
        ny += dy * w;
        nz += dz * w;
      }
  const len = Math.hypot(nx, ny, nz);
  if (len < 1e-3) return buriedNormal(m, x, y, z);
  return [nx / len, ny / len, nz / len];
}

/**
 * A voxel with no empty neighbour can still show: a slice's edge row covers less
 * than a pixel when the camera is tilted (and rotated slices stair-step), so some
 * pixel columns sample the voxel just behind a wall face, and every slice repeats
 * the same pattern, which makes a full-height stripe. Point it at the nearest open
 * space in its own layer (2–3 voxels out) so it shades like the face in front of it.
 */
function buriedNormal(m: VoxelModel, x: number, y: number, z: number): [number, number, number] {
  for (let r = 2; r <= 3; r++) {
    let nx = 0, ny = 0;
    for (let dy = -r; dy <= r; dy++)
      for (let dx = -r; dx <= r; dx++) {
        if (Math.max(Math.abs(dx), Math.abs(dy)) !== r || occ(m, x + dx, y + dy, z)) continue;
        const w = 1 / Math.hypot(dx, dy);
        nx += dx * w;
        ny += dy * w;
      }
    const len = Math.hypot(nx, ny);
    if (len > 1e-3) return [nx / len, ny / len, 0];
  }
  return [0, 0, 1];
}

export interface AtlasEntry {
  name: string;
  model: VoxelModel;
  /** How far the top of the stack bends in the wind, relative to other models (0 = rigid). */
  flex?: number;
}

export function buildAtlas(entries: AtlasEntry[], width = 4096): Atlas {
  // Pass 1: shelf-pack every non-empty slice.
  let px = 0, py = 0, rowH = 0;
  const models: ModelInfo[] = [];
  for (const { name, model: m, flex = 0 } of entries) {
    if (m.w + PAD > width) throw new Error(`model ${name} wider than atlas`);
    const slices: SliceRect[] = [];
    for (let z = 0; z < m.h; z++) {
      if (sliceEmpty(m, z)) continue;
      if (px + m.w + PAD > width) {
        px = 0;
        py += rowH + PAD;
        rowH = 0;
      }
      slices.push({ u: px, v: py, z: z + 1 });
      px += m.w + PAD;
      rowH = Math.max(rowH, m.d);
    }
    const anchors: ModelInfo['anchors'] = {};
    for (const [k, [ax, ay, az]] of Object.entries(m.anchors)) anchors[k] = [(ax - m.w / 2) / DETAIL, (ay - m.d / 2) / DETAIL, az / DETAIL];
    const top = slices.length ? slices[slices.length - 1].z : 1;
    models.push({
      id: models.length, name, slices,
      w: m.w / DETAIL, d: m.d / DETAIL, h: m.h / DETAIL, tw: m.w, td: m.d,
      radius: Math.hypot(m.w, m.d) / 2 / DETAIL,
      anchors,
      sway: flex / (top * top),
    });
  }
  const height = py + rowH + PAD;

  // Pass 2: write colours and normals.
  const albedo = new Uint8Array(width * height * 4);
  const normal = new Uint8Array(width * height * 4);
  entries.forEach(({ model: m }, mi) => {
    for (const s of models[mi].slices) {
      const z = s.z - 1;
      for (let y = 0; y < m.d; y++)
        for (let x = 0; x < m.w; x++) {
          const src = m.index(x, y, z);
          if (m.data[src + 3] === 0) continue;
          const dst = ((s.v + y) * width + (s.u + x)) * 4;
          albedo[dst] = m.data[src];
          albedo[dst + 1] = m.data[src + 1];
          albedo[dst + 2] = m.data[src + 2];
          albedo[dst + 3] = m.data[src + 3];
          const n = m.authoredNormal(x, y, z) ?? normalAt(m, x, y, z);
          normal[dst] = Math.round((n[0] * 0.5 + 0.5) * 255);
          normal[dst + 1] = Math.round((n[1] * 0.5 + 0.5) * 255);
          normal[dst + 2] = Math.round((n[2] * 0.5 + 0.5) * 255);
          normal[dst + 3] = 255;
        }
    }
  });

  return { width, height, albedo, normal, models, byName: new Map(models.map((m) => [m.name, m])) };
}
