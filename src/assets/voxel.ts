// A voxel model is the authoring representation of a sprite stack:
// layer z of the grid becomes slice z of the stack.

export type RGB = readonly [number, number, number];

/**
 * Art resolution: voxels (and therefore screen pixels at 1x zoom) per world unit.
 * World layout, object sizes and speeds are in world units; the models in
 * models.ts and the ground texture are authored on this finer grid.
 */
export const DETAIL = 2;

/** Material ids, stored in the albedo alpha channel. 0 = empty. */
export const MAT = {
  EMPTY: 0,
  WATER: 50,
  FOLIAGE: 100,
  FIRE: 150,
  WINDOW: 200,
  /** Characters: lit like SOLID, but weather (snow, puddles) leaves them alone. */
  ACTOR: 254,
  SOLID: 255,
} as const;

export type ColorSource = RGB | ((x: number, y: number, z: number) => RGB | null);

export class VoxelModel {
  readonly data: Uint8Array;
  /**
   * Optional authored normals (xyz per voxel, 0 = use occupancy-derived normal).
   * Organic shapes (canopy clumps, rocks) set these analytically so each lump
   * shades as a rounded form, the way a pixel artist would paint it.
   */
  normals: Float32Array | null = null;
  /**
   * Named points in voxel coordinates (continuous: a voxel spans [x, x+1)),
   * e.g. a house's doorstep or chimney mouth. The atlas converts them to
   * model-local world units.
   */
  anchors: Record<string, [number, number, number]> = {};
  /** Per-voxel priority used when overlapping shapes compete for a voxel. */
  private prio: Float32Array | null = null;

  constructor(
    readonly w: number,
    readonly d: number,
    readonly h: number,
  ) {
    this.data = new Uint8Array(w * d * h * 4);
  }

  inside(x: number, y: number, z: number): boolean {
    return x >= 0 && y >= 0 && z >= 0 && x < this.w && y < this.d && z < this.h;
  }

  index(x: number, y: number, z: number): number {
    return ((z * this.d + y) * this.w + x) * 4;
  }

  set(x: number, y: number, z: number, c: RGB, mat: number = MAT.SOLID): void {
    x |= 0; y |= 0; z |= 0;
    if (!this.inside(x, y, z)) return;
    const i = this.index(x, y, z);
    this.data[i] = c[0];
    this.data[i + 1] = c[1];
    this.data[i + 2] = c[2];
    this.data[i + 3] = mat;
  }

  setNormal(x: number, y: number, z: number, nx: number, ny: number, nz: number): void {
    if (!this.inside(x, y, z)) return;
    this.normals ??= new Float32Array(this.w * this.d * this.h * 3);
    const len = Math.hypot(nx, ny, nz) || 1;
    const i = ((z * this.d + y) * this.w + x) * 3;
    this.normals[i] = nx / len;
    this.normals[i + 1] = ny / len;
    this.normals[i + 2] = nz / len;
  }

  authoredNormal(x: number, y: number, z: number): [number, number, number] | null {
    if (!this.normals) return null;
    const i = ((z * this.d + y) * this.w + x) * 3;
    const n = this.normals;
    return n[i] || n[i + 1] || n[i + 2] ? [n[i], n[i + 1], n[i + 2]] : null;
  }

  /** Claim a voxel for a shape with priority p (higher wins). Returns true if the claim succeeds. */
  claim(x: number, y: number, z: number, p: number): boolean {
    if (!this.inside(x, y, z)) return false;
    this.prio ??= new Float32Array(this.w * this.d * this.h).fill(-Infinity);
    const i = (z * this.d + y) * this.w + x;
    if (p < this.prio[i]) return false;
    this.prio[i] = p;
    return true;
  }

  clear(x: number, y: number, z: number): void {
    if (this.inside(x, y, z)) this.data[this.index(x, y, z) + 3] = 0;
  }

  /** Re-tag every voxel of material `from` as `to`. */
  retag(from: number, to: number): void {
    for (let i = 3; i < this.data.length; i += 4) if (this.data[i] === from) this.data[i] = to;
  }

  solid(x: number, y: number, z: number): boolean {
    return this.inside(x, y, z) && this.data[this.index(x, y, z) + 3] !== 0;
  }

  /** Fill a half-open box [x0,x1) × [y0,y1) × [z0,z1). */
  box(x0: number, y0: number, z0: number, x1: number, y1: number, z1: number, color: ColorSource, mat: number = MAT.SOLID): void {
    for (let z = Math.max(0, z0); z < Math.min(this.h, z1); z++)
      for (let y = Math.max(0, y0); y < Math.min(this.d, y1); y++)
        for (let x = Math.max(0, x0); x < Math.min(this.w, x1); x++) {
          const c = typeof color === 'function' ? color(x, y, z) : color;
          if (c) this.set(x, y, z, c, mat);
        }
  }
}

const c8 = (v: number) => (v < 0 ? 0 : v > 255 ? 255 : Math.round(v));

export const shade = (c: RGB, f: number): RGB => [c8(c[0] * f), c8(c[1] * f), c8(c[2] * f)];

/** Multiply brightness by a factor in [1 - amt, 1 + amt] chosen by n ∈ [0,1). */
export const jitter = (c: RGB, amt: number, n: number): RGB => shade(c, 1 + (n - 0.5) * 2 * amt);

export const mix = (a: RGB, b: RGB, t: number): RGB => [
  c8(a[0] + (b[0] - a[0]) * t),
  c8(a[1] + (b[1] - a[1]) * t),
  c8(a[2] + (b[2] - a[2]) * t),
];
