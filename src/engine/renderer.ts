// Deferred sprite-stack renderer.
//
//  1. Heightmap pass  – top-down, MAX-blended: tallest surface per world texel.
//                       Static geometry is cached; only moving entities are redrawn.
//  2. G-buffer pass   – every slice as an instanced quad, depth-tested (no CPU
//                       sorting), writing albedo+material and normal+height.
//  3. Lighting pass   – per low-res pixel: ambient + sun/moon (heightfield shadow)
//                       + point lights (heightfield shadows), emissives.
//  4. Effects         – weather drawn over the lit frame, depth-tested against
//                       the G-buffer depth: chimney smoke, rain, snow, lightning.
//  5. Blit            – nearest-neighbour upscale to the canvas.
//
// Geometry and lighting only meet through the G-buffer, so the geometry path
// (slice quads) can be swapped for meshes later without touching lighting.

import type { Atlas, ModelInfo } from '../assets/atlas';
import { DETAIL } from '../assets/voxel';
import type { DayLighting } from '../world/daycycle';
import type { Placed, PointLight } from '../world/village';
import { RAIN_CARRY, SNOW_CARRY, type WeatherFrame } from '../world/weather';
import { framebuffer, program, texture, type GL, type Program } from './gl';
import * as SH from './shaders';

/** x, y, angle, slice z · atlas u, v, w, h · sway */
export const FLOATS_PER_INSTANCE = 9;
const BYTES_PER_INSTANCE = FLOATS_PER_INSTANCE * 4;
const CHUNK = 128;
/** Heightmap window: world units covered, and texels (one per voxel column). */
const HM_WORLD = 2048;
const HM_SIZE = HM_WORLD * DETAIL;
const HM_SNAP = 256;
/** Largest model footprint radius + slack; used to pad culling bounds. */
const CULL_MARGIN = 48;
/** Tallest model, in world units. Objects below the screen can poke up into it. */
const MAX_HEIGHT = 80;
/** Precipitation: particles fall from this height (world units) through a box around the view. */
const PRECIP_TOP = 90;
const RAIN_DENSITY = 0.02; // drops per square world unit at full rain
const SNOW_DENSITY = 0.03;
const MAX_PARTICLES = 20000;
/** Point-light grid cell size, world units. */
const LIGHT_CELL = 32;
/** Point-light brightness at full night. */
const LIGHT_GAIN = 1.35;

/** Smoke puffs alive per chimney, and how long each lives (seconds). */
const SMOKE_PUFFS = 20;
const SMOKE_LIFE = 7;

export interface View {
  x: number;
  y: number;
  /** Rotation around the vertical axis, radians. */
  angle: number;
  /** Camera elevation above the ground plane, radians (π/4 = classic 45° sprite stacking). */
  pitch: number;
}

export const PITCH_MIN = (25 * Math.PI) / 180;
export const PITCH_MAX = (65 * Math.PI) / 180;

export interface Bounds {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
}

export interface FrameInput {
  view: View;
  lighting: DayLighting;
  lights: PointLight[];
  time: number;
  debug: number;
  shadows: boolean;
  dynamic: { data: Float32Array; count: number };
  weather: WeatherFrame;
}

export interface FrameStats {
  instances: number;
  drawCalls: number;
  lights: number;
  heightmapRebuilt: boolean;
}

/** Append one instance per slice of `m` placed at (x, y, angle). Returns the new float offset. */
export function writeInstances(out: Float32Array, o: number, m: ModelInfo, x: number, y: number, angle: number): number {
  for (const s of m.slices) {
    out[o++] = x;
    out[o++] = y;
    out[o++] = angle;
    out[o++] = s.z;
    out[o++] = s.u;
    out[o++] = s.v;
    out[o++] = m.tw;
    out[o++] = m.td;
    out[o++] = m.sway;
  }
  return o;
}

/**
 * Returns a function that moves a world point to the nearest point that lands on
 * a whole screen pixel for this view. Applied to the camera and to anything that
 * moves, so moving sprites step a pixel at a time instead of resampling their
 * slices at a new sub-pixel phase every frame (which makes rows pop in and out).
 */
export function pixelSnapper(v: View): (x: number, y: number) => { x: number; y: number; px: number; py: number } {
  const c = Math.cos(v.angle), s = Math.sin(v.angle), k = Math.tan(v.pitch);
  return (x, y) => {
    const px = Math.round((c * x + s * y) * DETAIL), py = Math.round((-s * x + c * y) * k * DETAIL);
    const rx = px / DETAIL, ry = py / (k * DETAIL);
    return { x: c * rx - s * ry, y: s * rx + c * ry, px, py };
  };
}

/** Static instances bucketed into world chunks, stored contiguously row by row
 *  so a visible run of chunks in a row is a single draw call. */
class StaticBatch {
  readonly nx: number;
  readonly ny: number;
  readonly start: Int32Array;
  readonly count: Int32Array;
  readonly buffer: WebGLBuffer;
  readonly total: number;

  constructor(gl: GL, placed: Placed[], worldSize: number) {
    this.nx = this.ny = Math.ceil(worldSize / CHUNK);
    const buckets: Placed[][] = Array.from({ length: this.nx * this.ny }, () => []);
    for (const p of placed) {
      const cx = Math.min(this.nx - 1, Math.max(0, Math.floor(p.x / CHUNK)));
      const cy = Math.min(this.ny - 1, Math.max(0, Math.floor(p.y / CHUNK)));
      buckets[cy * this.nx + cx].push(p);
    }
    this.start = new Int32Array(buckets.length);
    this.count = new Int32Array(buckets.length);
    let total = 0;
    buckets.forEach((b, i) => {
      this.start[i] = total;
      for (const p of b) total += p.model.slices.length;
      this.count[i] = total - this.start[i];
    });
    this.total = total;
    const data = new Float32Array(total * FLOATS_PER_INSTANCE);
    let o = 0;
    for (const b of buckets) for (const p of b) o = writeInstances(data, o, p.model, p.x, p.y, p.angle);
    this.buffer = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
  }

  runs(b: Bounds, cb: (first: number, count: number) => void): void {
    const c = (v: number, n: number) => Math.min(n - 1, Math.max(0, Math.floor(v / CHUNK)));
    const x0 = c(b.minX - CULL_MARGIN, this.nx), x1 = c(b.maxX + CULL_MARGIN, this.nx);
    const y0 = c(b.minY - CULL_MARGIN, this.ny), y1 = c(b.maxY + CULL_MARGIN, this.ny);
    for (let cy = y0; cy <= y1; cy++) {
      const first = this.start[cy * this.nx + x0];
      let n = 0;
      for (let cx = x0; cx <= x1; cx++) n += this.count[cy * this.nx + cx];
      if (n) cb(first, n);
    }
  }
}

export class Renderer {
  private readonly gbufProg: Program;
  private readonly heightProg: Program;
  private readonly lightProg: Program;
  private readonly blitProg: Program;
  private readonly precipProg: Program;
  private readonly smokeProg: Program;
  private readonly boltProg: Program;
  private readonly haloProg: Program;
  private readonly smokeVao: WebGLVertexArrayObject;
  private readonly chimneyCount: number;
  private readonly boltVao: WebGLVertexArrayObject;
  private readonly boltBuf: WebGLBuffer;
  private readonly albedoTex: WebGLTexture;
  private readonly normalTex: WebGLTexture;
  private readonly groundTex: WebGLTexture;
  private readonly vao: WebGLVertexArrayObject;
  private readonly emptyVao: WebGLVertexArrayObject;
  private readonly statics: StaticBatch;
  private readonly groundBuf: WebGLBuffer;
  private readonly dynBuf: WebGLBuffer;
  private dynCapacity = 0;

  private readonly hmStatic: { tex: WebGLTexture; fb: WebGLFramebuffer };
  private readonly hmFinal: { tex: WebGLTexture; fb: WebGLFramebuffer };
  private hmOrigin: [number, number] | null = null;

  private res: [number, number] = [0, 0];
  private canvasSize: [number, number] = [0, 0];
  private scale = 1;
  private gAlbedo!: WebGLTexture;
  private gNormal!: WebGLTexture;
  private gDepth!: WebGLRenderbuffer;
  private gFb!: WebGLFramebuffer;
  private litTex!: WebGLTexture;
  private litFb!: WebGLFramebuffer;
  /** Lit colour + G-buffer depth, for the effects pass. */
  private fxFb!: WebGLFramebuffer;

  /** Point lights as GPU data + a world-space grid of per-cell light lists (rebuilt when the light array changes). */
  private lightsFor: PointLight[] | null = null;
  private lightCount = 0;
  private lightTex: WebGLTexture[] = [];
  private lightGrid: [number, number, number] = [0, 0, LIGHT_CELL];
  private lightGridSize: [number, number] = [0, 0];

  constructor(
    private readonly gl: GL,
    private readonly atlas: Atlas,
    private readonly worldSize: number,
    ground: Uint8Array,
    placed: Placed[],
    chimneys: [number, number, number][],
  ) {
    this.gbufProg = program(gl, SH.GBUFFER_VS, SH.GBUFFER_FS);
    this.heightProg = program(gl, SH.HEIGHT_VS, SH.HEIGHT_FS);
    this.lightProg = program(gl, SH.FULLSCREEN_VS, SH.LIGHTING_FS);
    this.blitProg = program(gl, SH.FULLSCREEN_VS, SH.BLIT_FS);
    this.precipProg = program(gl, SH.PRECIP_VS, SH.PRECIP_FS);
    this.smokeProg = program(gl, SH.SMOKE_VS, SH.SMOKE_FS);
    this.boltProg = program(gl, SH.BOLT_VS, SH.BOLT_FS);
    this.haloProg = program(gl, SH.HALO_VS, SH.HALO_FS);

    this.albedoTex = texture(gl, atlas.width, atlas.height, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, atlas.albedo);
    this.normalTex = texture(gl, atlas.width, atlas.height, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, atlas.normal);
    this.groundTex = texture(gl, worldSize * DETAIL, worldSize * DETAIL, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, ground);

    this.vao = gl.createVertexArray()!;
    gl.bindVertexArray(this.vao);
    const quad = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, quad);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([0, 0, 1, 0, 0, 1, 1, 1]), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    for (const loc of [1, 2, 3]) {
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribDivisor(loc, 1);
    }
    this.emptyVao = gl.createVertexArray()!;

    this.statics = new StaticBatch(gl, placed, worldSize);
    this.groundBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.groundBuf);
    const gt = worldSize * DETAIL; // ground texture size in texels
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([worldSize / 2, worldSize / 2, 0, 0, 0, 0, gt, gt, 0]), gl.STATIC_DRAW);
    this.dynBuf = gl.createBuffer()!;

    // Smoke: one vec4 (x, y, z, phase seed) per chimney, shared by SMOKE_PUFFS instances.
    this.chimneyCount = chimneys.length;
    this.smokeVao = gl.createVertexArray()!;
    gl.bindVertexArray(this.smokeVao);
    const chimBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, chimBuf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(chimneys.flatMap(([x, y, z], i) => [x, y, z, ((i * 0.618034) % 1)])), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 4, gl.FLOAT, false, 0, 0);
    gl.vertexAttribDivisor(0, SMOKE_PUFFS);

    this.boltVao = gl.createVertexArray()!;
    gl.bindVertexArray(this.boltVao);
    this.boltBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.boltBuf);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);

    const hm = () => {
      // r: top of the column, g: 1 - bottom (see HEIGHT_FS). Both combine with MAX blending.
      const tex = texture(gl, HM_SIZE, HM_SIZE, gl.RG8, gl.RG, gl.UNSIGNED_BYTE);
      return { tex, fb: framebuffer(gl, [tex]) };
    };
    this.hmStatic = hm();
    this.hmFinal = hm();
    gl.bindVertexArray(null);
  }

  get staticInstanceCount(): number {
    return this.statics.total;
  }

  get internalResolution(): [number, number] {
    return this.res;
  }

  setSize(canvasW: number, canvasH: number, scale: number): void {
    const gl = this.gl;
    const w = Math.ceil(canvasW / scale), h = Math.ceil(canvasH / scale);
    this.canvasSize = [canvasW, canvasH];
    this.scale = scale;
    if (w === this.res[0] && h === this.res[1]) return;
    this.res = [w, h];
    if (this.gFb) {
      gl.deleteFramebuffer(this.gFb);
      gl.deleteFramebuffer(this.litFb);
      gl.deleteFramebuffer(this.fxFb);
      gl.deleteTexture(this.gAlbedo);
      gl.deleteTexture(this.gNormal);
      gl.deleteTexture(this.litTex);
      gl.deleteRenderbuffer(this.gDepth);
    }
    this.gAlbedo = texture(gl, w, h, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE);
    this.gNormal = texture(gl, w, h, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE);
    this.gDepth = gl.createRenderbuffer()!;
    gl.bindRenderbuffer(gl.RENDERBUFFER, this.gDepth);
    gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT24, w, h);
    this.gFb = framebuffer(gl, [this.gAlbedo, this.gNormal], this.gDepth);
    this.litTex = texture(gl, w, h, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE);
    this.litFb = framebuffer(gl, [this.litTex]);
    this.fxFb = framebuffer(gl, [this.litTex], this.gDepth);
  }

  /** Camera position snapped to whole pixels in screen space, to avoid shimmer while panning.
   *  Returns the snapped world position and that position in screen pixels. */
  snappedCamera(v: View): { x: number; y: number; px: number; py: number } {
    return pixelSnapper(v)(v.x, v.y);
  }

  /** World-space AABB of everything that can appear on screen. */
  visibleBounds(v: View): Bounds {
    const w = this.res[0] / DETAIL, h = this.res[1] / DETAIL; // screen size in world units
    const c = Math.cos(v.angle), s = Math.sin(v.angle), k = Math.tan(v.pitch);
    const b: Bounds = { minX: Infinity, minY: Infinity, maxX: -Infinity, maxY: -Infinity };
    for (const rx of [-w / 2, w / 2])
      for (const ry of [-h / 2 / k, (h / 2 + MAX_HEIGHT) / k]) {
        const x = v.x + c * rx - s * ry, y = v.y + s * rx + c * ry;
        b.minX = Math.min(b.minX, x); b.maxX = Math.max(b.maxX, x);
        b.minY = Math.min(b.minY, y); b.maxY = Math.max(b.maxY, y);
      }
    return b;
  }

  heightmapBounds(v: View): Bounds {
    const ox = Math.floor((v.x - HM_WORLD / 2) / HM_SNAP) * HM_SNAP;
    const oy = Math.floor((v.y - HM_WORLD / 2) / HM_SNAP) * HM_SNAP;
    return { minX: ox, minY: oy, maxX: ox + HM_WORLD, maxY: oy + HM_WORLD };
  }

  private bindInstances(buf: WebGLBuffer, first: number): void {
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.vertexAttribPointer(1, 4, gl.FLOAT, false, BYTES_PER_INSTANCE, first * BYTES_PER_INSTANCE);
    gl.vertexAttribPointer(2, 4, gl.FLOAT, false, BYTES_PER_INSTANCE, first * BYTES_PER_INSTANCE + 16);
    gl.vertexAttribPointer(3, 1, gl.FLOAT, false, BYTES_PER_INSTANCE, first * BYTES_PER_INSTANCE + 32);
  }

  private bindTex(unit: number, tex: WebGLTexture, loc: WebGLUniformLocation | undefined): void {
    const gl = this.gl;
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_2D, tex);
    if (loc) gl.uniform1i(loc, unit);
  }

  render(f: FrameInput): FrameStats {
    const gl = this.gl;
    const stats: FrameStats = { instances: 0, drawCalls: 0, lights: 0, heightmapRebuilt: false };
    const cam = this.snappedCamera(f.view);
    const view: View = { x: cam.x, y: cam.y, angle: f.view.angle, pitch: f.view.pitch };
    const tilt = Math.tan(view.pitch);
    const camRot: [number, number] = [Math.cos(view.angle), Math.sin(view.angle)];
    const vis = this.visibleBounds(view);
    const hmb = this.heightmapBounds(view);

    // Upload moving entities.
    gl.bindVertexArray(this.vao);
    if (f.dynamic.count) {
      gl.bindBuffer(gl.ARRAY_BUFFER, this.dynBuf);
      const bytes = f.dynamic.count * BYTES_PER_INSTANCE;
      if (bytes > this.dynCapacity) {
        this.dynCapacity = Math.max(bytes, this.dynCapacity * 2);
        gl.bufferData(gl.ARRAY_BUFFER, this.dynCapacity, gl.DYNAMIC_DRAW);
      }
      gl.bufferSubData(gl.ARRAY_BUFFER, 0, f.dynamic.data, 0, f.dynamic.count * FLOATS_PER_INSTANCE);
    }
    const drawDynamic = () => {
      if (!f.dynamic.count) return;
      this.bindInstances(this.dynBuf, 0);
      gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, f.dynamic.count);
      stats.drawCalls++;
      stats.instances += f.dynamic.count;
    };

    // ── 1. Heightmap ────────────────────────────────────────────────────
    gl.disable(gl.DEPTH_TEST);
    gl.enable(gl.BLEND);
    gl.blendEquation(gl.MAX);
    gl.viewport(0, 0, HM_SIZE, HM_SIZE);
    gl.useProgram(this.heightProg.prog);
    gl.uniform2f(this.heightProg.u.u_atlasSize, this.atlas.width, this.atlas.height);
    gl.uniform2f(this.heightProg.u.u_hmOrigin, hmb.minX, hmb.minY);
    gl.uniform1f(this.heightProg.u.u_hmSize, HM_WORLD);
    gl.uniform1f(this.heightProg.u.u_detail, DETAIL);
    this.bindTex(0, this.albedoTex, this.heightProg.u.u_albedo);
    if (!this.hmOrigin || this.hmOrigin[0] !== hmb.minX || this.hmOrigin[1] !== hmb.minY) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.hmStatic.fb);
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);
      this.statics.runs(hmb, (first, n) => {
        this.bindInstances(this.statics.buffer, first);
        gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, n);
      });
      this.hmOrigin = [hmb.minX, hmb.minY];
      stats.heightmapRebuilt = true;
    }
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, this.hmStatic.fb);
    gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, this.hmFinal.fb);
    gl.blitFramebuffer(0, 0, HM_SIZE, HM_SIZE, 0, 0, HM_SIZE, HM_SIZE, gl.COLOR_BUFFER_BIT, gl.NEAREST);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.hmFinal.fb);
    drawDynamic();
    gl.blendEquation(gl.FUNC_ADD);
    gl.disable(gl.BLEND);

    // ── 2. G-buffer ─────────────────────────────────────────────────────
    const [w, h] = this.res;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.gFb);
    gl.viewport(0, 0, w, h);
    gl.clearColor(0, 0, 0, 0);
    gl.clearDepth(1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LESS);
    const gp = this.gbufProg;
    gl.useProgram(gp.prog);
    gl.uniform2f(gp.u.u_cam, view.x, view.y);
    gl.uniform2f(gp.u.u_camRot, camRot[0], camRot[1]);
    gl.uniform2f(gp.u.u_res, w, h);
    gl.uniform1f(gp.u.u_tilt, tilt);
    gl.uniform1f(gp.u.u_detail, DETAIL);
    gl.uniform2fv(gp.u.u_wind, f.weather.wind);
    gl.uniform1f(gp.u.u_time, f.time);

    // Ground: a single huge "slice" at z = 0 with a flat normal.
    gl.uniform2f(gp.u.u_atlasSize, this.worldSize * DETAIL, this.worldSize * DETAIL);
    gl.uniform1i(gp.u.u_flatNormal, 1);
    this.bindTex(0, this.groundTex, gp.u.u_albedo);
    this.bindTex(1, this.normalTex, gp.u.u_normal);
    this.bindInstances(this.groundBuf, 0);
    gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, 1);
    stats.drawCalls++;

    gl.uniform2f(gp.u.u_atlasSize, this.atlas.width, this.atlas.height);
    gl.uniform1i(gp.u.u_flatNormal, 0);
    this.bindTex(0, this.albedoTex, gp.u.u_albedo);
    this.statics.runs(vis, (first, n) => {
      this.bindInstances(this.statics.buffer, first);
      gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, n);
      stats.drawCalls++;
      stats.instances += n;
    });
    drawDynamic();
    gl.disable(gl.DEPTH_TEST);

    // ── 3. Lighting ─────────────────────────────────────────────────────
    // (Before any binding: creating textures binds them to the active unit.)
    this.uploadLights(f.lights);
    gl.bindVertexArray(this.emptyVao);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.litFb);
    gl.viewport(0, 0, w, h);
    const lp = this.lightProg;
    gl.useProgram(lp.prog);
    this.bindTex(0, this.gAlbedo, lp.u.u_gAlbedo);
    this.bindTex(1, this.gNormal, lp.u.u_gNormal);
    this.bindTex(2, this.hmFinal.tex, lp.u.u_height);
    gl.uniform2f(lp.u.u_res, w, h);
    gl.uniform2f(lp.u.u_cam, view.x, view.y);
    gl.uniform2f(lp.u.u_camRot, camRot[0], camRot[1]);
    gl.uniform2i(lp.u.u_camPix, cam.px, cam.py);
    gl.uniform1f(lp.u.u_tilt, tilt);
    gl.uniform2f(lp.u.u_hmOrigin, hmb.minX, hmb.minY);
    gl.uniform1f(lp.u.u_hmSize, HM_WORLD);
    gl.uniform1f(lp.u.u_detail, DETAIL);
    const L = f.lighting;
    gl.uniform3fv(lp.u.u_lightDir, L.lightDir);
    gl.uniform3fv(lp.u.u_lightColor, L.lightColor);
    gl.uniform3fv(lp.u.u_ambient, L.ambient);
    gl.uniform3fv(lp.u.u_sky, L.sky);
    gl.uniform1f(lp.u.u_night, L.night);
    gl.uniform1f(lp.u.u_time, f.time);
    gl.uniform1i(lp.u.u_debug, f.debug);
    gl.uniform1i(lp.u.u_shadows, f.shadows ? 1 : 0);
    stats.lights = L.night < 0.01 ? 0 : this.countLights(f.lights, vis);
    this.bindTex(3, this.lightTex[0], lp.u.u_lightData);
    this.bindTex(4, this.lightTex[1], lp.u.u_lightCells);
    this.bindTex(5, this.lightTex[2], lp.u.u_lightIndex);
    gl.uniform3fv(lp.u.u_lightGrid, this.lightGrid);
    gl.uniform2iv(lp.u.u_lightGridSize, this.lightGridSize);
    gl.uniform1f(lp.u.u_lightScale, L.night < 0.01 ? 0 : L.night * LIGHT_GAIN);
    const W = f.weather;
    gl.uniform2f(lp.u.u_viewCS, Math.cos(view.pitch), Math.sin(view.pitch));
    gl.uniform1f(lp.u.u_cloud, W.cloud);
    gl.uniform2fv(lp.u.u_cloudOffset, W.cloudOffset);
    gl.uniform1f(lp.u.u_wet, W.wetness);
    gl.uniform1f(lp.u.u_rain, W.rain);
    gl.uniform1f(lp.u.u_snowCover, W.snowCover);
    gl.uniform1f(lp.u.u_fog, W.fog);
    gl.uniform1f(lp.u.u_mist, W.mist);
    gl.uniform4fv(lp.u.u_mistA, W.mistA);
    gl.uniform4fv(lp.u.u_mistB, W.mistB);
    gl.uniform2fv(lp.u.u_mistW, W.mistW);
    gl.uniform3fv(lp.u.u_fogColor, W.fogColor);
    gl.drawArrays(gl.TRIANGLES, 0, 3);

    // ── 4. Weather effects ──────────────────────────────────────────────
    this.drawEffects(f, view, cam, tilt, camRot, hmb);

    // ── 5. Upscale to canvas ────────────────────────────────────────────
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, this.canvasSize[0], this.canvasSize[1]);
    gl.useProgram(this.blitProg.prog);
    this.bindTex(0, this.litTex, this.blitProg.u.u_tex);
    gl.uniform1f(this.blitProg.u.u_scale, this.scale);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.bindVertexArray(null);
    return stats;
  }

  private drawEffects(
    f: FrameInput,
    view: View,
    cam: { px: number; py: number },
    tilt: number,
    camRot: [number, number],
    hmb: Bounds,
  ): void {
    const gl = this.gl;
    const W = f.weather, L = f.lighting;
    const [w, h] = this.res;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fxFb);
    gl.viewport(0, 0, w, h);
    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LESS);
    gl.depthMask(false);
    const common = (p: Program) => {
      gl.useProgram(p.prog);
      gl.uniform2f(p.u.u_res, w, h);
      gl.uniform1f(p.u.u_detail, DETAIL);
      gl.uniform1f(p.u.u_tilt, tilt);
      gl.uniform2f(p.u.u_camRot, camRot[0], camRot[1]);
      gl.uniform2f(p.u.u_cam, view.x, view.y);
    };
    // Particles take the scene's light (and a share of the fog colour) rather than being lit per pixel.
    const lit = (albedo: [number, number, number], direct: number, fogMix: number): [number, number, number] =>
      albedo.map((a, i) => {
        const v = a * (L.ambient[i] + L.lightColor[i] * direct * (1 - 0.85 * W.cloud));
        return v + (W.fogColor[i] - v) * Math.min(1, W.fog + 0.6 * W.mist) * fogMix;
      }) as [number, number, number];

    // Lamp haloes in fog / rain mist: additive, not depth-tested. Scaled by 0.9 to
    // match the grade the lit image has already been through.
    const veil = W.fog + 0.6 * W.mist;
    if (this.lightCount && L.night >= 0.01 && veil > 0.01) {
      const p = this.haloProg;
      common(p);
      this.bindTex(0, this.lightTex[0], p.u.u_lightData);
      gl.uniform1f(p.u.u_time, f.time);
      gl.uniform1f(p.u.u_amount, L.night * LIGHT_GAIN * veil * 0.22 * 0.9);
      gl.disable(gl.DEPTH_TEST);
      gl.enable(gl.BLEND);
      gl.blendFuncSeparate(gl.ONE, gl.ONE, gl.ZERO, gl.ONE);
      gl.bindVertexArray(this.emptyVao);
      gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, this.lightCount);
      gl.disable(gl.BLEND);
      gl.enable(gl.DEPTH_TEST);
    }

    // Smoke
    if (this.chimneyCount && W.smoke > 0.01) {
      const p = this.smokeProg;
      common(p);
      gl.uniform1i(p.u.u_K, SMOKE_PUFFS);
      gl.uniform1f(p.u.u_life, SMOKE_LIFE);
      gl.uniform1f(p.u.u_time, f.time);
      gl.uniform2fv(p.u.u_wind, W.wind);
      gl.uniform1f(p.u.u_amount, W.smoke);
      gl.uniform2i(p.u.u_camPix, cam.px, cam.py);
      // Smoke stays greyish even under the blue night ambient.
      const sc = lit([0.78, 0.77, 0.76], 0.6, 0.5);
      const sl = (sc[0] + sc[1] + sc[2]) / 3;
      gl.uniform3fv(p.u.u_color, sc.map((v) => sl + (v - sl) * 0.35));
      gl.bindVertexArray(this.smokeVao);
      gl.drawArraysInstanced(gl.POINTS, 0, 1, this.chimneyCount * SMOKE_PUFFS);
    }

    // Rain and snow
    const rain = W.rain > 0.01, snow = W.snow > 0.01;
    if (rain || snow) {
      const p = this.precipProg;
      common(p);
      gl.bindVertexArray(this.emptyVao);
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      // The box (camera-rotated frame) that covers every position a falling particle can be seen at.
      const bx = w / DETAIL + 8;
      const halfH = h / 2 / DETAIL;
      const lo = -halfH / tilt - 4, hi = (halfH + PRECIP_TOP) / tilt + 4;
      const area = bx * (hi - lo);
      const [c, s] = camRot;
      gl.uniform1f(p.u.u_time, f.time);
      gl.uniform2f(p.u.u_wind, c * W.wind[0] + s * W.wind[1], -s * W.wind[0] + c * W.wind[1]);
      const drift = (d: [number, number], carry: number): void => {
        gl.uniform2f(p.u.u_drift, c * d[0] + s * d[1], -s * d[0] + c * d[1]);
        gl.uniform1f(p.u.u_carry, carry);
      };
      gl.uniform2f(p.u.u_camR, c * view.x + s * view.y, -s * view.x + c * view.y);
      gl.uniform4f(p.u.u_box, -bx / 2, lo, bx, hi - lo);
      gl.uniform1f(p.u.u_top, PRECIP_TOP);
      this.bindTex(0, this.hmFinal.tex, p.u.u_height);
      gl.uniform2f(p.u.u_hmOrigin, hmb.minX, hmb.minY);
      gl.uniform1f(p.u.u_hmSize, HM_WORLD);
      if (rain) {
        const n = Math.min(MAX_PARTICLES, Math.round(RAIN_DENSITY * W.rain * area));
        const col = lit([0.62, 0.68, 0.76], 0.3, 0.3);
        gl.uniform1i(p.u.u_mode, 0);
        drift(W.rainDrift, RAIN_CARRY);
        gl.uniform4f(p.u.u_color, col[0] + 0.06 + W.flash * 0.5, col[1] + 0.07 + W.flash * 0.5, col[2] + 0.09 + W.flash * 0.6, 1);
        gl.drawArraysInstanced(gl.LINES, 0, 2, n);
      }
      if (snow) {
        const n = Math.min(MAX_PARTICLES, Math.round(SNOW_DENSITY * W.snow * area));
        const col = lit([0.95, 0.96, 1.0], 0.5, 0.25);
        gl.uniform1i(p.u.u_mode, 1);
        drift(W.snowDrift, SNOW_CARRY);
        gl.uniform4f(p.u.u_color, col[0] + 0.05, col[1] + 0.05, col[2] + 0.06, 1);
        gl.drawArraysInstanced(gl.POINTS, 0, 1, n);
      }
      gl.disable(gl.BLEND);
    }

    // Lightning bolt: a white core with a pale-violet edge one pixel to the side.
    if (W.bolt) {
      const p = this.boltProg;
      common(p);
      gl.bindVertexArray(this.boltVao);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.boltBuf);
      gl.bufferData(gl.ARRAY_BUFFER, W.bolt, gl.DYNAMIC_DRAW);
      const n = W.bolt.length / 3;
      gl.uniform2f(p.u.u_offset, 1, 0);
      gl.uniform4f(p.u.u_color, 0.72, 0.7, 1, 1);
      gl.drawArrays(gl.LINES, 0, n);
      gl.uniform2f(p.u.u_offset, 0, 0);
      gl.uniform4f(p.u.u_color, 1, 1, 1, 1);
      gl.drawArrays(gl.LINES, 0, n);
    }

    gl.depthMask(true);
    gl.disable(gl.DEPTH_TEST);
    gl.bindVertexArray(this.emptyVao);
  }

  /** Lights whose reach overlaps the visible bounds (for the HUD). */
  private countLights(lights: PointLight[], vis: Bounds): number {
    let n = 0;
    for (const l of lights)
      if (l.x + l.radius > vis.minX && l.x - l.radius < vis.maxX && l.y + l.radius > vis.minY && l.y - l.radius < vis.maxY) n++;
    return n;
  }

  /**
   * Upload the lights and bin them into a world-space grid: each cell lists every
   * light whose (largest, flickering) radius reaches into it. The lighting shader
   * looks up its pixel's cell and only visits that list, so there is no cap on
   * lights on screen and the cost doesn't grow with zoom.
   */
  private uploadLights(lights: PointLight[]): void {
    if (this.lightsFor === lights) return;
    const gl = this.gl;
    this.lightsFor = lights;
    for (const t of this.lightTex) gl.deleteTexture(t);
    const n = lights.length;
    const reach = (l: PointLight) => l.radius * (1 + 0.03 * l.flicker);

    const data = new Float32Array(Math.max(1, n) * 12);
    lights.forEach((l, i) => data.set([l.x, l.y, l.z, l.radius, ...l.color, l.overhang ?? 0, l.flicker, l.phase, 0, 0], i * 12));

    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const l of lights) {
      const r = reach(l);
      minX = Math.min(minX, l.x - r); minY = Math.min(minY, l.y - r);
      maxX = Math.max(maxX, l.x + r); maxY = Math.max(maxY, l.y + r);
    }
    const C = LIGHT_CELL;
    const gw = n ? Math.ceil((maxX - minX) / C) : 0, gh = n ? Math.ceil((maxY - minY) / C) : 0;
    const lists: number[][] = Array.from({ length: gw * gh }, () => []);
    lights.forEach((l, i) => {
      const r = reach(l);
      const cx0 = Math.max(0, Math.floor((l.x - r - minX) / C)), cx1 = Math.min(gw - 1, Math.floor((l.x + r - minX) / C));
      const cy0 = Math.max(0, Math.floor((l.y - r - minY) / C)), cy1 = Math.min(gh - 1, Math.floor((l.y + r - minY) / C));
      for (let cy = cy0; cy <= cy1; cy++)
        for (let cx = cx0; cx <= cx1; cx++) {
          // Keep only cells the light's disc actually touches.
          const nx = Math.max(minX + cx * C, Math.min(l.x, minX + (cx + 1) * C));
          const ny = Math.max(minY + cy * C, Math.min(l.y, minY + (cy + 1) * C));
          if (Math.hypot(nx - l.x, ny - l.y) <= r) lists[cy * gw + cx].push(i);
        }
    });
    const cells = new Int32Array(Math.max(1, gw * gh) * 2);
    const index: number[] = [];
    lists.forEach((list, c) => {
      if (list.length > SH.MAX_CELL_LIGHTS) {
        // Too crowded: keep the lights nearest the cell centre.
        console.warn(`light grid cell ${c} has ${list.length} lights; keeping ${SH.MAX_CELL_LIGHTS}`);
        const mx = minX + ((c % gw) + 0.5) * C, my = minY + (Math.floor(c / gw) + 0.5) * C;
        const d = (i: number) => Math.hypot(lights[i].x - mx, lights[i].y - my);
        list.sort((a, b) => d(a) - d(b)).length = SH.MAX_CELL_LIGHTS;
      }
      cells[c * 2] = index.length;
      cells[c * 2 + 1] = list.length;
      index.push(...list);
    });
    const iw = SH.LIGHT_INDEX_W, ih = Math.max(1, Math.ceil(index.length / iw));
    const idx = new Int32Array(iw * ih);
    idx.set(index);

    this.lightTex = [
      texture(gl, 3, Math.max(1, n), gl.RGBA32F, gl.RGBA, gl.FLOAT, data),
      texture(gl, Math.max(1, gw), Math.max(1, gh), gl.RG32I, gl.RG_INTEGER, gl.INT, cells),
      texture(gl, iw, ih, gl.R32I, gl.RED_INTEGER, gl.INT, idx),
    ];
    this.lightCount = n;
    this.lightGrid = [minX, minY, C];
    this.lightGridSize = [gw, gh];
  }
}
