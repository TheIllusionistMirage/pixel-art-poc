// Deferred sprite-stack renderer.
//
//  1. Heightmap pass  – top-down, MAX-blended: tallest surface per world texel.
//                       Static geometry is cached; only moving entities are redrawn.
//  2. G-buffer pass   – every slice as an instanced quad, depth-tested (no CPU
//                       sorting), writing albedo+material and normal+height.
//  3. Lighting pass   – per low-res pixel: ambient + sun/moon (heightfield shadow)
//                       + point lights (heightfield shadows), emissives.
//  4. Blit            – nearest-neighbour upscale to the canvas.
//
// Geometry and lighting only meet through the G-buffer, so the geometry path
// (slice quads) can be swapped for meshes later without touching lighting.

import type { Atlas, ModelInfo } from '../assets/atlas';
import { DETAIL } from '../assets/voxel';
import type { DayLighting } from '../world/daycycle';
import type { Placed, PointLight } from '../world/village';
import { framebuffer, program, texture, type GL, type Program } from './gl';
import * as SH from './shaders';

const FLOATS_PER_INSTANCE = 8;
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
  }
  return o;
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

  private readonly lightPos = new Float32Array(SH.MAX_LIGHTS * 4);
  private readonly lightCol = new Float32Array(SH.MAX_LIGHTS * 4);

  constructor(
    private readonly gl: GL,
    private readonly atlas: Atlas,
    private readonly worldSize: number,
    ground: Uint8Array,
    placed: Placed[],
  ) {
    this.gbufProg = program(gl, SH.GBUFFER_VS, SH.GBUFFER_FS);
    this.heightProg = program(gl, SH.HEIGHT_VS, SH.HEIGHT_FS);
    this.lightProg = program(gl, SH.FULLSCREEN_VS, SH.LIGHTING_FS);
    this.blitProg = program(gl, SH.FULLSCREEN_VS, SH.BLIT_FS);

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
    for (const loc of [1, 2]) {
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribDivisor(loc, 1);
    }
    this.emptyVao = gl.createVertexArray()!;

    this.statics = new StaticBatch(gl, placed, worldSize);
    this.groundBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.groundBuf);
    const gt = worldSize * DETAIL; // ground texture size in texels
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([worldSize / 2, worldSize / 2, 0, 0, 0, 0, gt, gt]), gl.STATIC_DRAW);
    this.dynBuf = gl.createBuffer()!;

    const hm = () => {
      const tex = texture(gl, HM_SIZE, HM_SIZE, gl.R8, gl.RED, gl.UNSIGNED_BYTE);
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
  }

  /** Camera position snapped to whole pixels in screen space, to avoid shimmer while panning.
   *  Returns the snapped world position and that position in screen pixels. */
  snappedCamera(v: View): { x: number; y: number; px: number; py: number } {
    const c = Math.cos(v.angle), s = Math.sin(v.angle), k = Math.tan(v.pitch);
    const px = Math.round((c * v.x + s * v.y) * DETAIL), py = Math.round((-s * v.x + c * v.y) * k * DETAIL);
    const rx = px / DETAIL, ry = py / (k * DETAIL);
    return { x: c * rx - s * ry, y: s * rx + c * ry, px, py };
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
    stats.lights = this.packLights(f, vis, view);
    gl.uniform1i(lp.u.u_numLights, stats.lights);
    gl.uniform4fv(lp.u.u_lightPos, this.lightPos);
    gl.uniform4fv(lp.u.u_lightCol, this.lightCol);
    gl.drawArrays(gl.TRIANGLES, 0, 3);

    // ── 4. Upscale to canvas ────────────────────────────────────────────
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, this.canvasSize[0], this.canvasSize[1]);
    gl.useProgram(this.blitProg.prog);
    this.bindTex(0, this.litTex, this.blitProg.u.u_tex);
    gl.uniform1f(this.blitProg.u.u_scale, this.scale);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.bindVertexArray(null);
    return stats;
  }

  /** Pick the lights that can affect the screen (nearest first) and pack them into uniform arrays. */
  private packLights(f: FrameInput, vis: Bounds, view: View): number {
    const intensity = f.lighting.night;
    if (intensity < 0.01) return 0;
    const candidates = f.lights.filter(
      (l) => l.x + l.radius > vis.minX && l.x - l.radius < vis.maxX && l.y + l.radius > vis.minY && l.y - l.radius < vis.maxY,
    );
    candidates.sort((a, b) => Math.hypot(a.x - view.x, a.y - view.y) - Math.hypot(b.x - view.x, b.y - view.y));
    const n = Math.min(candidates.length, SH.MAX_LIGHTS);
    for (let i = 0; i < n; i++) {
      const l = candidates[i];
      const fl = l.flicker
        ? 1 + l.flicker * (0.6 * Math.sin(f.time * 11 + l.phase) + 0.4 * Math.sin(f.time * 23.7 + l.phase * 1.7))
        : 1;
      const k = intensity * fl * 1.35;
      this.lightPos.set([l.x, l.y, l.z, l.radius * (0.97 + 0.03 * fl)], i * 4);
      this.lightCol.set([l.color[0] * k, l.color[1] * k, l.color[2] * k, l.overhang ?? 0], i * 4);
    }
    return n;
  }
}
