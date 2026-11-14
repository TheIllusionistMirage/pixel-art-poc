// Projection used throughout (orthographic, camera pitched `pitch` above the ground,
// normalised so every slice stays exactly 1 screen pixel above the one below it):
//   D      = DETAIL                                 // pixels (and voxels) per world unit
//   k      = tan(pitch)                             // 1 at 45°; < 1 = lower camera
//   r      = rotate(world.xy - cam, -camAngle)      // camera-relative, world units
//   screen = D * (r.x, r.y * k - z)                 // y down; ground squashed by k
//   depth  = r.y + z * k                            // larger = closer to viewer
// Slices sit 1/D world units apart, i.e. exactly one pixel. Heights in the G-buffer
// and heightmap are stored in voxels (world z * D) / 255.
// The lighting pass inverts this using the per-pixel height stored in the G-buffer.

/** Point lights: at most this many may reach one light-grid cell, and the index texture's row width. */
export const MAX_CELL_LIGHTS = 64;
export const LIGHT_INDEX_W = 1024;

/** Lamp flicker: two incommensurate sines (shared by the lighting and halo shaders). */
const FLICKER = /* glsl */ `
float lightFlicker(float amt, float ph, float t) {
  return 1.0 + amt * (0.6 * sin(t * 11.0 + ph) + 0.4 * sin(t * 23.7 + ph * 1.7));
}
`;

/** Per-instance attributes shared by every slice draw. */
const SLICE_ATTRIBS = /* glsl */ `
layout(location = 0) in vec2 a_corner;   // unit quad corner
layout(location = 1) in vec4 a_inst0;    // world x, world y, angle, slice z
layout(location = 2) in vec4 a_inst1;    // atlas u, v, w, h (texels)
layout(location = 3) in float a_sway;    // wind bend per voxel² of height (0 = rigid)
uniform vec2 u_atlasSize;
uniform float u_detail;  // texels per world unit

vec2 sliceWorldPos(out vec2 uv, out vec2 rot) {
  vec2 local = (a_corner - 0.5) * a_inst1.zw / u_detail;
  float ca = cos(a_inst0.z), sa = sin(a_inst0.z);
  rot = vec2(ca, sa);
  uv = (a_inst1.xy + a_corner * a_inst1.zw) / u_atlasSize;
  return a_inst0.xy + vec2(ca * local.x - sa * local.y, sa * local.x + ca * local.y);
}
`;

export const GBUFFER_VS = /* glsl */ `#version 300 es
precision highp float;
${SLICE_ATTRIBS}
uniform vec2 u_cam;
uniform vec2 u_camRot;   // cos, sin of camera angle
uniform vec2 u_res;      // internal resolution
uniform float u_tilt;    // tan(camera pitch)
uniform vec2 u_wind;     // wind vector (world), ~[0, 1.3] long
uniform float u_time;
out vec2 v_uv;
flat out vec2 v_rot;
flat out float v_z;

void main() {
  vec2 uv, rot;
  vec2 wp = sliceWorldPos(uv, rot);
  vec2 rel = wp - u_cam;
  vec2 r = vec2(u_camRot.x * rel.x + u_camRot.y * rel.y, -u_camRot.y * rel.x + u_camRot.x * rel.y);
  float z = a_inst0.w / u_detail;
  vec2 s = vec2(r.x, r.y * u_tilt - z) * u_detail;
  if (a_sway > 0.0) {
    // Bend with the wind: each slice shifts by a whole number of pixels (no
    // smearing), more the higher it is, wobbling with a per-plant phase. The
    // heightmap stays rigid; shadows don't need to sway.
    float ph = dot(a_inst0.xy, vec2(0.37, 0.23));
    float wob = 0.6 + 0.4 * sin(u_time * 2.1 + ph) + 0.15 * sin(u_time * 5.3 + ph * 2.7);
    vec2 off = u_wind * (a_sway * a_inst0.w * a_inst0.w * wob);
    vec2 ro = vec2(u_camRot.x * off.x + u_camRot.y * off.y, -u_camRot.y * off.x + u_camRot.x * off.y);
    s += floor(vec2(ro.x, ro.y * u_tilt) * u_detail + 0.5);
  }
  gl_Position = vec4(s.x * 2.0 / u_res.x, -s.y * 2.0 / u_res.y, -(r.y + z * u_tilt) / 8192.0, 1.0);
  v_uv = uv;
  v_rot = rot;
  v_z = a_inst0.w;
}
`;

export const GBUFFER_FS = /* glsl */ `#version 300 es
precision highp float;
uniform sampler2D u_albedo;
uniform sampler2D u_normal;
uniform int u_flatNormal;
in vec2 v_uv;
flat in vec2 v_rot;
flat in float v_z;
layout(location = 0) out vec4 o_albedo;   // rgb colour, a material
layout(location = 1) out vec4 o_normal;   // rgb world normal, a height/255

void main() {
  vec4 a = texture(u_albedo, v_uv);
  if (a.a < 0.01) discard;
  vec3 n = u_flatNormal == 1 ? vec3(0.0, 0.0, 1.0) : texture(u_normal, v_uv).xyz * 2.0 - 1.0;
  vec2 nr = vec2(v_rot.x * n.x - v_rot.y * n.y, v_rot.y * n.x + v_rot.x * n.y);
  o_albedo = a;
  o_normal = vec4(vec3(nr, n.z) * 0.5 + 0.5, v_z / 255.0);
}
`;

/**
 * Top-down pass, MAX blending, per world texel: r = top of the tallest slice,
 * g = 1 - bottom of the lowest one. Each column is then treated as solid from
 * its bottom to its top, so light can pass under eaves and canopies instead of
 * every overhang counting as a wall down to the ground.
 *
 * Things in the air (flying birds) go in a second interval, b / a, so a bird
 * passing over a roof doesn't fuse with it into one tall solid column. Only
 * shadows look at that interval; ambient occlusion and rain clipping use r.
 */
export const HEIGHT_VS = /* glsl */ `#version 300 es
precision highp float;
${SLICE_ATTRIBS}
uniform vec2 u_hmOrigin;
uniform float u_hmSize;   // world units covered
out vec2 v_uv;
flat out float v_z;

void main() {
  vec2 uv, rot;
  vec2 wp = sliceWorldPos(uv, rot);
  gl_Position = vec4((wp - u_hmOrigin) / u_hmSize * 2.0 - 1.0, 0.0, 1.0);
  v_uv = uv;
  v_z = a_inst0.w;
}
`;

export const HEIGHT_FS = /* glsl */ `#version 300 es
precision highp float;
uniform sampler2D u_albedo;
uniform int u_layer;   // 0 = ground-based (rg), 1 = airborne (ba)
in vec2 v_uv;
flat in float v_z;
out vec4 o;
void main() {
  if (texture(u_albedo, v_uv).a < 0.01) discard;
  vec2 span = vec2(v_z / 255.0, (256.0 - v_z) / 255.0);  // slice v_z spans voxels [v_z - 1, v_z]
  o = u_layer == 0 ? vec4(span, 0.0, 0.0) : vec4(0.0, 0.0, span);
}
`;

export const FULLSCREEN_VS = /* glsl */ `#version 300 es
void main() {
  vec2 p = vec2(gl_VertexID == 1 ? 3.0 : -1.0, gl_VertexID == 2 ? 3.0 : -1.0);
  gl_Position = vec4(p, 0.0, 1.0);
}
`;

export const LIGHTING_FS = /* glsl */ `#version 300 es
precision highp float;
precision highp int;
#define MAX_CELL_LIGHTS ${MAX_CELL_LIGHTS}
#define LIGHT_INDEX_W ${LIGHT_INDEX_W}
uniform sampler2D u_gAlbedo;
uniform sampler2D u_gNormal;
uniform sampler2D u_height;
uniform vec2 u_res;
uniform vec2 u_cam;
uniform vec2 u_camRot;
uniform ivec2 u_camPix;     // snapped camera position in screen pixels
uniform float u_tilt;       // tan(camera pitch)
uniform float u_detail;     // pixels per world unit
uniform vec2 u_hmOrigin;
uniform float u_hmSize;
uniform vec3 u_lightDir;
uniform vec3 u_lightColor;
uniform vec3 u_ambient;
uniform vec3 u_sky;
uniform float u_night;
uniform float u_time;
// Point lights live in a world-space grid; a pixel only visits its own cell's list.
uniform highp sampler2D u_lightData;    // 3 texels per light: (x, y, z, radius) (r, g, b, overhang) (flicker, phase, -, -)
uniform highp isampler2D u_lightCells;  // per cell: (first entry in the index, count)
uniform highp isampler2D u_lightIndex;  // light ids, LIGHT_INDEX_W per row
uniform vec3 u_lightGrid;               // grid origin x, y, cell size (world units)
uniform ivec2 u_lightGridSize;
uniform float u_lightScale;             // night × gain; 0 = lamps off
uniform int u_debug;
uniform int u_shadows;
// Weather
uniform vec2 u_viewCS;      // cos, sin of camera pitch
uniform float u_cloud;
uniform vec2 u_cloudOffset;
uniform float u_wet;
uniform float u_rain;
uniform float u_snowCover;
uniform float u_fog;
uniform float u_mist;
uniform vec4 u_mistA;        // rain-sheet layer: dir.xy, offset.xy
uniform vec4 u_mistB;
uniform vec2 u_mistW;        // layer weights (sum to 1)
uniform vec3 u_fogColor;
out vec4 o;

const float BAYER[16] = float[16](0., 8., 2., 10., 12., 4., 14., 6., 3., 11., 1., 9., 15., 7., 13., 5.);

float heightAt(vec2 w) {
  vec2 uv = (w - u_hmOrigin) / u_hmSize;
  if (uv.x < 0.0 || uv.y < 0.0 || uv.x > 1.0 || uv.y > 1.0) return 0.0;
  return texture(u_height, uv).r * 255.0 / u_detail;
}
/** Does the heightfield column at q.xy block the point q? Columns are solid from
 *  their lowest to their highest voxel, so overhangs (eaves) don't shade what's under them. */
bool blocked(vec3 q) {
  vec2 uv = (q.xy - u_hmOrigin) / u_hmSize;
  if (uv.x < 0.0 || uv.y < 0.0 || uv.x > 1.0 || uv.y > 1.0) return false;
  vec4 c = texture(u_height, uv) * 255.0;
  float z = q.z + 0.5;
  vec2 top = c.xz / u_detail, bottom = (255.0 - c.yw) / u_detail;
  return (top.x > z && bottom.x < z) || (top.y > z && bottom.y < z);
}
${FLICKER}
float hash12(vec2 p) {
  return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
}

uint pcg(uint v) {
  uint s = v * 747796405u + 2891336453u;
  uint w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}
/** Stable integer hash in [0, 1] (no sin(): identical on every GPU, at any coordinate). */
float hash2i(ivec2 c) {
  return float(pcg(uint(c.x) * 1597334677u ^ pcg(uint(c.y)))) / 4294967295.0;
}
float vnoise(vec2 x) {
  vec2 i = floor(x), f = x - i;
  f = f * f * (3.0 - 2.0 * f);
  ivec2 c = ivec2(i);
  return mix(mix(hash2i(c), hash2i(c + ivec2(1, 0)), f.x), mix(hash2i(c + ivec2(0, 1)), hash2i(c + ivec2(1, 1)), f.x), f.y);
}

/** Rain rings on water and puddles: a world-aligned grid of cells, a few
 *  of which (more in heavier rain) hold an expanding one-pixel ring. */
// One layer of rain sheets: bands across its (frozen) direction, carried by its offset.
float rainSheets(vec2 p, vec4 L, float salt) {
  vec2 q = p - L.zw;
  float along = dot(q, L.xy), across = dot(q, vec2(-L.y, L.x));
  return vnoise(vec2(along / 26.0, across / 80.0) + salt) * 0.7 + vnoise(vec2(along / 11.0, across / 30.0) + salt + 4.2) * 0.3;
}

float rainRipple(vec3 p) {
  vec2 q = p.xy * u_detail;              // ground texels
  ivec2 cell = ivec2(floor(q / 7.0));
  float h = hash2i(cell + ivec2(31, 17));
  if (h > u_rain * 0.9) return 0.0;
  float h2 = hash2i(cell + ivec2(5, 91)), h3 = hash2i(cell + ivec2(77, 3));
  vec2 c = (vec2(cell) + 0.25 + 0.5 * vec2(h2, h3)) * 7.0;
  float t = fract(u_time * 1.3 + h * 17.0);
  float d = abs(length(q - c) - t * 3.4);
  return d < 0.6 ? 1.0 - t : 0.0;
}

// March towards the light through the world heightfield, one texel per step.
float dirShadow(vec3 p) {
  vec2 d = u_lightDir.xy;
  float l = length(d);
  if (l < 1e-4) return 1.0;
  vec2 stepXY = d / l;
  float stepZ = u_lightDir.z / l;
  vec3 q = p;
  for (int i = 0; i < 200; i++) {
    q.xy += stepXY;
    q.z += stepZ;
    if (q.z > 90.0) return 1.0;
    if (blocked(q)) return 0.0;
  }
  return 1.0;
}

/** overhang > 0: heights above it are canopies the light shines under, not solid columns. */
float pointShadow(vec3 p, vec3 lp, float overhang) {
  vec3 d = lp - p;
  float n = min(ceil(length(d.xy)), 72.0);
  for (float k = 1.0; k < 72.0; k += 1.0) {
    if (k >= n) break;
    vec3 q = p + d * (k / n);
    if (blocked(q) && (overhang <= 0.0 || heightAt(q.xy) < overhang)) return 0.0;
  }
  return 1.0;
}

/** Ground contact occlusion: how much tall stuff surrounds this ground point. */
float groundOcclusion(vec3 p) {
  float occ = 0.0;
  for (int i = 0; i < 8; i++) {
    float a = float(i) * 0.785398;
    vec2 dir = vec2(cos(a), sin(a));
    occ += clamp((heightAt(p.xy + dir * 3.0) - p.z - 1.0) / 6.0, 0.0, 1.0);
    occ += clamp((heightAt(p.xy + dir * 7.0) - p.z - 2.0) / 10.0, 0.0, 1.0) * 0.6;
  }
  return occ / 12.8;
}

float depthAt(ivec2 c) {
  vec4 a = texelFetch(u_gAlbedo, c, 0);
  float z = a.a < 0.01 ? 0.0 : texelFetch(u_gNormal, c, 0).a * 255.0 / u_detail;
  // View depth in world units, scaled so the outline threshold is pitch-independent.
  return (u_res.y * 0.5 - (float(c.y) + 0.5)) / u_detail + z + z * u_tilt * u_tilt;
}

void main() {
  ivec2 pc = ivec2(gl_FragCoord.xy);
  vec4 alb = texelFetch(u_gAlbedo, pc, 0);
  vec4 nh = texelFetch(u_gNormal, pc, 0);
  if (alb.a < 0.01) { o = vec4(u_sky, 1.0); return; }

  // Ordered dither anchored to the world (stable while panning).
  ivec2 sp = ivec2(pc.x - int(u_res.x * 0.5), int(u_res.y * 0.5) - pc.y) + u_camPix + ivec2(65536);
  float dth = (BAYER[(sp.y & 3) * 4 + (sp.x & 3)] + 0.5) / 16.0;

  vec3 n = normalize(nh.xyz * 2.0 - 1.0);
  float z = nh.a * 255.0 / u_detail;
  // Invert the projection to recover the world position of this pixel.
  vec2 r = vec2((gl_FragCoord.x - u_res.x * 0.5) / u_detail, ((u_res.y * 0.5 - gl_FragCoord.y) / u_detail + z) / u_tilt);
  vec2 rel = vec2(u_camRot.x * r.x - u_camRot.y * r.y, u_camRot.y * r.x + u_camRot.x * r.y);
  vec3 p = vec3(u_cam + rel, z);
  vec3 ps = p + vec3(n.xy * 0.75, 0.0);
  bool ground = z < 0.25;

  float mat = alb.a * 255.0;
  bool isFire = abs(mat - 150.0) < 2.0;
  bool isWindow = abs(mat - 200.0) < 2.0;
  bool isFoliage = abs(mat - 100.0) < 2.0;
  bool isWater = abs(mat - 50.0) < 2.0;
  bool isActor = abs(mat - 254.0) < 0.5;
  bool weathered = !isWater && !isFire && !isWindow && !isActor;
  // Toward the viewer, world space.
  vec3 V = vec3(-u_camRot.y * u_viewCS.x, u_camRot.x * u_viewCS.x, u_viewCS.y);

  // ── Weather on surfaces: snow settles on what faces up, rain darkens
  // everything and pools on bare ground.
  vec3 base = alb.rgb;
  bool grassy = alb.g > alb.r * 1.08;
  bool snowy = false;
  bool puddle = false;
  if (weathered && u_snowCover > 0.01) {
    float amt = u_snowCover * smoothstep(0.35, 0.75, n.z);
    if (ground && !grassy) amt *= 0.55;  // trodden roads stay slushy
    float sn = vnoise(p.xy / 6.0) * 0.55 + vnoise(p.xy / 2.2 + 5.7) * 0.3 + hash12(floor(p.xy * u_detail)) * 0.08 + dth * 0.07;
    if (sn < amt * 1.12 - 0.04) {
      snowy = true;
      base = vec3(0.86, 0.89, 0.94);
      isFoliage = false;
    }
  }
  float wet = weathered && !snowy ? u_wet : 0.0;
  if (wet > 0.01) {
    base *= 1.0 - wet * (isFoliage ? 0.14 : 0.28);
    if (ground && !grassy) {
      float pn = vnoise(p.xy / 9.0) * 0.7 + vnoise(p.xy / 3.5 + 3.1) * 0.3;
      puddle = pn + (dth - 0.5) * 0.05 < wet * 0.42;
    }
  }

  // ── Cloud shadows: drifting noise with soft edges. Full overcast covers
  // everything; the lost direct light has already gone into the ambient.
  float cloudMul = 1.0;
  if (u_cloud > 0.01) {
    vec2 cq = p.xy + u_cloudOffset;
    float cn = vnoise(cq / 110.0) * 0.65 + vnoise(cq / 45.0 + 7.3) * 0.35;
    float thr = mix(1.1, -0.1, u_cloud);
    float cov = smoothstep(thr - 0.07, thr + 0.07, cn);
    cloudMul = 1.0 - 0.88 * cov;
  }

  // ── Sun / moon: cel bands. Flat ground keeps its smooth value (it's uniform
  // anyway), everything else snaps to 4 tones with a dithered seam.
  float ndl = max(dot(n, u_lightDir), 0.0);
  if (isFoliage) ndl = ndl * 0.8 + 0.2 * max(dot(-n, u_lightDir) * 0.5 + 0.5, 0.0) * 0.5;
  if (!ground) {
    float v = ndl + (dth - 0.5) * (isFoliage ? 0.06 : 0.12);
    ndl = v < 0.12 ? 0.0 : v < 0.45 ? 0.35 : v < 0.78 ? 0.7 : 1.0;
  }
  float sh = (u_shadows == 1 && ndl > 0.0 && cloudMul > 0.2 && dot(u_lightColor, vec3(1.0)) > 0.001) ? dirShadow(ps) : 1.0;

  // Soft hemispheric ambient, plus contact occlusion on the ground.
  vec3 hemi = u_ambient * (0.55 + 0.45 * clamp(0.5 + n.z, 0.0, 1.0));
  float ao = 1.0;
  if (ground) {
    float occ = groundOcclusion(p);
    ao = 1.0 - 0.24 * floor(clamp(occ * 2.6, 0.0, 1.0) * 3.0 + dth) / 3.0;
  }

  // ── Point lights: stepped, dithered falloff rings, heightfield shadows.
  vec3 pl = vec3(0.0);
  vec3 plSpec = vec3(0.0);
  ivec2 lcell = ivec2(floor((p.xy - u_lightGrid.xy) / u_lightGrid.z));
  ivec2 cellList = u_lightScale > 0.0 && all(greaterThanEqual(lcell, ivec2(0))) && all(lessThan(lcell, u_lightGridSize))
    ? texelFetch(u_lightCells, lcell, 0).xy : ivec2(0);
  for (int j = 0; j < MAX_CELL_LIGHTS; j++) {
    if (j >= cellList.y) break;
    int e = cellList.x + j;
    int id = texelFetch(u_lightIndex, ivec2(e % LIGHT_INDEX_W, e / LIGHT_INDEX_W), 0).x;
    vec4 l0 = texelFetch(u_lightData, ivec2(0, id), 0);
    vec4 l1 = texelFetch(u_lightData, ivec2(1, id), 0);
    vec4 l2 = texelFetch(u_lightData, ivec2(2, id), 0);
    float fl = lightFlicker(l2.x, l2.y, u_time);
    vec3 lp = l0.xyz;
    float rad = l0.w * (0.97 + 0.03 * fl);
    vec3 lcol = l1.rgb * (u_lightScale * fl);
    vec3 d = lp - p;
    float dist = length(d);
    if (dist > rad) continue;
    vec3 L = d / max(dist, 1e-3);
    float att = 1.0 - dist / rad;
    att = floor((att * att) * 5.0 + dth) / 5.0;
    if (att <= 0.0) continue;
    float lambert = max(dot(n, L), 0.0) * 0.6 + 0.4;
    float s = u_shadows == 1 ? pointShadow(ps, lp, l1.w) : 1.0;
    pl += lcol * att * lambert * s;
    if (wet > 0.01) plSpec += lcol * att * s * pow(max(dot(n, normalize(L + V)), 0.0), puddle ? 40.0 : 16.0);
  }

  // Shadows only take away part of the direct light: soft, readable tone shifts.
  vec3 light = (hemi + u_lightColor * ndl * mix(0.3, 1.0, sh) * cloudMul) * ao + pl;
  vec3 col = base * light;

  // ── Wet: puddles mirror the sky, and lamps and the sun leave stepped highlights.
  if (wet > 0.01) {
    float gloss = puddle ? 1.0 : isFoliage ? 0.2 : ground ? 0.35 : 0.6;
    if (puddle) col = mix(col, u_sky * 0.8 + pl * 0.25, 0.5 * wet);
    vec3 ns = puddle ? normalize(vec3(vnoise(p.xy * 1.7) - 0.5, vnoise(p.xy * 1.7 + 9.1) - 0.5, 3.0)) : n;
    if (!ground || puddle) {
      float sp = pow(max(dot(ns, normalize(u_lightDir + V)), 0.0), puddle ? 50.0 : 20.0) * sh * cloudMul;
      col += u_lightColor * floor(sp * 3.0) / 3.0 * 0.45 * gloss * wet;
    }
    // Hard steps, no dither: faint highlights vanish rather than sparkle as noise.
    col += floor(plSpec * 3.0) / 3.0 * 0.7 * gloss * wet;
  }
  if ((isWater || puddle) && u_rain > 0.01) {
    float rip = rainRipple(p);
    if (rip > 0.0) col = mix(col, u_sky * 0.55 + light * 0.45, 0.6 * step(dth, rip + 0.2));
  }

  if (isWater) {
    // Drifting ripple glints, laid out in screen-aligned rows but anchored to the world.
    vec2 wr = vec2(u_camRot.x * p.x + u_camRot.y * p.y, (-u_camRot.y * p.x + u_camRot.x * p.y) * u_tilt) * u_detail;
    wr.x += u_time * 3.0;
    vec2 size = vec2(26.0, 9.0);
    vec2 cell = floor(wr / size);
    vec2 f = floor(wr - cell * size);
    float h1 = hash12(cell), h2 = hash12(cell + 17.31);
    float ph = fract(u_time * 0.18 + h1 * 3.7);
    float x0 = floor(h1 * 14.0), len = 3.0 + floor(h2 * 7.0);
    bool row1 = f.y == 2.0 && f.x >= x0 && f.x < x0 + len;
    bool row2 = f.y == 3.0 && f.x >= x0 + 1.0 && f.x < x0 + len - 1.0 + step(0.5, h2) * 2.0;
    if (h2 > 0.25 && ph < 0.6 && (row1 || row2)) col = mix(col, vec3(0.78, 0.86, 0.86) * (light * 0.85 + 0.1), 0.5);
    col += u_lightColor * pow(max(dot(n, u_lightDir), 0.0), 8.0) * 0.04;
  }
  if (isWindow) col += vec3(1.0, 0.7, 0.36) * u_night * 0.95 * (0.7 + alb.rgb);
  if (isFire) col = alb.rgb * (1.05 + 0.1 * sin(u_time * 13.0 + p.x));

  // ── Selective outline: darken silhouette pixels that sit in front of something far behind.
  float dSelf = depthAt(pc);
  float gap = 0.0;
  gap = max(gap, dSelf - depthAt(pc + ivec2(1, 0)));
  gap = max(gap, dSelf - depthAt(pc - ivec2(1, 0)));
  gap = max(gap, dSelf - depthAt(pc + ivec2(0, 1)));
  gap = max(gap, dSelf - depthAt(pc - ivec2(0, 1)));
  if (!ground && !isFire && gap > 7.0) col *= vec3(0.72, 0.77, 0.78);

  // ── Fog and rain mist: anchored to the world, drifting, thinning with height.
  // Nothing here may depend on screen position: the view is orthographic, so a
  // "further away" gradient is really a screen gradient and slides with every
  // pan. Kept smooth: dithering a low-contrast veil only turns it into
  // screen-wide noise.
  if (u_mist > 0.005) {
    // Spray hugging the ground where the drops hit, plus broad sheets of rain
    // that sweep through downwind. Two layers crossfade so the sheets can follow
    // a turning wind; the blend is renormalised to keep its contrast steady.
    float s0 = rainSheets(p.xy, u_mistA, 0.0), s1 = rainSheets(p.xy, u_mistB, 13.7);
    float sheets = clamp((u_mistW.x * s0 + u_mistW.y * s1 - 0.5) / length(u_mistW) + 0.5, 0.0, 1.0);
    float spray = exp(-z / 6.0);
    float ma = u_mist * 0.65 * (0.5 + 0.5 * spray) * (0.45 + 1.1 * sheets);
    col = mix(col, u_fogColor, min(ma, 0.75));
  }
  float veil = u_fog + 0.6 * u_mist;
  if (veil > 0.01) {
    // Broad banks (a couple of houses across) with smaller wisps through them,
    // both carried by the wind. Averages the same density as before.
    vec2 fq = p.xy + u_cloudOffset * 0.6;
    float banks = vnoise(fq / 110.0 + 2.3) * 0.65 + vnoise(fq / 34.0) * 0.35;
    float fa = u_fog * (0.35 + 0.6 * banks) * mix(1.0, exp(-z / 22.0), 0.5);
    fa = min(fa, 0.92);
    col = mix(col, u_fogColor, fa);
    // Lamp haloes are drawn in the effects pass (HALO_VS), one sprite per light.
  }

  // ── Grade: muted, slightly lifted, cool shadows (the painterly look).
  float lum = dot(col, vec3(0.299, 0.587, 0.114));
  col = mix(vec3(lum), col, 0.8);
  col = col * 0.9 + vec3(0.03, 0.044, 0.042) * (1.0 - 0.6 * u_night);

  if (u_debug == 1) col = alb.rgb;
  else if (u_debug == 2) col = n * 0.5 + 0.5;
  else if (u_debug == 3) col = vec3(z / 70.0);
  else if (u_debug == 4) col = vec3(sh * ndl);
  else if (u_debug == 5) col = light * 0.6;

  o = vec4(col, 1.0);
}
`;

/** Shared by the weather effect shaders: world (camera-relative, rotated) → clip,
 *  snapped to the pixel grid, with the same depth as the G-buffer. */
const FX_PROJECT = /* glsl */ `
uniform vec2 u_res;
uniform float u_detail;
uniform float u_tilt;
uniform vec2 u_camRot;
uniform vec2 u_cam;
vec2 toCamera(vec2 w) {
  vec2 rel = w - u_cam;
  return vec2(u_camRot.x * rel.x + u_camRot.y * rel.y, -u_camRot.y * rel.x + u_camRot.x * rel.y);
}
vec2 toWorld(vec2 r) {
  return u_cam + vec2(u_camRot.x * r.x - u_camRot.y * r.y, u_camRot.y * r.x + u_camRot.x * r.y);
}
/** odd = 1: centre on a pixel centre (lines, odd-sized points); 0: on a pixel corner. */
vec4 project(vec2 r, float z, float odd) {
  vec2 s = vec2(r.x, r.y * u_tilt - z) * u_detail + u_res * 0.5;   // pixels from the top-left
  s = floor(s) + 0.5 * odd - u_res * 0.5;
  return vec4(s.x * 2.0 / u_res.x, -s.y * 2.0 / u_res.y, -(r.y + z * u_tilt) / 8192.0, 1.0);
}
uint pcg(uint v) {
  uint s = v * 747796405u + 2891336453u;
  uint w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}
float rnd(inout uint s) { s = pcg(s); return float(s) / 4294967295.0; }
`;

/**
 * Lamp haloes in fog and rain mist: one additive Gaussian sprite per light
 * (instanced, light data fetched by gl_InstanceID), centred on the lamp's
 * projected position and 3σ wide. Not depth-tested: it's glow in the air.
 */
export const HALO_VS = /* glsl */ `#version 300 es
precision highp float;
precision highp int;
${FX_PROJECT}
${FLICKER}
uniform highp sampler2D u_lightData;
uniform float u_time;
uniform float u_amount;      // halo strength (night × veil × gain)
out vec2 v_off;
out float v_sig;
out vec3 v_col;

void main() {
  vec4 l0 = texelFetch(u_lightData, ivec2(0, gl_InstanceID), 0);
  vec4 l1 = texelFetch(u_lightData, ivec2(1, gl_InstanceID), 0);
  vec4 l2 = texelFetch(u_lightData, ivec2(2, gl_InstanceID), 0);
  float fl = lightFlicker(l2.x, l2.y, u_time);
  vec2 r = toCamera(l0.xy);
  vec2 c = vec2(r.x, r.y * u_tilt - l0.z) * u_detail;       // pixels from the centre, y down
  float sig = 4.0 + l0.w * (0.97 + 0.03 * fl) * 0.22;
  vec2 corner = vec2(float(gl_VertexID & 1), float(gl_VertexID >> 1)) * 2.0 - 1.0;
  v_off = corner * sig * 3.0;
  vec2 s = c + v_off;
  gl_Position = vec4(s.x * 2.0 / u_res.x, -s.y * 2.0 / u_res.y, 0.0, 1.0);
  v_sig = sig;
  v_col = l1.rgb * fl * u_amount;
}
`;

export const HALO_FS = /* glsl */ `#version 300 es
precision highp float;
in vec2 v_off;
in float v_sig;
in vec3 v_col;
out vec4 o;
void main() {
  o = vec4(v_col * exp(-dot(v_off, v_off) / (2.0 * v_sig * v_sig)), 0.0);
}
`;

/**
 * Rain (LINES) and snow (POINTS), no vertex buffers: each instance is one
 * drop, derived from gl_InstanceID. Drops live at fixed world positions inside
 * a box around the camera (wrapping at its off-screen edges), fall through it,
 * and stop at whatever the heightmap says is below them; rain leaves a brief
 * splash there.
 */
export const PRECIP_VS = /* glsl */ `#version 300 es
precision highp float;
precision highp int;
${FX_PROJECT}
uniform int u_mode;           // 0 rain, 1 snow
uniform float u_time;
uniform vec2 u_wind;          // camera-rotated
uniform vec2 u_drift;         // wind integrated over time (camera-rotated)
uniform float u_carry;        // sideways speed per unit of wind
uniform vec2 u_camR;          // camera position, camera-rotated
uniform vec4 u_box;           // x lo, y lo, x size, y size (camera-rotated world units)
uniform float u_top;          // fall height
uniform sampler2D u_height;
uniform vec2 u_hmOrigin;
uniform float u_hmSize;
out float v_a;

float heightAt(vec2 w) {
  vec2 uv = (w - u_hmOrigin) / u_hmSize;
  if (uv.x < 0.0 || uv.y < 0.0 || uv.x > 1.0 || uv.y > 1.0) return 0.0;
  return texture(u_height, uv).r * 255.0 / u_detail;
}

void main() {
  uint seed = uint(gl_InstanceID) * 3u + uint(u_mode) * 7919u;
  float ph = rnd(seed), hv = rnd(seed);
  bool snow = u_mode == 1;
  float speed = snow ? 12.0 + 6.0 * hv : 150.0 + 40.0 * hv;
  float period = u_top / speed;
  float cyc = u_time / period + ph;
  float age = fract(cyc) * period;                    // seconds since leaving the top
  float z = u_top - age * speed;
  // A fresh spot for every fall, so calm air doesn't drop flakes down the same column.
  uint s2 = seed ^ (uint(int(floor(cyc))) * 2654435761u);
  float hx = rnd(s2), hy = rnd(s2);
  // Sideways: the wind integrated over time (shared by all), so every particle
  // moves with the current wind and a gust never shifts the field as a whole.
  vec2 drift = u_drift;
  if (snow) drift += vec2(sin(u_time * 1.7 + ph * 40.0), cos(u_time * 1.3 + hv * 40.0)) * 0.9;
  vec2 r = mod(vec2(hx, hy) * 8192.0 + drift - u_camR - u_box.xy, u_box.zw) + u_box.xy;
  float ground = heightAt(toWorld(r));
  gl_PointSize = hv > 0.7 ? 2.0 : 1.0;
  v_a = 1.0;
  if (z < ground) {
    float since = (ground - z) / speed;
    if (snow || since > 0.09) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }  // clipped
    // Splash: a tiny horizontal tick on the surface.
    v_a = 0.85;
    gl_Position = project(r, ground + 0.4, 1.0) + vec4((gl_VertexID == 0 ? -1.5 : 1.5) * 2.0 / u_res.x, 0.0, 0.0, 0.0);
    return;
  }
  if (snow) { gl_Position = project(r, z, gl_PointSize == 1.0 ? 1.0 : 0.0); return; }
  // Rain streak: head, and a tail back along the velocity.
  v_a = 0.32 + 0.3 * hv;
  if (gl_VertexID == 1) {
    float tl = 0.022;
    r -= u_wind * u_carry * tl;
    z += speed * tl;
  }
  gl_Position = project(r, z, 1.0);
}
`;

export const PRECIP_FS = /* glsl */ `#version 300 es
precision highp float;
uniform vec4 u_color;
in float v_a;
out vec4 o;
void main() { o = vec4(u_color.rgb, u_color.a * v_a); }
`;

/**
 * Chimney smoke: a stream of K puffs per chimney (attribute divisor K), each
 * rising, growing and drifting downwind over its life. Drawn as dithered discs.
 */
export const SMOKE_VS = /* glsl */ `#version 300 es
precision highp float;
precision highp int;
${FX_PROJECT}
layout(location = 0) in vec4 a_chimney;   // x, y, z, seed
uniform int u_K;
uniform float u_life;
uniform float u_time;
uniform vec2 u_wind;      // world
uniform float u_amount;
out float v_a;
out float v_r;

void main() {
  int k = gl_InstanceID % u_K;
  uint seed = uint(gl_InstanceID) * 13u + 5u;
  float j1 = rnd(seed), j2 = rnd(seed);
  float a = fract(u_time / u_life + float(k) / float(u_K) + a_chimney.w);
  float t = a * u_life;
  float wl = min(length(u_wind), 1.2);
  vec3 p = a_chimney.xyz;
  // Strong wind flattens the plume and tears it apart sooner.
  p.z += 4.5 * (1.0 - 0.7 * min(wl, 1.0)) * t + 1.0;
  p.xy += u_wind * 10.0 * t + vec2(sin(t * 0.9 + j1 * 6.3), cos(t * 0.7 + j2 * 6.3)) * (0.4 + 0.4 * t);
  float size = floor(mix(2.0, 8.0 + 2.0 * j1, sqrt(a)) * (1.0 + 0.25 * wl));
  gl_PointSize = size;
  v_r = size;
  v_a = u_amount * smoothstep(0.0, 0.06, a) * (1.0 - a * a) * (0.85 + 0.25 * j2) * 1.35;
  v_a *= 1.0 - 0.85 * min(wl, 1.0) * smoothstep(0.1, 0.55, a);   // torn apart sooner in strong wind
  gl_Position = project(toCamera(p.xy), p.z, mod(size, 2.0));
}
`;

export const SMOKE_FS = /* glsl */ `#version 300 es
precision highp float;
precision highp int;
uniform vec3 u_color;
uniform vec2 u_res;
uniform ivec2 u_camPix;
in float v_a;
in float v_r;
out vec4 o;
const float BAYER[16] = float[16](0., 8., 2., 10., 12., 4., 14., 6., 3., 11., 1., 9., 15., 7., 13., 5.);
void main() {
  vec2 q = gl_PointCoord * 2.0 - 1.0;
  float d = dot(q, q);
  if (d > 1.0) discard;
  ivec2 pc = ivec2(gl_FragCoord.xy);
  ivec2 sp = ivec2(pc.x - int(u_res.x * 0.5), int(u_res.y * 0.5) - pc.y) + u_camPix + ivec2(65536);
  float dth = (BAYER[(sp.y & 3) * 4 + (sp.x & 3)] + 0.5) / 16.0;
  if (v_a * (1.0 - d * d) < dth) discard;
  // Lit from above: a darker underside.
  o = vec4(u_color * (0.82 + 0.18 * -q.y + 0.1 * (1.0 - d)), 1.0);
}
`;

/** Lightning bolt: world-space line list. */
export const BOLT_VS = /* glsl */ `#version 300 es
precision highp float;
precision highp int;
${FX_PROJECT}
layout(location = 0) in vec3 a_pos;
uniform vec2 u_offset;   // pixels
void main() {
  gl_Position = project(toCamera(a_pos.xy), a_pos.z, 1.0) + vec4(u_offset * 2.0 / u_res, 0.0, 0.0);
}
`;

export const BOLT_FS = /* glsl */ `#version 300 es
precision highp float;
uniform vec4 u_color;
out vec4 o;
void main() { o = u_color; }
`;

/** Nearest-neighbour upscale of the low-res frame to the canvas. */
export const BLIT_FS = /* glsl */ `#version 300 es
precision highp float;
uniform sampler2D u_tex;
uniform float u_scale;
out vec4 o;
void main() {
  o = texelFetch(u_tex, ivec2(floor(gl_FragCoord.xy / u_scale)), 0);
}
`;
