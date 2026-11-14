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

export const MAX_LIGHTS = 48;

/** Per-instance attributes shared by every slice draw. */
const SLICE_ATTRIBS = /* glsl */ `
layout(location = 0) in vec2 a_corner;   // unit quad corner
layout(location = 1) in vec4 a_inst0;    // world x, world y, angle, slice z
layout(location = 2) in vec4 a_inst1;    // atlas u, v, w, h (texels)
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

/** Top-down pass: writes the tallest surface per world texel (MAX blending). */
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
in vec2 v_uv;
flat in float v_z;
out vec4 o;
void main() {
  if (texture(u_albedo, v_uv).a < 0.01) discard;
  o = vec4(v_z / 255.0, 0.0, 0.0, 1.0);
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
#define MAX_LIGHTS ${MAX_LIGHTS}
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
uniform int u_numLights;
uniform vec4 u_lightPos[MAX_LIGHTS];   // xyz, radius
uniform vec4 u_lightCol[MAX_LIGHTS];   // rgb premultiplied by intensity, w = overhang height (0 = none)
uniform int u_debug;
uniform int u_shadows;
out vec4 o;

const float BAYER[16] = float[16](0., 8., 2., 10., 12., 4., 14., 6., 3., 11., 1., 9., 15., 7., 13., 5.);

float heightAt(vec2 w) {
  vec2 uv = (w - u_hmOrigin) / u_hmSize;
  if (uv.x < 0.0 || uv.y < 0.0 || uv.x > 1.0 || uv.y > 1.0) return 0.0;
  return texture(u_height, uv).r * 255.0 / u_detail;
}

float hash12(vec2 p) {
  return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
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
    if (heightAt(q.xy) > q.z + 0.5) return 0.0;
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
    float h = heightAt(q.xy);
    if (h > q.z + 0.5 && (overhang <= 0.0 || h < overhang)) return 0.0;
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

  // ── Sun / moon: cel bands. Flat ground keeps its smooth value (it's uniform
  // anyway), everything else snaps to 4 tones with a dithered seam.
  float ndl = max(dot(n, u_lightDir), 0.0);
  if (isFoliage) ndl = ndl * 0.8 + 0.2 * max(dot(-n, u_lightDir) * 0.5 + 0.5, 0.0) * 0.5;
  if (!ground) {
    float v = ndl + (dth - 0.5) * (isFoliage ? 0.06 : 0.12);
    ndl = v < 0.12 ? 0.0 : v < 0.45 ? 0.35 : v < 0.78 ? 0.7 : 1.0;
  }
  float sh = (u_shadows == 1 && ndl > 0.0 && dot(u_lightColor, vec3(1.0)) > 0.001) ? dirShadow(ps) : 1.0;

  // Soft hemispheric ambient, plus contact occlusion on the ground.
  vec3 hemi = u_ambient * (0.55 + 0.45 * clamp(0.5 + n.z, 0.0, 1.0));
  float ao = 1.0;
  if (ground) {
    float occ = groundOcclusion(p);
    ao = 1.0 - 0.24 * floor(clamp(occ * 2.6, 0.0, 1.0) * 3.0 + dth) / 3.0;
  }

  // ── Point lights: stepped, dithered falloff rings, heightfield shadows.
  vec3 pl = vec3(0.0);
  for (int i = 0; i < MAX_LIGHTS; i++) {
    if (i >= u_numLights) break;
    vec3 lp = u_lightPos[i].xyz;
    float rad = u_lightPos[i].w;
    vec3 d = lp - p;
    float dist = length(d);
    if (dist > rad) continue;
    vec3 L = d / max(dist, 1e-3);
    float att = 1.0 - dist / rad;
    att = floor((att * att) * 5.0 + dth) / 5.0;
    if (att <= 0.0) continue;
    float lambert = max(dot(n, L), 0.0) * 0.6 + 0.4;
    float s = u_shadows == 1 ? pointShadow(ps, lp, u_lightCol[i].w) : 1.0;
    pl += u_lightCol[i].rgb * att * lambert * s;
  }

  // Shadows only take away part of the direct light: soft, readable tone shifts.
  vec3 light = (hemi + u_lightColor * ndl * mix(0.3, 1.0, sh)) * ao + pl;
  vec3 col = alb.rgb * light;

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
