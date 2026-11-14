export type GL = WebGL2RenderingContext;

function shader(gl: GL, type: number, src: string): WebGLShader {
  const s = gl.createShader(type)!;
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(s);
    const numbered = src.split('\n').map((l, i) => `${String(i + 1).padStart(3)}: ${l}`).join('\n');
    throw new Error(`shader compile failed:\n${log}\n${numbered}`);
  }
  return s;
}

export interface Program {
  prog: WebGLProgram;
  u: Record<string, WebGLUniformLocation>;
}

export function program(gl: GL, vs: string, fs: string): Program {
  const prog = gl.createProgram()!;
  gl.attachShader(prog, shader(gl, gl.VERTEX_SHADER, vs));
  gl.attachShader(prog, shader(gl, gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(`link failed: ${gl.getProgramInfoLog(prog)}`);
  const u: Record<string, WebGLUniformLocation> = {};
  const n = gl.getProgramParameter(prog, gl.ACTIVE_UNIFORMS) as number;
  for (let i = 0; i < n; i++) {
    const info = gl.getActiveUniform(prog, i)!;
    const name = info.name.replace(/\[0\]$/, '');
    u[name] = gl.getUniformLocation(prog, info.name)!;
  }
  return { prog, u };
}

export function texture(
  gl: GL,
  w: number,
  h: number,
  internalFormat: number,
  format: number,
  type: number,
  data: ArrayBufferView | null = null,
): WebGLTexture {
  const t = gl.createTexture()!;
  gl.bindTexture(gl.TEXTURE_2D, t);
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
  gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, w, h, 0, format, type, data);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  return t;
}

export function framebuffer(gl: GL, colors: WebGLTexture[], depth?: WebGLRenderbuffer): WebGLFramebuffer {
  const fb = gl.createFramebuffer()!;
  gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
  colors.forEach((t, i) => gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0 + i, gl.TEXTURE_2D, t, 0));
  if (depth) gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, depth);
  gl.drawBuffers(colors.map((_, i) => gl.COLOR_ATTACHMENT0 + i));
  const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
  if (status !== gl.FRAMEBUFFER_COMPLETE) throw new Error(`framebuffer incomplete: 0x${status.toString(16)}`);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  return fb;
}
