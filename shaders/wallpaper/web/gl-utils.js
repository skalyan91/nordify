// Small WebGL2 helper layer -- the JS analogue of moderngl's convenience
// API (see shaders/wallpaper/pipeline.py), just enough to drive the
// wallpaper/night/mixing pipelines' pattern of "compile once, allocate
// float textures, render fullscreen triangles or attributeless points
// into (possibly multiple) render targets."

export function compileShader(gl, type, src) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, src);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw new Error(`Shader compile error:\n${log}`);
  }
  return shader;
}

export function createProgram(gl, vertSrc, fragSrc) {
  const vs = compileShader(gl, gl.VERTEX_SHADER, vertSrc);
  const fs = compileShader(gl, gl.FRAGMENT_SHADER, fragSrc);
  const program = gl.createProgram();
  gl.attachShader(program, vs);
  gl.attachShader(program, fs);
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const log = gl.getProgramInfoLog(program);
    throw new Error(`Program link error:\n${log}`);
  }
  gl.deleteShader(vs);
  gl.deleteShader(fs);

  // Scan active uniforms so setUniform() can dispatch to the right
  // gl.uniformNfv/iv call by type, the way moderngl's per-uniform
  // `.value =` setter does via its own GLSL reflection. Uniforms the
  // shader doesn't actually reference are absent here (GLSL strips
  // them) -- setUniform() silently no-ops for those, same guard
  // pipeline.py's _draw_fullscreen/_draw_scatter use.
  const uniforms = {};
  const n = gl.getProgramParameter(program, gl.ACTIVE_UNIFORMS);
  for (let i = 0; i < n; i++) {
    const info = gl.getActiveUniform(program, i);
    const name = info.name.replace(/\[0\]$/, '');
    uniforms[name] = { location: gl.getUniformLocation(program, info.name), type: info.type, size: info.size };
  }
  return { program, uniforms };
}

export function setUniform(gl, uniforms, name, value) {
  const u = uniforms[name];
  if (!u) return;
  const { location, type } = u;
  const isArr = Array.isArray(value) || ArrayBuffer.isView(value);
  switch (type) {
    case gl.FLOAT:
      isArr ? gl.uniform1fv(location, value) : gl.uniform1f(location, value);
      return;
    case gl.INT:
    case gl.SAMPLER_2D:
      isArr ? gl.uniform1iv(location, value) : gl.uniform1i(location, value | 0);
      return;
    case gl.BOOL:
      isArr ? gl.uniform1iv(location, value) : gl.uniform1i(location, value ? 1 : 0);
      return;
    case gl.FLOAT_VEC2: gl.uniform2fv(location, value); return;
    case gl.FLOAT_VEC3: gl.uniform3fv(location, value); return;
    case gl.FLOAT_VEC4: gl.uniform4fv(location, value); return;
    case gl.INT_VEC2: gl.uniform2iv(location, value); return;
    case gl.FLOAT_MAT3: gl.uniformMatrix3fv(location, false, value); return;
    default:
      throw new Error(`unhandled uniform type 0x${type.toString(16)} for ${name}`);
  }
}

export function setUniforms(gl, uniforms, obj) {
  for (const [k, v] of Object.entries(obj)) setUniform(gl, uniforms, k, v);
}

// Texture-valued uniforms get auto-assigned consecutive texture units,
// starting fresh at 0 on *every call* -- this is the fix for a real bug
// hit while building the Python pipeline (pipeline.py's own comments
// call it out): a texture-unit counter that persists across draws
// eventually exceeds the GPU's available units and silently corrupts
// later draws' texture bindings. Scoping `unit` to this call's local
// variable, like pipeline.py's `_set_uniform` resetting `prog._next_unit`
// per draw, avoids that entirely.
export function applyUniforms(gl, progObj, uniformValues) {
  let unit = 0;
  for (const [name, value] of Object.entries(uniformValues)) {
    if (value && typeof value === 'object' && 'tex' in value) {
      bindTextureUnit(gl, unit, value);
      setUniform(gl, progObj.uniforms, name, unit);
      unit++;
    } else {
      setUniform(gl, progObj.uniforms, name, value);
    }
  }
}

// One fullscreen-triangle draw: apply uniforms, bind the target FBO
// (optionally a sub-rectangle viewport, for entropy_1d.frag's
// per-candidate single-texel writes), draw. Mirrors
// wallpaper/pipeline.py's `_draw_fullscreen`.
export function runFullscreen(gl, vao, progObj, targetFboObj, uniformValues, viewport = null) {
  gl.useProgram(progObj.program);
  applyUniforms(gl, progObj, uniformValues);
  const [x, y, w, h] = viewport ?? [0, 0, targetFboObj.w, targetFboObj.h];
  drawFullscreenTriangleViewport(gl, vao, targetFboObj.fbo, x, y, w, h);
}

// One GL_POINTS scatter draw (additive-blended). Mirrors
// wallpaper/pipeline.py's `_draw_scatter`.
export function runScatter(gl, vao, progObj, targetFboObj, uniformValues, numPoints) {
  gl.useProgram(progObj.program);
  applyUniforms(gl, progObj, uniformValues);
  drawScatterPoints(gl, vao, targetFboObj.fbo, targetFboObj.w, targetFboObj.h, numPoints);
}

const FLOAT_FORMATS = {
  1: (gl) => ({ internalFormat: gl.R32F, format: gl.RED }),
  2: (gl) => ({ internalFormat: gl.RG32F, format: gl.RG }),
  3: (gl) => ({ internalFormat: gl.RGB32F, format: gl.RGB }),
  4: (gl) => ({ internalFormat: gl.RGBA32F, format: gl.RGBA }),
};

// Float texture -- used for every intermediate/render-target texture in
// all three pipelines, and for the segmentation map (needs raw integer
// IDs preserved, not the [0,1] normalisation a plain 8-bit texture would
// apply -- see web/pipeline.js's upload code).
export function createTextureF(gl, w, h, components, data /* Float32Array | null */) {
  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tex);
  const { internalFormat, format } = FLOAT_FORMATS[components](gl);
  gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, w, h, 0, format, gl.FLOAT, data ?? null);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  return { tex, w, h, components, float: true };
}

// 8-bit texture -- used only for the initial image (RGBA) and depth map
// (R) uploads, where the automatic /255 normalisation on sample is
// exactly the desired semantics (u_image wants [0,1] colour; the depth
// contract is literally "pixel value/255 = normalised disparity").
export function createTextureU8(gl, w, h, components, data /* Uint8Array | null */) {
  const internalFormat = components === 4 ? gl.RGBA8 : gl.R8;
  const format = components === 4 ? gl.RGBA : gl.RED;
  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, w, h, 0, format, gl.UNSIGNED_BYTE, data ?? null);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  return { tex, w, h, components, float: false };
}

// Convenience: allocate a float render-target texture and its
// single-attachment FBO together, the most common pattern (mirrors
// `self._fbo(self._tex(size, components))` throughout the Python
// pipelines). Returns { tex: <texture obj>, fbo: <fbo obj with .fbo/.w/.h> }.
export function texAndFbo(gl, w, h, components) {
  const tex = createTextureF(gl, w, h, components, null);
  return { tex, fbo: createFramebuffer(gl, [tex]) };
}

export function createFramebuffer(gl, textures) {
  const fbo = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  const drawBuffers = textures.map((t, i) => gl.COLOR_ATTACHMENT0 + i);
  textures.forEach((t, i) => {
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0 + i, gl.TEXTURE_2D, t.tex, 0);
  });
  gl.drawBuffers(drawBuffers);
  const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
  if (status !== gl.FRAMEBUFFER_COMPLETE) {
    throw new Error(`Framebuffer incomplete: 0x${status.toString(16)}`);
  }
  return { fbo, w: textures[0].w, h: textures[0].h };
}

// Deleting a WebGL object twice (e.g. via a defensively-over-inclusive
// free list) is a documented no-op per spec, not an error -- so callers
// of these helpers are free to over-push into a free list rather than
// track exact single ownership.
export function deleteTexture(gl, texObj) {
  if (texObj) gl.deleteTexture(texObj.tex);
}

export function deleteFbo(gl, fboObj) {
  if (fboObj) gl.deleteFramebuffer(fboObj.fbo);
}

// Frees the { tex, fbo } pair returned by texAndFbo().
export function deleteTexAndFbo(gl, obj) {
  if (!obj) return;
  deleteTexture(gl, obj.tex);
  deleteFbo(gl, obj.fbo);
}

export function bindTextureUnit(gl, unit, tex) {
  gl.activeTexture(gl.TEXTURE0 + unit);
  gl.bindTexture(gl.TEXTURE_2D, tex.tex);
}

// Attributeless draws (gl_VertexID / gl_FragCoord only) still need some
// VAO bound in WebGL2 -- one empty VAO, shared by every pass.
export function createEmptyVao(gl) {
  return gl.createVertexArray();
}

export function drawFullscreenTriangle(gl, vao, targetFbo, w, h) {
  gl.bindFramebuffer(gl.FRAMEBUFFER, targetFbo);
  gl.viewport(0, 0, w, h);
  gl.disable(gl.BLEND);
  gl.bindVertexArray(vao);
  gl.drawArrays(gl.TRIANGLES, 0, 3);
}

export function drawFullscreenTriangleViewport(gl, vao, targetFbo, x, y, w, h) {
  gl.bindFramebuffer(gl.FRAMEBUFFER, targetFbo);
  gl.viewport(x, y, w, h);
  gl.disable(gl.BLEND);
  gl.bindVertexArray(vao);
  gl.drawArrays(gl.TRIANGLES, 0, 3);
}

export function drawScatterPoints(gl, vao, targetFbo, w, h, numPoints) {
  gl.bindFramebuffer(gl.FRAMEBUFFER, targetFbo);
  gl.viewport(0, 0, w, h);
  gl.clearColor(0, 0, 0, 0);
  gl.clear(gl.COLOR_BUFFER_BIT);
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.ONE, gl.ONE);
  gl.bindVertexArray(vao);
  gl.drawArrays(gl.POINTS, 0, numPoints);
  gl.disable(gl.BLEND);
}

// Always reads back as RGBA/FLOAT (the one read combination WebGL2
// guarantees works against a floating-point framebuffer -- narrower
// formats' "implementation preferred" read combo isn't guaranteed
// portable) and lets the caller slice out the channels it needs.
export function readFramebufferRGBA(gl, fboObj) {
  const { fbo, w, h } = fboObj;
  gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  const out = new Float32Array(w * h * 4);
  gl.readPixels(0, 0, w, h, gl.RGBA, gl.FLOAT, out);
  return out;
}
