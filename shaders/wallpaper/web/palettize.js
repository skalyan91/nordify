// JS/WebGL2 ports of additive_mix.frag and subtractive_mix.frag, both
// single self-contained passes. Palette-derived geometry (hull facets,
// K/S triangles) for every scheme in palettize.PALETTES is precomputed
// offline by build_palette_data.py into palette_data.js -- not computed
// live in the browser the way an earlier version of this file did (via
// Pyodide running palettize.py's own build_uniforms() functions
// in-browser; retired once the palette set settled into a fixed, known
// list, since paying a multi-second Pyodide+numpy CDN load and a live
// recompute on every switch stopped buying anything once "any palette
// works automatically" was no longer the goal).
//
// GLSL array sizes must be compile-time constants, and different
// palettes produce different facet counts (confirmed: NUM_FACES ranges
// 14-29 and N_PAL ranges 12-17 across the schemes in palettize.PALETTES)
// -- so each palette still needs its own compiled program. Rather than
// recompiling on every switch, prewarmAll() compiles one program per
// palette up front (at page load, from the precomputed geometry), and
// setPalette(name) becomes an O(1) map lookup with no GL calls at all.
//
// subtractive mixing was originally a 4-pass pipeline (snap -> blur ->
// blur -> resolve) that spatially blurred each pixel's simplex weight
// vector before reconstructing colour, meant to fix visible per-pixel
// jitter in smooth gradients. Removed: the blur radius needed to
// meaningfully smooth that jitter also visibly softened real image
// detail -- confirmed numerically before removing it that snap's own
// per-pixel output, fed straight to resolve with no blur in between, is
// byte-identical (max diff ~1e-5, float noise) to this single-pass
// shader's own output, since resolve's simplex reprojection is a no-op
// on weights already sitting on the simplex. That made the whole 4-pass
// split redundant with this file once blur was gone, not just
// blurry -- so it was retired rather than kept as a slower way to
// compute the same thing.

import { FULLSCREEN_VERT, MIXING } from './shaders.js';
import { createProgram, createEmptyVao, createTextureF, texAndFbo, runFullscreen } from './gl-utils.js';

const flat = (nested) => new Float32Array(nested.flat(Infinity));

// Substitutes `const int NAME = <n>;` declarations in a shader source
// string with the palette-specific counts computed for this call --
// e.g. { NUM_FACES: 29 } turns `const int NUM_FACES = 24;` (whatever the
// canonical .frag file's own literal happens to be, valid GLSL on its
// own for Nord) into `const int NUM_FACES = 29;`. Throws if a name isn't
// found, rather than silently compiling against the wrong array size.
export function withConstInts(source, replacements) {
  let out = source;
  for (const [name, value] of Object.entries(replacements)) {
    const re = new RegExp(`const int ${name} = \\d+;`);
    if (!re.test(out)) {
      throw new Error(`withConstInts: no "const int ${name} = N;" found to substitute`);
    }
    out = out.replace(re, `const int ${name} = ${value};`);
  }
  return out;
}

export class AdditivePipeline {
  constructor(gl) {
    this.gl = gl;
    this.vao = createEmptyVao(gl);
    this._programs = new Map();   // palette name -> { prog, uniforms }
    this.prog = null;
    this.uniforms = null;
  }

  // geometryMap: { paletteName: <additive geometry> } -- the `additive`
  // half of each entry in palette_data.js's PALETTE_GEOMETRY. Compiles
  // one program per palette up front; call once at startup. setPalette()
  // afterwards is just a lookup into what this builds.
  prewarmAll(geometryMap) {
    const gl = this.gl;
    for (const [name, geometry] of Object.entries(geometryMap)) {
      const src = withConstInts(MIXING.additive_mix_frag, { NUM_FACES: geometry.num_faces });
      const prog = createProgram(gl, FULLSCREEN_VERT, src);
      const uniforms = {
        u_rgb2lms: flat(geometry.rgb2lms),
        u_lms2oklab: flat(geometry.lms2oklab),
        u_hullEqs: flat(geometry.hull_eqs),
        u_V0: flat(geometry.V0), u_U: flat(geometry.U), u_Wv: flat(geometry.Wv),
        u_L0: flat(geometry.L0), u_P: flat(geometry.P), u_Q: flat(geometry.Q),
      };
      this._programs.set(name, { prog, uniforms });
    }
  }

  // Switches the active program/uniforms to the palette prewarmAll()
  // already compiled for `name`. No GL work here -- call before the
  // first run() and again whenever the selected palette changes.
  setPalette(name) {
    const entry = this._programs.get(name);
    if (!entry) {
      throw new Error(`AdditivePipeline.setPalette: no precompiled program for '${name}' `
        + '-- call prewarmAll() with geometry for this palette first (see palette_data.js).');
    }
    this.prog = entry.prog;
    this.uniforms = entry.uniforms;
  }

  run(imageTex, size) {
    if (!this.prog) throw new Error('call setPalette() first');
    const gl = this.gl;
    const out = texAndFbo(gl, size[0], size[1], 4);
    runFullscreen(gl, this.vao, this.prog, out.fbo, { u_image: imageTex, ...this.uniforms });
    return out;
  }
}

export class SubtractivePipeline {
  constructor(gl) {
    this.gl = gl;
    this.vao = createEmptyVao(gl);
    this._programs = new Map();   // palette name -> { prog, uniforms, kmTex, cmfTex }
    this.prog = null;
    this.uniforms = null;
  }

  // geometryMap: { paletteName: <subtractive geometry> } -- the
  // `subtractive` half of each entry in palette_data.js's
  // PALETTE_GEOMETRY. Compiles one program (and its K/S textures) per
  // palette up front; call once at startup.
  prewarmAll(geometryMap) {
    const gl = this.gl;
    for (const [name, geometry] of Object.entries(geometryMap)) {
      const src = withConstInts(MIXING.subtractive_mix_frag, {
        NUM_FACES: geometry.num_faces, N_PAL: geometry.n_pal,
      });
      const prog = createProgram(gl, FULLSCREEN_VERT, src);

      const numFaces = geometry.km_triangles_texture.length;
      const numBands = geometry.km_triangles_texture[0].length;
      const kmTex = createTextureF(gl, numBands, numFaces, 3, flat(geometry.km_triangles_texture));
      const cmfTex = createTextureF(gl, numBands, 1, 3, flat(geometry.d65n_cmf_texture));
      const uniforms = {
        u_rgb2lms: flat(geometry.rgb2lms),
        u_lms2oklab: flat(geometry.lms2oklab),
        u_xyz2rgb: flat(geometry.xyz2rgb),
        u_paletteRgb: flat(geometry.palette_rgb),
        u_paletteOklab: flat(geometry.palette_oklab),
        u_kmTriangles: kmTex,
        u_d65nCmf: cmfTex,
      };
      this._programs.set(name, { prog, uniforms, kmTex, cmfTex });
    }
  }

  // Switches the active program/uniforms/textures to the palette
  // prewarmAll() already compiled for `name`. No GL work here.
  setPalette(name) {
    const entry = this._programs.get(name);
    if (!entry) {
      throw new Error(`SubtractivePipeline.setPalette: no precompiled program for '${name}' `
        + '-- call prewarmAll() with geometry for this palette first (see palette_data.js).');
    }
    this.prog = entry.prog;
    this.uniforms = entry.uniforms;
  }

  // imageTex: gl-utils texture (RGBA, sRGB [0,1]). size: [W, H].
  run(imageTex, size) {
    if (!this.prog) throw new Error('call setPalette() first');
    const gl = this.gl;
    const out = texAndFbo(gl, size[0], size[1], 4);
    runFullscreen(gl, this.vao, this.prog, out.fbo, { u_image: imageTex, ...this.uniforms });
    return out;
  }
}
