// JS/WebGL2 port of shaders/night/pipeline.py's NightPipeline. Like the
// subtractive-mixing pipeline, this has no live parameter of its own --
// run(imageTex, size) redoes the whole 24-draw pipeline every call
// (detection runs at reduced resolution, see DOWNSAMPLE below).

import { FULLSCREEN_VERT, NIGHT } from './shaders.js';
import { createProgram, createEmptyVao, texAndFbo, runFullscreen, deleteTexAndFbo } from './gl-utils.js';

const SIGMAS = [2.5, 4.0, 6.0, 10.0, 16.0, 24.0];
const THRESHOLD = 0.12;
const STRENGTH_SCALE = 3.0;
const SPREAD = 1.5;
const LIGHT_BOOST = 0.2;

const LEVEL_SIGMAS = [];
for (let i = 0; i < SIGMAS.length - 1; i++) LEVEL_SIGMAS.push(Math.sqrt(SIGMAS[i] * SIGMAS[i + 1]));

// Detection (blur stack + DoG + protection map) runs at 1/DOWNSAMPLE
// resolution -- see run()'s own comment for why. Not exposed as a public
// knob: 2 was validated (timing + peak-detection accuracy) against the
// full-resolution reference; a larger factor risks losing the smallest
// detection scale (sigma=2.5, already near the documented noise floor
// where sub-2px scales pick up specular glints instead of real lights --
// see palettize.py's _dog_light_peaks docs) to downsample blur entirely.
const DOWNSAMPLE = 2;

const PASSES = ['oklab_convert', 'minmax_seed', 'minmax_reduce',
  'downsample2x', 'upsample_bilinear',
  'gaussian_blur_h', 'gaussian_blur_v', 'dog_nms', 'combine_max', 'nighttime_resolve'];

export class NightPipeline {
  constructor(gl) {
    this.gl = gl;
    this.prog = {};
    for (const name of PASSES) {
      this.prog[name] = createProgram(gl, FULLSCREEN_VERT, NIGHT[`${name}_frag`]);
    }
    this.vao = createEmptyVao(gl);
  }

  // Pushes every *intermediate* step into `toFree` (including the seed --
  // it gets consumed by the first iteration same as any later step), but
  // NOT the final returned value: prepare() keeps that one as persistent
  // state (this.bMinMax), read by every later resolve() call, so it must
  // survive past this function's own cleanup. (An earlier version pushed
  // every `dst` unconditionally, including the one that becomes the final
  // return value -- harmless when bMinMax was only ever used once within
  // the same call that computed it, but a real bug once prepare()/
  // resolve() split it into persistent state: prepare()'s own
  // end-of-call cleanup would delete this.bMinMax the instant it was
  // assigned, and every subsequent resolve() call would bind an already-
  // deleted texture. Matches wallpaper.js's _reduce2dMinMax, which never
  // had this bug.)
  _minmaxB(oklabFbo, W, H, toFree) {
    const gl = this.gl;
    const seed = texAndFbo(gl, W, H, 2);
    runFullscreen(gl, this.vao, this.prog.minmax_seed, seed.fbo, { u_oklab: oklabFbo.tex });
    let cur = seed, curW = W, curH = H;
    while (curW > 1 || curH > 1) {
      const nw = Math.max(1, Math.ceil(curW / 2));
      const nh = Math.max(1, Math.ceil(curH / 2));
      const dst = texAndFbo(gl, nw, nh, 2);
      runFullscreen(gl, this.vao, this.prog.minmax_reduce, dst.fbo,
        { u_src: cur.tex, u_srcSize: [curW, curH] });
      toFree.push(cur);
      cur = dst; curW = nw; curH = nh;
    }
    return cur;
  }

  _blur(srcFbo, size, sigma, normalize, toFree) {
    const gl = this.gl;
    const [W, H] = size;
    const radius = Math.ceil(3.0 * sigma);
    const hOut = texAndFbo(gl, W, H, 4);
    toFree.push(hOut);
    runFullscreen(gl, this.vao, this.prog.gaussian_blur_h, hOut.fbo,
      { u_src: srcFbo.tex, u_imageSize: [W, H], u_sigma: sigma, u_radius: radius, u_normalize: normalize });
    const vOut = texAndFbo(gl, W, H, 4);
    toFree.push(vOut);
    runFullscreen(gl, this.vao, this.prog.gaussian_blur_v, vOut.fbo,
      { u_src: hOut.tex, u_imageSize: [W, H], u_sigma: sigma, u_radius: radius, u_normalize: normalize });
    return vOut;
  }

  // imageTex: gl-utils texture object (RGBA, sRGB [0,1]). size: [W, H].
  // Runs everything independent of `amount`: oklab conversion and the
  // downsampled multi-scale DoG light-peak detection, ending in a full-
  // resolution protection map. Stores the results on `this`; resolve()
  // (cheap, ~0.1ms measured -- see its own comment) reads them, as many
  // times as you like for different `amount` values, without rerunning
  // any of this. Call this once per image (or whenever the image
  // changes) -- not on every tick of a live amount slider.
  prepare(imageTex, size) {
    const gl = this.gl;
    const [W, H] = size;
    const toFree = [];

    // A second prepare() call replaces this state -- free the PREVIOUS
    // generation now, and drop the previous resolve() output too, since
    // it read the old oklab/protectMap and is about to be superseded.
    if (this._prepared) {
      deleteTexAndFbo(gl, this.oklab);
      deleteTexAndFbo(gl, this.protectMap);
      deleteTexAndFbo(gl, this.bMinMax);
    }
    if (this._lastResolveOut) {
      deleteTexAndFbo(gl, this._lastResolveOut);
      this._lastResolveOut = null;
    }

    const oklab = texAndFbo(gl, W, H, 4);
    runFullscreen(gl, this.vao, this.prog.oklab_convert, oklab.fbo, { u_image: imageTex });
    this.oklab = oklab;

    this.bMinMax = this._minmaxB(oklab, W, H, toFree);

    // Detection (every blur + DoG + protection-map pass below) runs at
    // 1/DOWNSAMPLE resolution -- confirmed by profiling that the 11
    // separable-Gaussian passes here (radii up to ~90px at full
    // resolution) dominate this pipeline's total cost, since blur cost
    // scales as pixels x radius. Downsampling once, halving every sigma
    // to match (same *physical* detection scale, fewer pixels and a
    // proportionally smaller radius), then upsampling only the final
    // protect map back to full resolution keeps oklab_convert/_minmaxB/
    // nighttime_resolve untouched (they still see the real full-resolution
    // image) while cutting the dominant cost roughly DOWNSAMPLE^3
    // (DOWNSAMPLE^2 fewer pixels x DOWNSAMPLE smaller radius per blur pass).
    const dW = Math.max(1, Math.ceil(W / DOWNSAMPLE));
    const dH = Math.max(1, Math.ceil(H / DOWNSAMPLE));
    const dSize = [dW, dH];
    const oklabSmall = texAndFbo(gl, dW, dH, 4);
    toFree.push(oklabSmall);
    runFullscreen(gl, this.vao, this.prog.downsample2x, oklabSmall.fbo, { u_src: oklab.tex, u_srcSize: [W, H] });

    const blurred = SIGMAS.map((s) => this._blur(oklabSmall, dSize, s / DOWNSAMPLE, true, toFree));

    const peaks = [];
    for (let level = 0; level < 5; level++) {
      const peak = texAndFbo(gl, dW, dH, 4);
      toFree.push(peak);
      runFullscreen(gl, this.vao, this.prog.dog_nms, peak.fbo, {
        u_blur0: blurred[0].tex, u_blur1: blurred[1].tex, u_blur2: blurred[2].tex,
        u_blur3: blurred[3].tex, u_blur4: blurred[4].tex, u_blur5: blurred[5].tex,
        u_imageSize: dSize, u_level: level, u_threshold: THRESHOLD, u_strengthScale: STRENGTH_SCALE,
      });
      peaks.push(peak);
    }

    const protects = peaks.map((p, i) => this._blur(p, dSize, (LEVEL_SIGMAS[i] * SPREAD) / DOWNSAMPLE, false, toFree));

    const protectMapSmall = texAndFbo(gl, dW, dH, 4);
    toFree.push(protectMapSmall);
    runFullscreen(gl, this.vao, this.prog.combine_max, protectMapSmall.fbo, {
      u_p0: protects[0].tex, u_p1: protects[1].tex, u_p2: protects[2].tex,
      u_p3: protects[3].tex, u_p4: protects[4].tex,
    });

    const protectMap = texAndFbo(gl, W, H, 4);
    runFullscreen(gl, this.vao, this.prog.upsample_bilinear, protectMap.fbo, {
      u_src: protectMapSmall.tex, u_srcSize: dSize, u_dstSize: [W, H],
    });
    this.protectMap = protectMap;

    for (const obj of toFree) deleteTexAndFbo(gl, obj);
    this._prepared = true;
  }

  // Cheap, single-pass: applies the darken/cool (or brighten/warm)
  // transform at `amount` (continuous in [-1, 1] -- see
  // nighttime_resolve.frag's own header comment) using prepare()'s
  // cached detection results. Measured ~0.1ms in isolation (vs. ~100ms+
  // for the detection stage prepare() runs) -- genuinely cheap enough to
  // call on every tick of a live "night <-> day" slider. Call prepare()
  // first. Frees the PREVIOUS resolve() output before returning this
  // one, same one-generation-delayed pattern wallpaper.js's render() uses.
  resolve(amount = 1.0) {
    if (!this._prepared) throw new Error('call prepare() first');
    const gl = this.gl;
    if (this._lastResolveOut) deleteTexAndFbo(gl, this._lastResolveOut);
    const { w: W, h: H } = this.oklab.fbo;
    const out = texAndFbo(gl, W, H, 4);
    runFullscreen(gl, this.vao, this.prog.nighttime_resolve, out.fbo, {
      u_oklab: this.oklab.tex, u_protect: this.protectMap.tex, u_bMinMax: this.bMinMax.tex,
      u_lightBoost: LIGHT_BOOST, u_amount: amount,
    });
    this._lastResolveOut = out;
    return out;
  }

  // Convenience: prepare() + resolve() in one call, for one-shot use
  // (warmup, or anywhere the caller doesn't need to vary `amount` live
  // without redoing detection -- see demo.js's rebuildColor() vs.
  // onNightAmountInput() for the two different call patterns).
  run(imageTex, size, amount = 1.0) {
    this.prepare(imageTex, size);
    return this.resolve(amount);
  }
}
