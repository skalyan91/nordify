// JS/WebGL2 port of shaders/wallpaper/pipeline.py's WallpaperPipeline.
// Mechanical translation -- same 14 passes, same pass order, same
// setup()/render() split; see gl-utils.js for the moderngl-call
// equivalents and pipeline.py's own module docstring for why the split
// exists (setup is the expensive, multi-pass, "recompute only when the
// image/target-size changes" half; render is the cheap composite pass
// meant to rerun on every focal-depth change).

import { FULLSCREEN_VERT, WALLPAPER } from './shaders.js';
import {
  createProgram, createEmptyVao, texAndFbo, createFramebuffer,
  runFullscreen, runScatter, readFramebufferRGBA, deleteTexAndFbo,
} from './gl-utils.js';

const NUM_BINS = 256;
const MAX_SEGMENTS = 64;
const NUM_CANDIDATES = 32;
const MAX_K = 8;

const FULLSCREEN_PASSES = [
  'minmax_reduce', 'argmax_1d', 'median_1d',
  'luminance_blur_h', 'luminance_blur_v', 'gradient_mag',
  'prefix_sum_1d', 'weighted_mag_for_offset', 'reduce_pairwise',
  'entropy_seed', 'entropy_1d', 'combine_and_argmin',
  'gmm_init', 'gmm_iterate', 'sort_and_bounds',
  'composite', 'minmax_seed',
];
const SCATTER_PASSES = ['segment_score', 'depth_hist_masked', 'figure_count_1d', 'depth_hist_cropped'];

export class WallpaperPipeline {
  constructor(gl) {
    this.gl = gl;
    this.prog = {};
    for (const name of FULLSCREEN_PASSES) {
      this.prog[name] = createProgram(gl, FULLSCREEN_VERT, WALLPAPER[`${name}_frag`]);
    }
    for (const name of SCATTER_PASSES) {
      this.prog[name] = createProgram(gl, WALLPAPER[`${name}_vert`], WALLPAPER[`${name}_frag`]);
    }
    this.vao = createEmptyVao(gl);
    this._setupDone = false;
  }

  // Each of these three reduction helpers pushes every *intermediate*
  // step (including the seed/src passed in -- it gets consumed by the
  // first iteration same as any later step) into `toFree`, but does NOT
  // push the final returned result: some call sites keep that result as
  // persistent pipeline state (this.depthMinMax etc, reused by render()),
  // others are purely transient within setup() and push the result into
  // `toFree` themselves right after getting it back. See setup()'s own
  // comments at each call site.
  _reduce2dMinMax(seedFbo, w, h, toFree) {
    const gl = this.gl;
    let cur = seedFbo, curW = w, curH = h;
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

  _reduce1dSum(srcFbo, length, other, axisIsX, toFree) {
    const gl = this.gl;
    let cur = srcFbo, curLen = length;
    while (curLen > 1) {
      const nextLen = Math.max(1, Math.ceil(curLen / 2));
      const size = axisIsX ? [nextLen, other] : [other, nextLen];
      const srcSize = axisIsX ? [curLen, other] : [other, curLen];
      const dst = texAndFbo(gl, size[0], size[1], 4);
      runFullscreen(gl, this.vao, this.prog.reduce_pairwise, dst.fbo,
        { u_src: cur.tex, u_srcSize: srcSize, u_axisIsX: axisIsX ? 1 : 0 });
      toFree.push(cur);
      cur = dst; curLen = nextLen;
    }
    return cur;
  }

  _prefixSum1d(srcFbo, length, toFree) {
    const gl = this.gl;
    let cur = srcFbo, step = 1;
    while (step < length) {
      const dst = texAndFbo(gl, length, 1, 4);
      runFullscreen(gl, this.vao, this.prog.prefix_sum_1d, dst.fbo, { u_src: cur.tex, u_step: step });
      toFree.push(cur);
      cur = dst; step *= 2;
    }
    return cur;
  }

  // cropImageTex/depth_tex/segmentation_tex: gl-utils texture objects.
  // image_size/target_size: [W, H]. ratio_w/ratio_h: target aspect.
  //
  // cropImageTex should be the ORIGINAL, unfiltered source image -- every
  // decision this call makes (crop offset, figure identification, the
  // figure's own median depth used as the default focal plane, and the
  // GMM depth layers) is geometry, and is decided here, once, from
  // that original image plus depth/segmentation, *before* any night or
  // palette-mixing colour filtering runs. Colour filters change what a pixel
  // looks like, never where the figure is or how the frame should be
  // cropped, so letting them run first would make crop/focus decisions
  // depend on which filters happen to be enabled -- e.g. palette mixing's
  // palette-snapping measurably changes the image's own gradient
  // magnitude, which is exactly what the crop search scores candidates
  // on. render() below takes the (possibly filtered) colour source as
  // its own separate parameter instead of one fixed here, so toggling a
  // filter never needs to rerun any of this.
  setup(cropImageTex, depthTex, segmentationTex, imageSize, targetSize, ratioW, ratioH, opts = {}) {
    const gl = this.gl;

    // A second setup() call (a filter toggle, a new upload, a target-size
    // change) replaces every persistent object below -- free the PREVIOUS
    // generation now, before it's unreachable, rather than leaking a full
    // setup()'s worth of GPU objects on every rebuild. Also drop the
    // previous render() output, since it sampled the old persistent state
    // and is about to be superseded by this setup() + the render() call
    // that follows it.
    if (this._setupDone) {
      deleteTexAndFbo(gl, this.depthMinMax);
      deleteTexAndFbo(gl, this.figureInfo);
      deleteTexAndFbo(gl, this.figureMedianDepth);
      deleteTexAndFbo(gl, this.cropOffset);
      deleteTexAndFbo(gl, this.layerCenters);
    }
    if (this._lastRenderOut) {
      deleteTexAndFbo(gl, this._lastRenderOut);
      this._lastRenderOut = null;
    }

    // Every OTHER texture/FBO this call allocates is purely transient
    // (consumed by a later pass within this same setup() call and never
    // touched again) -- collected here and freed in one pass at the end,
    // rather than kept alive for the pipeline's whole lifetime the way
    // they were before this fix.
    const toFree = [];

    const kLayers = opts.kLayers ?? 5;
    const emIters = opts.emIters ?? 8;
    const figurePenaltyWeight = opts.figurePenaltyWeight ?? 4.0;
    const viewingDistanceFactor = opts.viewingDistanceFactor ?? 1.5;
    const e2Degrees = opts.e2Degrees ?? 2.3;
    const gp = { u_viewingDistanceFactor: viewingDistanceFactor, u_e2Degrees: e2Degrees };
    const [W, H] = imageSize;

    // Pass 0: depth min/max (persistent -- reused by render()/sampleDepthAt()).
    const seed = texAndFbo(gl, W, H, 2);
    runFullscreen(gl, this.vao, this.prog.minmax_seed, seed.fbo, { u_depth: depthTex });
    this.depthMinMax = this._reduce2dMinMax(seed, W, H, toFree);

    // Passes 1-2: figure identification (persistent).
    const segAccum = texAndFbo(gl, MAX_SEGMENTS, 1, 4);
    toFree.push(segAccum);
    runScatter(gl, this.vao, this.prog.segment_score, segAccum.fbo, {
      u_depth: depthTex, u_segmentation: segmentationTex,
      u_depthMinMax: this.depthMinMax.tex, u_imageSize: [W, H], ...gp,
    }, W * H);
    this.figureInfo = texAndFbo(gl, 1, 1, 4);
    runFullscreen(gl, this.vao, this.prog.argmax_1d, this.figureInfo.fbo, {
      u_accum: segAccum.tex, u_count: MAX_SEGMENTS, u_startIndex: 1, u_findMax: true,
    });

    // Passes 3-4: figure median depth (persistent).
    const depthHist = texAndFbo(gl, NUM_BINS, 1, 4);
    toFree.push(depthHist);
    runScatter(gl, this.vao, this.prog.depth_hist_masked, depthHist.fbo, {
      u_depth: depthTex, u_segmentation: segmentationTex, u_depthMinMax: this.depthMinMax.tex,
      u_figureInfo: this.figureInfo.tex, u_imageSize: [W, H],
    }, W * H);
    this.figureMedianDepth = texAndFbo(gl, 1, 1, 4);
    runFullscreen(gl, this.vao, this.prog.median_1d, this.figureMedianDepth.fbo, { u_hist: depthHist.tex });

    // Crop axis/lengths -- plain host arithmetic on known sizes (entropy_crop.py:182-189).
    let axisIsX, length, other, newLength;
    if (W * ratioH > H * ratioW) {
      axisIsX = true; length = W; other = H;
      newLength = Math.floor((other * ratioW) / ratioH);
    } else {
      axisIsX = false; length = H; other = W;
      newLength = Math.floor((other * ratioH) / ratioW);
    }
    newLength = Math.max(1, Math.min(newLength, length));
    const excess = length - newLength;
    this.cropAxisIsX = axisIsX;
    this.newLength = newLength;

    // Pass 5: gradient magnitude of the crop edge-source (the image).
    const grayH = texAndFbo(gl, W, H, 1);
    toFree.push(grayH);
    runFullscreen(gl, this.vao, this.prog.luminance_blur_h, grayH.fbo, { u_image: cropImageTex, u_imageSize: [W, H] });
    const grayHV = texAndFbo(gl, W, H, 1);
    toFree.push(grayHV);
    runFullscreen(gl, this.vao, this.prog.luminance_blur_v, grayHV.fbo, { u_gray: grayH.tex, u_imageSize: [W, H] });
    const gradMag = texAndFbo(gl, W, H, 1);
    toFree.push(gradMag);
    runFullscreen(gl, this.vao, this.prog.gradient_mag, gradMag.fbo, { u_grayBlurred: grayHV.tex, u_imageSize: [W, H] });

    // Pass 6-7: figure-pixel count by crop-axis position, prefix-summed.
    const figCount = texAndFbo(gl, length, 1, 4);
    toFree.push(figCount);
    runScatter(gl, this.vao, this.prog.figure_count_1d, figCount.fbo, {
      u_segmentation: segmentationTex, u_figureInfo: this.figureInfo.tex, u_imageSize: [W, H],
      u_cropAxisIsX: axisIsX ? 1 : 0, u_cropAxisLength: length,
    }, W * H);
    const figPrefix = this._prefixSum1d(figCount, length, toFree);
    toFree.push(figPrefix);

    // Pass 8: per-candidate weighted-entropy reduction.
    let candidateOffsets;
    if (excess <= 0) {
      candidateOffsets = new Array(NUM_CANDIDATES).fill(0);
    } else {
      candidateOffsets = [];
      for (let i = 0; i < NUM_CANDIDATES; i++) {
        const x = (excess * i) / (NUM_CANDIDATES - 1);
        candidateOffsets.push(Math.round(x));
      }
    }
    const candidateScores = texAndFbo(gl, NUM_CANDIDATES, 1, 4);
    toFree.push(candidateScores);
    for (let c = 0; c < NUM_CANDIDATES; c++) {
      const offset = candidateOffsets[c];
      const weighted = texAndFbo(gl, newLength, other, 2);
      toFree.push(weighted);
      runFullscreen(gl, this.vao, this.prog.weighted_mag_for_offset, weighted.fbo, {
        u_gradMag: gradMag.tex, u_cropAxisIsX: axisIsX ? 1 : 0, u_offset: offset,
        u_newLength: newLength, u_other: other, ...gp,
      });
      const profile = this._reduce1dSum(weighted, newLength, other, true, toFree);
      toFree.push(profile);
      const seeded = texAndFbo(gl, 1, other, 4);
      toFree.push(seeded);
      runFullscreen(gl, this.vao, this.prog.entropy_seed, seeded.fbo, { u_profile: profile.tex });
      const moments = this._reduce1dSum(seeded, other, 1, false, toFree);
      toFree.push(moments);
      runFullscreen(gl, this.vao, this.prog.entropy_1d, candidateScores.fbo,
        { u_moments: moments.tex }, [c, 0, 1, 1]);
    }

    // Pass 9: combine entropy + figure-cut penalty, argmin (persistent).
    this.cropOffset = texAndFbo(gl, 1, 1, 4);
    runFullscreen(gl, this.vao, this.prog.combine_and_argmin, this.cropOffset.fbo, {
      u_candidateScores: candidateScores.tex, u_figureCountPrefix: figPrefix.tex,
      u_candidateOffsets: new Int32Array(candidateOffsets), u_newLength: newLength,
      u_cropAxisLength: length, u_maxEntropyBits: Math.log2(Math.max(other, 2)),
      u_figurePenaltyWeight: figurePenaltyWeight,
    });

    // Pass 10-13: adaptive GMM depth layers, restricted to the crop window.
    const croppedHist = texAndFbo(gl, NUM_BINS, 1, 4);
    toFree.push(croppedHist);
    runScatter(gl, this.vao, this.prog.depth_hist_cropped, croppedHist.fbo, {
      u_depth: depthTex, u_depthMinMax: this.depthMinMax.tex, u_cropOffset: this.cropOffset.tex,
      u_imageSize: [W, H], u_cropAxisIsX: axisIsX ? 1 : 0, u_newLength: newLength, ...gp,
    }, W * H);

    this.k = Math.min(kLayers, MAX_K);
    let components = texAndFbo(gl, this.k, 1, 4);   // (mean, variance, weight, _) per K
    toFree.push(components);
    runFullscreen(gl, this.vao, this.prog.gmm_init, components.fbo, { u_k: this.k });
    for (let i = 0; i < emIters; i++) {
      const nxt = texAndFbo(gl, this.k, 1, 4);
      toFree.push(nxt);
      runFullscreen(gl, this.vao, this.prog.gmm_iterate, nxt.fbo,
        { u_hist: croppedHist.tex, u_components: components.tex, u_k: this.k });
      components = nxt;
    }
    // this.layerCenters is persistent (reused by render()) -- not pushed.
    this.layerCenters = texAndFbo(gl, this.k, 1, 4);
    runFullscreen(gl, this.vao, this.prog.sort_and_bounds, this.layerCenters.fbo,
      { u_centroids: components.tex, u_k: this.k });

    for (const obj of toFree) deleteTexAndFbo(gl, obj);

    // Deliberately NOT storing cropImageTex -- render() takes the colour
    // source explicitly on every call instead (see this method's own
    // header comment).
    this.depthTex = depthTex;
    this.imageSize = imageSize;
    this.targetSize = targetSize;
    this._setupDone = true;
  }

  // colorTex: the texture to actually sample colour from -- the source
  // image, or night/palette-mixing's filtered output, chosen independently by
  // the caller on every call (see setup()'s header comment for why this
  // isn't fixed at setup() time). Must be the same size as the image
  // setup() was given. focalDepth: null to fall back to the figure's
  // median depth, else a number in [0,1]. Returns { fbo, w, h } (also
  // readable directly for display -- see web/pipeline.js's
  // blit-to-canvas helper).
  render(colorTex, focalDepth = null, sigmaMax = 24.0) {
    if (!this._setupDone) throw new Error('call setup() first');
    const gl = this.gl;
    // Free the *previous* render() call's output now that it's no longer
    // on screen (the caller blits it and moves on) -- otherwise every
    // slider tick during focal-depth dragging leaks one more full-size
    // RGBA32F texture+FBO.
    if (this._lastRenderOut) deleteTexAndFbo(gl, this._lastRenderOut);
    const out = texAndFbo(gl, this.targetSize[0], this.targetSize[1], 4);
    runFullscreen(gl, this.vao, this.prog.composite, out.fbo, {
      u_image: colorTex, u_depth: this.depthTex, u_depthMinMax: this.depthMinMax.tex,
      u_layerCenters: this.layerCenters.tex, u_cropOffset: this.cropOffset.tex,
      u_figureMedianDepth: this.figureMedianDepth.tex,
      u_imageSize: this.imageSize, u_targetSize: this.targetSize,
      u_cropAxisIsX: this.cropAxisIsX ? 1 : 0, u_newLength: this.newLength,
      u_k: this.k, u_sigmaMax: sigmaMax,
      u_focalDepth: focalDepth ?? 0.0, u_focalDepthIsSet: focalDepth !== null,
    });
    this._lastRenderOut = out;
    return out;
  }

  // Reads back the depth texture at one source pixel -- used by the
  // demo's click-to-focus UI. Cheap (a 1x1 readback via a tiny FBO), not
  // meant to run every frame.
  sampleDepthAt(px, py) {
    const gl = this.gl;
    // Attach the depth texture itself as an FBO colour target so a
    // single pixel can be read straight back -- readable this way since
    // EXT_color_buffer_float makes even an 8-bit texture a valid (if
    // unusual) read source, not just a render target. The read type has
    // to match how the texture was actually uploaded (web/demo.js
    // uploads depth as plain 8-bit via createTextureU8, matching the
    // depth contract's own /255 = normalised-disparity convention;
    // createTextureF's float path is also supported here for any other
    // caller that uploads depth as true float). Either way this
    // reproduces exactly the [0,1] value minmax_seed.frag's own
    // `texture(u_depth, v_uv).r` sample would give -- UNORM textures
    // normalise on GPU sample automatically, so the two must be read
    // back consistently for `raw` to land in the same space as
    // depthMinMax below.
    const depthFbo = createFramebuffer(gl, [this.depthTex]);
    gl.bindFramebuffer(gl.FRAMEBUFFER, depthFbo.fbo);
    let raw;
    if (this.depthTex.float) {
      const out = new Float32Array(4);
      gl.readPixels(px, py, 1, 1, gl.RGBA, gl.FLOAT, out);
      raw = out[0];
    } else {
      const out = new Uint8Array(4);
      gl.readPixels(px, py, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, out);
      raw = out[0] / 255.0;
    }
    gl.deleteFramebuffer(depthFbo.fbo);
    const mm = readFramebufferRGBA(gl, this.depthMinMax.fbo);
    const dmin = mm[0], dmax = mm[1];
    return dmax - dmin > 1e-6 ? (raw - dmin) / (dmax - dmin) : 0.0;
  }
}
