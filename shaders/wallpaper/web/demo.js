// App logic for the Wallpaper Ricer web demo. See README.md for the
// upload contract and shaders/wallpaper/README.md for the pipeline this
// drives.
//
// Two independent things happen here, not one filter chain:
//
// 1. GEOMETRY -- crop offset, figure identification, the figure's own
//    median depth (the default focal plane), and the GMM depth
//    layers -- is decided ONCE, from the ORIGINAL uploaded image, depth,
//    and segmentation map, via wallpaper.setup() (rebuildGeometry()
//    below). Night/palette-mixing colour filters never influence this:
//    they change what a pixel looks like, never where the figure is or
//    how the frame should be cropped, and letting a colour filter run
//    first would make crop/focus decisions depend on which filters
//    happen to be enabled (palette mixing in particular measurably
//    changes the image's own gradient magnitude -- exactly what the crop
//    search scores candidates on).
// 2. COLOUR -- night, then palette mixing, applied to the original
//    image -- produces the texture wallpaper.render() actually samples
//    (rebuildColor() below). Neither depends on focal depth, so they
//    only rerun when the *source image* changes or a filter/mode/amount
//    changes -- never on a focal-depth slider tick.
//
// Because wallpaper.render() takes its colour source as an explicit
// parameter on every call rather than one fixed at setup() time,
// changing a colour filter only reruns rebuildColor() (night + palette
// mixing + one composite pass, a few ms) -- never the geometry stage.
// That's also what makes night colouring's own `amount` a genuinely live
// slider (see night.js): it's just another rebuildColor() input, not a
// full pipeline re-setup.
//
// Which PALETTE the mixing shaders target (Nord, Solarized, Gruvbox,
// Everforest, Catppuccin, Dracula, ...) is a third, independent axis:
// palette_data.js precomputes every palette's geometry offline (see
// build_palette_data.py), and warmup() below precompiles one
// additive/subtractive program per palette up front, so switching the
// dropdown (onPaletteChange()) is just picking which already-compiled
// program to use -- never wallpaper's own crop/focus geometry, and never
// night colouring, both of which are palette-agnostic.
//
// Depth map and segmentation map are themselves optional uploads: they
// drive wallpaper.js's geometry stage alone (crop offset, figure
// identification, depth-guided blur). Without both, rebuildGeometry()
// below skips that stage entirely -- crop and focus are simply
// unavailable -- and rebuildColor()'s output (night + palette mixing
// applied to the original image) is blitted straight to the canvas
// instead of going through wallpaper.render().

import { WallpaperPipeline } from './wallpaper.js';
import { NightPipeline } from './night.js';
import { AdditivePipeline, SubtractivePipeline } from './palettize.js';
import { PALETTE_GEOMETRY } from './palette_data.js';
import { estimateDepth, estimateSegmentation } from './inference.js';
import { downloadShaderZip } from './shader_export.js';
import {
  createTextureU8, createTextureF, createProgram, createEmptyVao,
  runFullscreen, readFramebufferRGBA, deleteTexAndFbo,
} from './gl-utils.js';
import { FULLSCREEN_VERT } from './shaders.js';

// GLSL ES 300 requires the version/precision header every other shader
// gets from build_shaders.py's transform -- this one's small enough to
// just write directly rather than round-trip through the build step.
const BLIT_FRAG = `#version 300 es
precision highp float;
in vec2 v_uv;
out vec4 fragColor;
uniform sampler2D u_tex;
void main() {
  // WebGL displays the default framebuffer's NDC y=+1 at the *bottom*
  // of the visible canvas (confirmed empirically -- an unflipped blit
  // showed a top-red/bottom-blue test image upside down), unlike every
  // internal pass in this pipeline, which stays framebuffer-to-
  // framebuffer and never needs this correction. Only this final
  // blit-to-screen step flips.
  fragColor = texture(u_tex, vec2(v_uv.x, 1.0 - v_uv.y));
}
`;

function readPixelsToUint8(file) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement('canvas');
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0);
      const data = ctx.getImageData(0, 0, canvas.width, canvas.height);
      resolve({ data: data.data, width: canvas.width, height: canvas.height });
      URL.revokeObjectURL(img.src);
    };
    img.onerror = reject;
    img.src = URL.createObjectURL(file);
  });
}

class App {
  constructor() {
    this.canvas = document.getElementById('canvas');
    this.statusEl = document.getElementById('status');
    // preserveDrawingBuffer: the Download button reads the canvas back
    // (toBlob()) whenever the user happens to click it, not immediately
    // after a draw -- without this, the browser is free to clear/discard
    // the default framebuffer's contents right after compositing it to
    // the screen, so a later read can come back blank. Every *internal*
    // pass in these pipelines renders to its own offscreen FBO regardless
    // (see gl-utils.js), so this only affects the final blit-to-canvas
    // step and costs nothing there beyond what downloading needs.
    this.gl = this.canvas.getContext('webgl2', { preserveDrawingBuffer: true });
    if (!this.gl) return this.fail('WebGL2 is not supported in this browser.');
    if (!this.gl.getExtension('EXT_color_buffer_float')) {
      return this.fail('This browser supports WebGL2 but not the EXT_color_buffer_float extension, which every pass in this pipeline needs to render to a floating-point texture.');
    }

    const gl = this.gl;
    this.blitProg = createProgram(gl, FULLSCREEN_VERT, BLIT_FRAG);
    this.vao = createEmptyVao(gl);

    this.wallpaper = new WallpaperPipeline(gl);
    this.night = new NightPipeline(gl);
    this.additive = new AdditivePipeline(gl);
    this.subtractive = new SubtractivePipeline(gl);

    this.sources = null;   // { imageTex, depthTex, segTex, imageSize }
    this._currentPalette = null;

    this._bindUI();
  }

  // Two warmup costs, paid once at page load with an honest status
  // message rather than as a surprise hitch later:
  // 1. Compiling one additive + one subtractive program per palette in
  //    palette_data.js (precomputed offline -- see build_palette_data.py
  //    and palettize.js's own header comment) -- so every later
  //    onPaletteChange() is a free map lookup, never a recompile.
  // 2. Some GL drivers defer real shader compilation/optimisation to a
  //    program's *first* draw call rather than doing it at
  //    createProgram() time (confirmed here: enabling night colouring
  //    for the first time took ~4s wall-clock while the pipeline's own
  //    reported run() time was ~70ms -- every subsequent run is fast).
  //    Running each pipeline once on a throwaway 2x2 texture exercises
  //    that path upfront -- and since prewarmAll() above compiled a
  //    *separate* program per palette, this loops over every palette's
  //    additive/subtractive program too, not just the default one, so a
  //    later onPaletteChange() to any of them never eats that first-draw
  //    cost live.
  async warmup() {
    const gl = this.gl;
    this.setStatus(`Compiling shaders for ${Object.keys(PALETTE_GEOMETRY).length} palettes…`);
    await new Promise((r) => requestAnimationFrame(r));

    const additiveGeom = {}, subtractiveGeom = {};
    for (const [name, g] of Object.entries(PALETTE_GEOMETRY)) {
      additiveGeom[name] = g.additive;
      subtractiveGeom[name] = g.subtractive;
    }
    this.additive.prewarmAll(additiveGeom);
    this.subtractive.prewarmAll(subtractiveGeom);

    this.setStatus('Warming up shaders…');
    await new Promise((r) => requestAnimationFrame(r));
    const dummy = createTextureU8(gl, 2, 2, 4, new Uint8Array(16).fill(128));
    const t0 = performance.now();
    deleteTexAndFbo(gl, this.night.run(dummy, [2, 2]));
    for (const name of Object.keys(PALETTE_GEOMETRY)) {
      this.additive.setPalette(name);
      this.subtractive.setPalette(name);
      deleteTexAndFbo(gl, this.additive.run(dummy, [2, 2]));
      deleteTexAndFbo(gl, this.subtractive.run(dummy, [2, 2]));
    }
    gl.deleteTexture(dummy.tex);

    const defaultPalette = document.getElementById('paletteSelect').value;
    this.additive.setPalette(defaultPalette);
    this.subtractive.setPalette(defaultPalette);
    this._currentPalette = defaultPalette;

    this.setStatus(`Ready (warmup ${(performance.now() - t0).toFixed(0)}ms). `
      + 'Choose an image to begin (depth map + segmentation map are optional).');
  }

  fail(msg) {
    this.statusEl.textContent = `Error: ${msg}`;
    this.statusEl.classList.add('error');
    throw new Error(msg);
  }

  setStatus(msg) { this.statusEl.textContent = msg; }

  _bindUI() {
    document.getElementById('loadBtn').addEventListener('click', () => this.onLoad());
    document.getElementById('focalDepth').addEventListener('input', () => this.onFocalDepthInput());
    document.getElementById('focalAuto').addEventListener('change', () => this.onFocalDepthInput());
    // All colour-filter controls only ever touch rebuildColor() -- never
    // wallpaper's geometry (crop/figure/focus), decided once from the
    // original image in rebuildGeometry() and never revisited here.
    // The night-amount slider is the one exception: it goes through
    // onNightAmountInput() instead of the full rebuildColor(), so
    // dragging it doesn't rerun night's own detection stage (the
    // expensive part -- see night.js's prepare()/resolve() split and
    // this method's own comment).
    document.getElementById('nightEnabled').addEventListener('change', () => this.rebuildColor());
    document.getElementById('nightAmount').addEventListener('input', () => this.onNightAmountInput());
    document.getElementById('mixEnabled').addEventListener('change', () => this.rebuildColor());
    document.getElementById('mixMode').addEventListener('change', () => this.rebuildColor());
    document.getElementById('paletteSelect').addEventListener('change', () => this.onPaletteChange());
    this.canvas.addEventListener('click', (e) => this.onCanvasClick(e));
    document.getElementById('downloadBtn').addEventListener('click', () => this.onDownload());
    document.getElementById('downloadShadersBtn').addEventListener('click', () => this.onDownloadShaders());

    // Auto depth/segmentation and manual file uploads are mutually
    // exclusive (onLoad() only ever uses one source) -- disabling the
    // file inputs while auto mode is on makes that visible rather than
    // leaving a chosen file silently ignored.
    document.getElementById('autoDepthEnabled').addEventListener('change', () => {
      const auto = document.getElementById('autoDepthEnabled').checked;
      document.getElementById('depthFile').disabled = auto;
      document.getElementById('segFile').disabled = auto;
    });
  }

  // Switches both mixing pipelines to the palette warmup() already
  // precompiled a program for, then reruns rebuildColor() so the change
  // is visible immediately if an image is already loaded. No GL
  // recompile here -- see palettize.js's prewarmAll()/setPalette().
  onPaletteChange() {
    const name = document.getElementById('paletteSelect').value;
    if (name === this._currentPalette) return;
    this.additive.setPalette(name);
    this.subtractive.setPalette(name);
    this._currentPalette = name;
    this.rebuildColor();
  }

  async onLoad() {
    const imageFile = document.getElementById('imageFile').files[0];
    const depthFile = document.getElementById('depthFile').files[0];
    const segFile = document.getElementById('segFile').files[0];
    const autoDepth = document.getElementById('autoDepthEnabled').checked;
    if (!imageFile) {
      this.setStatus('Choose at least an image to begin.');
      return;
    }

    this.setStatus('Loading image…');
    const imgPix = await readPixelsToUint8(imageFile);
    const W = imgPix.width, H = imgPix.height;

    // Depth map and segmentation map are optional -- both are needed
    // together (wallpaper.js's geometry stage takes both or neither); if
    // neither source below produces them, crop and focus are simply
    // disabled and the image is used as-is (see rebuildGeometry()/
    // rebuildColor() below). Three mutually exclusive sources, in
    // priority order: computed in-browser (autoDepth), uploaded files,
    // or nothing.
    let depthBytes = null, segFloats = null, hasDepth = false;

    if (autoDepth) {
      try {
        this.setStatus('Running depth + segmentation models in your browser…');
        const [depthIds, segIds] = await Promise.all([
          estimateDepth(imageFile, W, H, (msg) => this.setStatus(msg)),
          estimateSegmentation(imageFile, W, H, (msg) => this.setStatus(msg)),
        ]);
        depthBytes = depthIds;
        segFloats = Float32Array.from(segIds);
        hasDepth = true;
      } catch (e) {
        console.error(e);
        this.setStatus(`In-browser depth/segmentation failed (${e.message}) -- `
          + 'continuing without crop/focus.');
      }
    } else if (depthFile && segFile) {
      const [depthPix, segPix] = await Promise.all([depthFile, segFile].map(readPixelsToUint8));
      if (W !== depthPix.width || H !== depthPix.height ||
          W !== segPix.width || H !== segPix.height) {
        this.setStatus(`Error: image (${W}x${H}), depth `
          + `(${depthPix.width}x${depthPix.height}) and segmentation `
          + `(${segPix.width}x${segPix.height}) must all be the same pixel size.`);
        return;
      }
      depthBytes = new Uint8Array(W * H);
      for (let i = 0; i < W * H; i++) depthBytes[i] = depthPix.data[i * 4];   // R channel; /255 normalisation matches the depth contract

      // Segmentation needs the raw integer ID preserved (0 = background,
      // 1..N = candidate region) -- NOT normalised the way depth's [0,1]
      // convention wants, so this goes into a float texture directly
      // rather than createTextureU8's automatic /255.
      segFloats = new Float32Array(W * H);
      for (let i = 0; i < W * H; i++) segFloats[i] = segPix.data[i * 4];
      hasDepth = true;
    }

    const gl = this.gl;
    const imageTex = createTextureU8(gl, W, H, 4, new Uint8Array(imgPix.data.buffer));
    let depthTex = null, segTex = null;
    if (hasDepth) {
      depthTex = createTextureU8(gl, W, H, 1, depthBytes);
      segTex = createTextureF(gl, W, H, 1, segFloats);
    }

    // A re-upload replaces this.sources entirely -- free the previous
    // generation's textures (owned exclusively by this class, never by
    // the pipelines) and any still-kept filter-chain output from the
    // previous rebuildColor() (see that method's own comments for why
    // that one has to survive until here rather than being freed
    // immediately). Also mark the pipeline not-yet-set-up and drop the
    // stale colour source: rebuildGeometry() below yields a frame before
    // its own WebGL work runs, and a focal-depth/click event landing in
    // that window would otherwise try to render() against textures just
    // deleted above.
    if (this.sources) {
      gl.deleteTexture(this.sources.imageTex.tex);
      if (this.sources.depthTex) gl.deleteTexture(this.sources.depthTex.tex);
      if (this.sources.segTex) gl.deleteTexture(this.sources.segTex.tex);
    }
    if (this._keptFilterObj) {
      deleteTexAndFbo(gl, this._keptFilterObj);
      this._keptFilterObj = null;
    }
    this.wallpaper._setupDone = false;
    this._colorTex = null;

    this.sources = { imageTex, depthTex, segTex, imageSize: [W, H], hasDepth };
    this._setGeometryControlsEnabled(hasDepth);
    this.setStatus(`Loaded ${W}x${H}` + (hasDepth ? '.' : ' (no depth/segmentation -- crop & focus disabled).'));
    await this.rebuildGeometry();
  }

  // Depth/segmentation-driven controls (crop target + focus) are inert
  // without both uploads -- greyed out and disabled rather than left
  // active but silently ignored, so the UI doesn't lie about what the
  // current upload can do.
  _setGeometryControlsEnabled(enabled) {
    for (const id of ['aspectRatio', 'targetW', 'kLayers']) {
      document.getElementById(id).disabled = !enabled;
    }
    document.getElementById('focusFieldset').disabled = !enabled;
  }

  // Parses the "W:H" text input (same convention as depth_blur.py's own
  // --aspect flag) into [ratioW, ratioH]. Falls back to 16:9 on anything
  // unparseable rather than failing silently on garbage input.
  _aspectRatio() {
    const m = document.getElementById('aspectRatio').value.trim()
      .match(/^(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)$/);
    if (!m) return [16, 9];
    const [rw, rh] = [parseFloat(m[1]), parseFloat(m[2])];
    return (rw > 0 && rh > 0) ? [rw, rh] : [16, 9];
  }

  // Output pixel size: width is the one thing the user sets directly;
  // height follows from it and the aspect ratio, so the two never fall
  // out of sync with each other.
  _targetSize() {
    const w = Math.max(1, parseInt(document.getElementById('targetW').value, 10) || 960);
    const [ratioW, ratioH] = this._aspectRatio();
    const h = Math.max(1, Math.round(w * ratioH / ratioW));
    return [w, h];
  }

  _sigmaMax() { return parseFloat(document.getElementById('sigmaMax').value) || 10.0; }

  _focalDepth() {
    return document.getElementById('focalAuto').checked
      ? null
      : parseFloat(document.getElementById('focalDepth').value);
  }

  // GEOMETRY stage: crop offset, figure identification/median depth, and
  // GMM depth layers, decided once from the ORIGINAL uploaded image
  // (never a filtered one -- see this file's header comment). Runs on
  // upload and on a target-size/K change; never on a filter toggle or a
  // focal-depth tick. Skipped entirely without depth+segmentation --
  // there's nothing for wallpaper.js to compute without a depth map, and
  // rebuildColor() below falls back to blitting the filtered image as-is.
  async rebuildGeometry() {
    if (!this.sources) return;
    if (!this.sources.hasDepth) {
      this.wallpaper._setupDone = false;
      await this.rebuildColor();
      return;
    }
    this.setStatus('Computing focus & crop…');
    // Yield one frame so the status message above actually paints before
    // the (synchronous) WebGL work below blocks the main thread.
    await new Promise((r) => requestAnimationFrame(r));

    const t0 = performance.now();
    const [ratioW, ratioH] = this._aspectRatio();
    const k = parseInt(document.getElementById('kLayers').value, 10) || 8;
    this.wallpaper.setup(
      this.sources.imageTex, this.sources.depthTex, this.sources.segTex,
      this.sources.imageSize, this._targetSize(), ratioW, ratioH, { kLayers: k });
    this._geomMs = performance.now() - t0;

    await this.rebuildColor();
  }

  // COLOUR stage: night (continuous amount) then palette mixing, applied to the
  // *original* source image, producing the texture render() will sample.
  // Runs on upload (via rebuildGeometry() above) and whenever a filter
  // toggle, mode, or the night-amount slider changes -- cheap (single-digit
  // ms at demo resolutions -- see shaders/night/README.md and
  // shaders/README.md's timing tables), since it's a small multi-pass
  // "compute once" pipeline with no dependency on focal depth or geometry.
  async rebuildColor() {
    if (!this.sources) return;
    this.setStatus('Applying filters…');
    await new Promise((r) => requestAnimationFrame(r));

    const gl = this.gl;
    const t0 = performance.now();
    const size = this.sources.imageSize;

    // Each enabled filter's output is this call's own transient object,
    // *except* the last one in the chain (if any ran) -- that one becomes
    // `filtered` below, stored as `this._colorTex` and sampled by every
    // render() call until the *next* rebuildColor() replaces it. It can't
    // be freed until that happens, so it's tracked separately as `keep`
    // rather than in `chainObjs` (safe to free the moment this function
    // is done with it, since nothing downstream reads through it twice).
    const chainObjs = [];
    let filtered = this.sources.imageTex;
    if (document.getElementById('nightEnabled').checked) {
      const amount = parseFloat(document.getElementById('nightAmount').value);
      const out = this.night.run(filtered, size, amount);
      chainObjs.push(out);
      filtered = out.tex;
    }
    if (document.getElementById('mixEnabled').checked) {
      const mode = document.getElementById('mixMode').value;
      const pipe = mode === 'additive' ? this.additive : this.subtractive;
      const out = pipe.run(filtered, size);
      chainObjs.push(out);
      filtered = out.tex;
    }
    const keep = chainObjs.length > 0 ? chainObjs[chainObjs.length - 1] : null;
    for (const obj of chainObjs) if (obj !== keep) deleteTexAndFbo(gl, obj);
    const filterMs = performance.now() - t0;

    // `filtered` becomes this._colorTex below, sampled on every render()
    // call from here on -- free the PREVIOUS generation's kept object
    // only now that this one has taken over, never the moment it was
    // computed.
    if (this._keptFilterObj) deleteTexAndFbo(gl, this._keptFilterObj);
    this._keptFilterObj = keep;
    this._colorTex = filtered;

    this.setStatus(this.sources.hasDepth
      ? `Focus/crop: ${this._geomMs.toFixed(0)}ms. Filters: ${filterMs.toFixed(0)}ms.`
      : `Filters: ${filterMs.toFixed(0)}ms. (No depth/segmentation -- crop & focus disabled.)`);
    this._redrawComposite();
  }

  // Draws whatever rebuildColor()/onNightAmountInput() just produced:
  // through wallpaper.render() (crop + depth blur) if geometry was
  // computed, or the filtered image blitted straight to the canvas
  // otherwise (see rebuildGeometry()'s no-depth branch).
  _redrawComposite() {
    if (this.sources.hasDepth) {
      this.onFocalDepthInput();
    } else {
      const [w, h] = this.sources.imageSize;
      this.blit({ tex: this._colorTex, fbo: { w, h } });
    }
  }

  // Fast path for dragging the night-amount slider: reuses night.js's
  // already-prepared detection state (from the last rebuildColor()) and
  // only reruns its cheap resolve() pass, then palette mixing (if enabled --
  // real GPU cost measured at low single-digit ms for both modes, so
  // rerunning it on every tick is fine too), then the composite. Skips
  // rebuildColor()'s own night.prepare() call entirely -- that's the
  // expensive part (the multi-scale DoG detection stack), and it doesn't
  // depend on `amount` at all, so there's nothing to gain from rerunning
  // it here. Falls back to doing nothing if nothing's prepared yet (no
  // image loaded, or night hasn't been toggled on).
  onNightAmountInput() {
    if (!document.getElementById('nightEnabled').checked || !this.night._prepared) return;
    const gl = this.gl;
    const amount = parseFloat(document.getElementById('nightAmount').value);
    let filtered = this.night.resolve(amount).tex;   // owned/freed internally by night.js

    let newKept = null;
    if (document.getElementById('mixEnabled').checked) {
      const mode = document.getElementById('mixMode').value;
      const pipe = mode === 'additive' ? this.additive : this.subtractive;
      const out = pipe.run(filtered, this.sources.imageSize);
      newKept = out;
      filtered = out.tex;
    }
    // Same ownership rule as rebuildColor(): _keptFilterObj only ever
    // holds a palette-mixing output (never night's own resolve output, which
    // night.js already manages), freed one generation late.
    if (this._keptFilterObj) deleteTexAndFbo(gl, this._keptFilterObj);
    this._keptFilterObj = newKept;
    this._colorTex = filtered;

    this._redrawComposite();
  }

  onFocalDepthInput() {
    if (!this.wallpaper._setupDone || !this._colorTex) return;
    // Live on every tick: geometry was already decided in
    // rebuildGeometry() and the colour source in rebuildColor(), so this
    // composite pass is the only thing that needs to rerun -- consistently
    // sub-millisecond to a few ms at demo resolutions.
    const out = this.wallpaper.render(this._colorTex, this._focalDepth(), this._sigmaMax());
    this.blit(out);
  }

  blit(resultObj) {
    const gl = this.gl;
    const w = resultObj.fbo.w, h = resultObj.fbo.h;
    this.canvas.width = w;
    this.canvas.height = h;
    runFullscreen(gl, this.vao, this.blitProg, { fbo: null, w, h }, { u_tex: resultObj.tex });
  }

  onCanvasClick(e) {
    if (!this.wallpaper._setupDone) return;
    const rect = this.canvas.getBoundingClientRect();
    const cx = ((e.clientX - rect.left) / rect.width) * this.canvas.width;
    const cy = ((e.clientY - rect.top) / rect.height) * this.canvas.height;

    const gl = this.gl;
    const wp = this.wallpaper;
    const [targetW, targetH] = wp.targetSize;
    const [imgW, imgH] = wp.imageSize;
    const mm = readFramebufferRGBA(gl, wp.cropOffset.fbo);
    const offset = Math.round(mm[0]);

    let srcX, srcY;
    if (wp.cropAxisIsX) {
      srcX = offset + ((cx + 0.5) / targetW) * wp.newLength - 0.5;
      srcY = ((cy + 0.5) / targetH) * imgH - 0.5;
    } else {
      srcY = offset + ((cy + 0.5) / targetH) * wp.newLength - 0.5;
      srcX = ((cx + 0.5) / targetW) * imgW - 0.5;
    }
    const px = Math.max(0, Math.min(imgW - 1, Math.round(srcX)));
    const py = Math.max(0, Math.min(imgH - 1, Math.round(srcY)));
    const depth = wp.sampleDepthAt(px, py);

    document.getElementById('focalAuto').checked = false;
    document.getElementById('focalDepth').value = depth.toFixed(3);
    this.onFocalDepthInput();
  }

  // Saves the canvas's current pixels as a PNG. Relies on the context
  // having been created with preserveDrawingBuffer: true (see
  // constructor) -- without that, toBlob() here could read back blank
  // content, since it runs whenever the user happens to click the
  // button, not right after a draw.
  onDownload() {
    if (!this.sources || !this._colorTex) {
      this.setStatus('Nothing to download yet -- load an image first.');
      return;
    }
    this.canvas.toBlob((blob) => {
      if (!blob) {
        this.setStatus('Download failed: could not read the canvas.');
        return;
      }
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'wallricer-wallpaper.png';
      a.click();
      URL.revokeObjectURL(url);
    }, 'image/png');
  }

  // Zips up exactly the GLSL shaders the demo is actually using right
  // now -- read fresh off the current UI/state at click time, not
  // whatever was active when an image was last loaded, so toggling a
  // filter after loading still changes what a subsequent export
  // contains. See shader_export.js for what "relevant" means precisely.
  async onDownloadShaders() {
    const state = {
      paletteName: this._currentPalette,
      mixEnabled: document.getElementById('mixEnabled').checked,
      mixMode: document.getElementById('mixMode').value,
      nightEnabled: document.getElementById('nightEnabled').checked,
      wallpaperActive: !!(this.sources && this.sources.hasDepth),
    };
    await downloadShaderZip(state, (msg) => this.setStatus(msg));
  }
}

window.addEventListener('DOMContentLoaded', async () => {
  const app = new App();
  await app.warmup();
});
