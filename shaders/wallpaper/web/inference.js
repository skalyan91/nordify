// Optional client-side depth estimation + figure segmentation via
// transformers.js (ONNX models run in-browser through WASM, or WebGPU if
// the browser supports it) -- an alternative to uploading pre-made depth
// map / segmentation map files. See index.html's own disclosure text:
// unlike everything else in this demo, this downloads real neural-network
// weights (tens of MB) the first time it's used, not just a few KB of JS.
//
// Depth: Depth Anything V2 Small -- the same model depth_blur.py uses by
// default -- via transformers.js's `depth-estimation` pipeline, run
// through the *same* tiled multi-pass refinement depth_blur.py's own
// `_estimate_depth` uses (see the comment above estimateDepth() below),
// not a single whole-image pass -- Depth Anything resizes any input down
// to a small fixed size (518px) internally regardless of source
// resolution, so a single pass alone loses thin structures (wires,
// masts, lattice towers) before the network ever sees them.
//
// Segmentation: DETR ResNet-50 panoptic segmentation (`image-segmentation`
// pipeline), NOT a port of depth_blur.py's own SAM-based automatic mask
// generation + depth-guided region merging -- that process (~1000
// candidate masks, NMS, a union-find merge against the depth map) is both
// too slow for interactive in-browser use and depends on Python-side
// logic with no browser equivalent model. Panoptic segmentation is a
// reasonable, honestly-simpler substitute for what the segmentation map
// actually gets USED for downstream: wallpaper.js's u_segmentation
// contract only needs a per-pixel integer region ID (0 = background,
// 1..N = one figure candidate each) so its own GPU-side scoring pass
// (segment_score / argmax_1d) can rank whichever region is most
// prominent -- it doesn't care how the regions were produced, only that
// each is spatially contiguous.

const DEPTH_MODEL = 'onnx-community/depth-anything-v2-small';
const SEGMENTATION_MODEL = 'Xenova/detr-resnet-50-panoptic';
const TRANSFORMERS_CDN_URL = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.0.1/+esm';

// wallpaper.js reserves ID 0 for background/unassigned and sizes its
// per-segment accumulator at 64 (MAX_SEGMENTS there) -- cap here matches.
const MAX_SEGMENTS = 63;

// Same defaults as depth_blur.py's _DEFAULT_TILE_WIDTH_FRACS / the
// tile_overlap default on _estimate_depth -- see that function's own
// docstring for why these particular fractions/overlap.
const TILE_WIDTH_FRACS = [0.5, 0.25, 0.125];
const TILE_OVERLAP = 0.5;
// Depth Anything V2 Small's own preprocessor_config.json: size.height =
// size.width = 518 (confirmed on both the PyTorch and ONNX repos). Used
// only as a fallback if introspecting the loaded pipeline's own
// processor config (below) doesn't turn up a value -- mirrors
// depth_blur.py's own `getattr(..., 518)` fallback.
const NATIVE_TILE_FALLBACK = 518;

let _transformersPromise = null;
function _getTransformers() {
  if (!_transformersPromise) {
    _transformersPromise = import(/* webpackIgnore: true */ TRANSFORMERS_CDN_URL);
  }
  return _transformersPromise;
}

function _reportProgress(p, onStatus, label) {
  if (!onStatus || !p) return;
  if (p.status === 'progress' && p.total) {
    const pct = Math.round((p.loaded / p.total) * 100);
    const mb = (p.total / 1e6).toFixed(0);
    onStatus(`Downloading ${label}… ${pct}% of ${mb}MB (${p.file})`);
  } else if (p.status === 'ready') {
    onStatus(`${label} ready, running…`);
  }
}

let _depthPipelinePromise = null;
function _loadDepthEstimator(onStatus) {
  if (!_depthPipelinePromise) {
    _depthPipelinePromise = (async () => {
      const { pipeline } = await _getTransformers();
      return pipeline('depth-estimation', DEPTH_MODEL, {
        dtype: 'q8',
        progress_callback: (p) => _reportProgress(p, onStatus, 'depth model'),
      });
    })();
  }
  return _depthPipelinePromise;
}

let _segmentationPipelinePromise = null;
function _loadSegmenter(onStatus) {
  if (!_segmentationPipelinePromise) {
    _segmentationPipelinePromise = (async () => {
      const { pipeline } = await _getTransformers();
      return pipeline('image-segmentation', SEGMENTATION_MODEL, {
        dtype: 'q8',
        progress_callback: (p) => _reportProgress(p, onStatus, 'segmentation model'),
      });
    })();
  }
  return _segmentationPipelinePromise;
}

// Nearest-available resample of a single-channel (or first-channel-of-N)
// raster to (targetW, targetH), via a throwaway canvas -- both the depth
// model's own output resolution and each segmentation mask's resolution
// can differ from the source image's, and everything downstream (GPU
// texture upload) needs them at the source image's exact size. Returns a
// flat Uint8Array of length targetW*targetH.
function _resampleGrayscale(rawImage, targetW, targetH) {
  const { data, width, height, channels } = rawImage;
  const srcCanvas = document.createElement('canvas');
  srcCanvas.width = width;
  srcCanvas.height = height;
  const srcCtx = srcCanvas.getContext('2d');
  const imgData = srcCtx.createImageData(width, height);
  for (let i = 0; i < width * height; i++) {
    const v = data[i * channels];
    imgData.data[i * 4] = v;
    imgData.data[i * 4 + 1] = v;
    imgData.data[i * 4 + 2] = v;
    imgData.data[i * 4 + 3] = 255;
  }
  srcCtx.putImageData(imgData, 0, 0);

  const dstCanvas = document.createElement('canvas');
  dstCanvas.width = targetW;
  dstCanvas.height = targetH;
  const dstCtx = dstCanvas.getContext('2d');
  dstCtx.drawImage(srcCanvas, 0, 0, targetW, targetH);
  const outData = dstCtx.getImageData(0, 0, targetW, targetH).data;
  const out = new Uint8Array(targetW * targetH);
  for (let i = 0; i < out.length; i++) out[i] = outData[i * 4];
  return out;
}

// depth_blur.py's _tile_starts: 1-D tile start offsets covering
// [0, length) with the given tile size and stride, snapping the last
// tile flush with the far edge.
function _tileStarts(length, tile, stride) {
  if (length <= tile) return [0];
  const starts = [];
  for (let s = 0; s <= length - tile; s += stride) starts.push(s);
  if (starts[starts.length - 1] !== length - tile) starts.push(length - tile);
  return starts;
}

// depth_blur.py's _feather_weights: 1-D tent-feathered blend weight for
// a tile spanning [start, start+length) within an axis of size total --
// ramps 0->1 over the tile's leading `overlap` pixels and 1->0 over its
// trailing `overlap` pixels, but only on sides with a neighbouring tile
// to hand off to (a tile flush with the image border holds full weight
// right up to that border).
function _featherWeights(start, length, total, overlap) {
  const w = new Float32Array(length).fill(1);
  if (overlap <= 0) return w;
  for (let i = 0; i < length; i++) {
    const left = start > 0 ? i : overlap;
    const right = start + length < total ? length - 1 - i : overlap;
    w[i] = Math.min(1, Math.max(0, Math.min(left, right) / overlap));
  }
  return w;
}

// Bilinear resample of a flat row-major Float32 raster -- used for both
// upsampling a tile's raw model output back to its own footprint size
// and for resizing the global pass to the full image size. A plain JS
// implementation rather than a canvas round-trip since the values here
// are arbitrary-scale raw disparity, not [0,255] pixels.
function _resizeFloatBilinear(src, srcW, srcH, dstW, dstH) {
  if (srcW === dstW && srcH === dstH) {
    return src instanceof Float32Array ? src : Float32Array.from(src);
  }
  const out = new Float32Array(dstW * dstH);
  const xRatio = srcW / dstW, yRatio = srcH / dstH;
  for (let dy = 0; dy < dstH; dy++) {
    const sy = Math.min(srcH - 1, Math.max(0, (dy + 0.5) * yRatio - 0.5));
    const sy0 = Math.floor(sy), sy1 = Math.min(srcH - 1, sy0 + 1), fy = sy - sy0;
    for (let dx = 0; dx < dstW; dx++) {
      const sx = Math.min(srcW - 1, Math.max(0, (dx + 0.5) * xRatio - 0.5));
      const sx0 = Math.floor(sx), sx1 = Math.min(srcW - 1, sx0 + 1), fx = sx - sx0;
      const v00 = src[sy0 * srcW + sx0], v01 = src[sy0 * srcW + sx1];
      const v10 = src[sy1 * srcW + sx0], v11 = src[sy1 * srcW + sx1];
      const top = v00 + (v01 - v00) * fx;
      const bot = v10 + (v11 - v10) * fx;
      out[dy * dstW + dx] = top + (bot - top) * fy;
    }
  }
  return out;
}

// Decodes an image source once into a full-resolution canvas -- every
// tile crop and the global pass are drawn from this, matching
// depth_blur.py's own single `rgb` array. Decoded to exactly
// (width, height) (the caller's already-known source pixel size), so
// tile crop coordinates line up with the final output 1:1.
async function _toCanvas(source, width, height) {
  const bitmap = await createImageBitmap(source);
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  canvas.getContext('2d').drawImage(bitmap, 0, 0, width, height);
  bitmap.close?.();
  return canvas;
}

function _cropCanvas(srcCanvas, x0, y0, w, h) {
  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  c.getContext('2d').drawImage(srcCanvas, x0, y0, w, h, 0, 0, w, h);
  return c;
}

// Best-effort introspection of the loaded pipeline's own image processor
// config, mirroring depth_blur.py's `getattr(pipe.image_processor,
// 'size', {})`. Falls back to the confirmed 518 constant above if the
// property path isn't there (a defensive guard against a transformers.js
// internal-structure change, not an expected case).
function _nativeTileSize(estimator) {
  const size = estimator?.processor?.image_processor?.size;
  return size?.height ?? size?.width ?? size?.shortest_edge ?? NATIVE_TILE_FALLBACK;
}

// Runs the depth pipeline on one crop and returns its *raw* (unnormalised)
// predicted-depth values -- the model's output up to an unknown per-
// inference affine transform, exactly what _tileRefine's own
// least-squares fit needs. Deliberately NOT the pipeline's own `depth`
// RawImage output (already per-call min-max normalised to [0,255] by the
// library, which would throw away the precision the affine fit relies
// on -- normalising per crop is also exactly the arbitrary-scale problem
// the fit exists to correct, so feeding it already-normalised data would
// be circular).
async function _inferRaw(estimator, canvas) {
  const { predicted_depth } = await estimator(canvas);
  const dims = predicted_depth.dims;
  const h = dims[dims.length - 2], w = dims[dims.length - 1];
  const data = predicted_depth.data instanceof Float32Array
    ? predicted_depth.data : Float32Array.from(predicted_depth.data);
  return { data, w, h };
}

// depth_blur.py's _tile_refine: one pass of tiled depth refinement
// against `reference`. Crops overlapping tiles sized to `footprint`
// (the model downsamples internally to its own native resolution
// regardless), each least-squares aligned (scale + shift) to
// `reference`'s own values over the same footprint -- since the model's
// depth output is only defined up to an unknown per-inference affine
// transform -- then blended together with tent-feathered edges. Returns
// a Float32Array the same (W, H) shape as `reference` (flat, row-major).
async function _tileRefine(estimator, srcCanvas, reference, W, H, footprint, overlapFrac, onStatus, passIdx, passCount) {
  if (W <= footprint && H <= footprint) return reference;   // whole image already fits in one tile's footprint

  const overlap = Math.round(footprint * overlapFrac);
  const stride = footprint - overlap;
  const xs = _tileStarts(W, footprint, stride);
  const ys = _tileStarts(H, footprint, stride);
  const nTiles = xs.length * ys.length;

  const depthAcc = new Float32Array(W * H);
  const weightAcc = new Float32Array(W * H);

  let tileIdx = 0;
  for (const y0 of ys) {
    const th = Math.min(footprint, H - y0);
    const wy = _featherWeights(y0, th, H, overlap);
    for (const x0 of xs) {
      tileIdx++;
      onStatus?.(`Estimating depth (tiled pass ${passIdx + 1}/${passCount}, tile ${tileIdx}/${nTiles})…`);

      const tw = Math.min(footprint, W - x0);
      const crop = _cropCanvas(srcCanvas, x0, y0, tw, th);
      const raw = await _inferRaw(estimator, crop);
      const tileDepth = _resizeFloatBilinear(raw.data, raw.w, raw.h, tw, th);

      // Least-squares scale+shift fit of this tile to the reference
      // map's own values over the same footprint, subsampled 4x4 for
      // speed -- mirrors depth_blur.py's own [::4, ::4] subsample. Closed
      // -form OLS on (tileDepth, reference) pairs rather than a general
      // lstsq call, since there are only two parameters (scale, shift).
      let sx = 0, sy = 0, sxx = 0, sxy = 0, n = 0;
      for (let ty = 0; ty < th; ty += 4) {
        const refRow = (y0 + ty) * W;
        const tileRow = ty * tw;
        for (let tx = 0; tx < tw; tx += 4) {
          const s = tileDepth[tileRow + tx];
          const d = reference[refRow + x0 + tx];
          sx += s; sy += d; sxx += s * s; sxy += s * d; n++;
        }
      }
      const meanX = sx / n, meanY = sy / n;
      const varX = sxx / n - meanX * meanX;
      const covXY = sxy / n - meanX * meanY;
      const a = varX > 1e-9 ? covXY / varX : 0;
      const b = meanY - a * meanX;

      const wx = _featherWeights(x0, tw, W, overlap);
      for (let ty = 0; ty < th; ty++) {
        const weightY = wy[ty];
        const accRow = (y0 + ty) * W + x0;
        const tileRow = ty * tw;
        for (let tx = 0; tx < tw; tx++) {
          const weight = weightY * wx[tx];
          const aligned = a * tileDepth[tileRow + tx] + b;
          depthAcc[accRow + tx] += weight * aligned;
          weightAcc[accRow + tx] += weight;
        }
      }
    }
  }

  const out = new Float32Array(W * H);
  for (let i = 0; i < out.length; i++) out[i] = depthAcc[i] / Math.max(weightAcc[i], 1e-6);
  return out;
}

// Min-max normalises a raw (arbitrary-scale) disparity field to a flat
// [0,255] Uint8Array -- this demo's own depth-map contract (pixel/255 =
// normalised disparity), matching what depth_blur.py's --save-depth
// writes.
function _toDisparityBytes(raw, w, h) {
  let min = Infinity, max = -Infinity;
  for (let i = 0; i < raw.length; i++) {
    if (raw[i] < min) min = raw[i];
    if (raw[i] > max) max = raw[i];
  }
  const range = Math.max(max - min, 1e-6);
  const out = new Uint8Array(w * h);
  for (let i = 0; i < raw.length; i++) {
    out[i] = Math.round(Math.min(255, Math.max(0, ((raw[i] - min) / range) * 255)));
  }
  return out;
}

// imageSource: anything createImageBitmap() accepts -- a File/Blob works
// directly. Runs depth_blur.py's own multi-pass strategy: one global
// (whole-image) pass, then one independent tiled refinement pass per
// TILE_WIDTH_FRACS entry (each aligned directly against the global pass,
// not chained through each other), combined by per-pixel maximum --
// different tile footprints have different blind spots that mostly
// don't overlap, and max lets each pass contribute only where it's more
// confident something is close, without one pass's mistake erasing
// another's correct detail. See depth_blur.py's _estimate_depth
// docstring for the full rationale. Returns a flat Uint8Array(width*
// height), the depth contract's own pixel/255 = normalised-disparity
// convention.
export async function estimateDepth(imageSource, width, height, onStatus) {
  onStatus?.('Loading depth model…');
  const estimator = await _loadDepthEstimator(onStatus);
  const srcCanvas = await _toCanvas(imageSource, width, height);
  const nativeTile = _nativeTileSize(estimator);

  onStatus?.('Estimating depth (global pass)…');
  const global = await _inferRaw(estimator, srcCanvas);
  // globalDepth is the fixed reference every tiled pass aligns against;
  // combined is the separate, mutated-in-place running max -- passes are
  // deliberately NOT chained through each other's output (see this
  // function's own header comment), only through the untouched global
  // pass, so Float32Array.from() here makes an independent copy rather
  // than mutating globalDepth's own backing array in place.
  const globalDepth = _resizeFloatBilinear(global.data, global.w, global.h, width, height);
  const combined = Float32Array.from(globalDepth);

  for (let i = 0; i < TILE_WIDTH_FRACS.length; i++) {
    const footprint = Math.max(nativeTile, Math.round(width * TILE_WIDTH_FRACS[i]));
    const passResult = await _tileRefine(
      estimator, srcCanvas, globalDepth, width, height, footprint, TILE_OVERLAP,
      onStatus, i, TILE_WIDTH_FRACS.length);
    for (let p = 0; p < combined.length; p++) {
      if (passResult[p] > combined[p]) combined[p] = passResult[p];
    }
  }

  return _toDisparityBytes(combined, width, height);
}

// Returns a flat Uint8Array(width*height) of small integer region IDs
// (0 = background/unassigned, 1..N = one detected object each, ranked
// highest-confidence first and capped at MAX_SEGMENTS) -- directly
// upload-ready for wallpaper.js's u_segmentation contract.
export async function estimateSegmentation(imageSource, width, height, onStatus) {
  onStatus?.('Loading segmentation model…');
  const segmenter = await _loadSegmenter(onStatus);
  onStatus?.('Segmenting…');
  const segments = await segmenter(imageSource);

  const ranked = segments
    .filter((s) => s.mask)
    .sort((a, b) => (b.score ?? 0) - (a.score ?? 0))
    .slice(0, MAX_SEGMENTS);

  const ids = new Uint8Array(width * height);
  for (let i = 0; i < ranked.length; i++) {
    const id = i + 1;
    const maskPix = _resampleGrayscale(ranked[i].mask, width, height);
    for (let p = 0; p < maskPix.length; p++) {
      if (maskPix[p] > 127) ids[p] = id;
    }
  }
  return ids;
}
