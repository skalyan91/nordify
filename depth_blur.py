#!/usr/bin/env python3
"""Crop and apply depth-guided defocus blur for wallpaper preparation.

Implements the telecentric (infinite focal length) defocus model via layered
forward-scatter compositing.  The depth map is divided into K slabs; each slab
forward-scatters its content with a disc (pillbox) kernel of radius
r = sigma_max · |d − d_focus| / denom (the telecentric CoC formula), then slabs
are composited front-to-back with premultiplied alpha.  This correctly places
background bokeh circles at foreground edges without heuristic depth dilation.
All computation runs in linear light with GPU acceleration through MLX.

Requirements:
    pip install transformers torch pillow
    pip install mlx   # optional — used for GPU-accelerated blur pyramid
"""

import argparse
import sys

import cv2
import numpy as np

_DEFAULT_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
_DEFAULT_TILE_WIDTH_FRACS = [0.5, 0.25, 0.125]


def _srgb_to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c):
    c = np.maximum(c, 0.0)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1.0 / 2.4) - 0.055)


def _crop_to_aspect(image, ratio_w, ratio_h, align='center'):
    """Crop to the given aspect ratio.

    When width is cropped: align is 'left'/'center'/'right'.
    When height is cropped: align is 'top'/'center'/'bottom'.
    """
    h, w = image.shape[:2]
    if w * ratio_h > h * ratio_w:   # image wider than target: crop width
        new_w = h * ratio_w // ratio_h
        excess = w - new_w
        x0 = 0 if align == 'left' else excess if align == 'right' else excess // 2
        return image[:, x0:x0 + new_w]
    else:                            # image taller than target: crop height
        new_h = w * ratio_h // ratio_w
        excess = h - new_h
        y0 = 0 if align == 'top' else excess if align == 'bottom' else excess // 2
        return image[y0:y0 + new_h, :]


def _make_disc_kernel(radius):
    """Anti-aliased uniform disc (pillbox) kernel, normalised to sum 1.

    The edge gets a 1-pixel linear ramp (clip(radius − dist + 0.5, 0, 1))
    rather than a hard cutoff so that adjacent pyramid levels blend smoothly
    during trilinear interpolation.
    """
    if radius < 0.5:
        return np.ones((1, 1), dtype=np.float32)
    r_int = round(radius)
    y, x = np.ogrid[-r_int:r_int + 1, -r_int:r_int + 1]
    dist = np.sqrt(x.astype(np.float32) ** 2 + y.astype(np.float32) ** 2)
    kernel = np.clip(radius - dist + 0.5, 0.0, 1.0).astype(np.float32)
    kernel /= kernel.sum()
    return kernel


def _disc_blur_mlx(img, radius):
    """Disc blur for (H, W, C) array on GPU via MLX 2-D depthwise convolution."""
    import mlx.core as mx
    kernel = _make_disc_kernel(radius)
    if kernel.shape == (1, 1):
        return img
    C = img.shape[2]
    pad = kernel.shape[0] // 2
    padded = np.pad(img, [(pad, pad), (pad, pad), (0, 0)], mode='reflect')
    t = mx.array(padded[None])                                      # (1, H+2p, W+2p, C)
    w = mx.array(np.tile(kernel[None, :, :, None], (C, 1, 1, 1)))  # (C, kH, kW, 1)
    t = mx.conv2d(t, w, groups=C)
    mx.eval(t)
    return np.array(t[0])  # (H, W, C)


def _disc_blur_cpu(img, radius):
    """Disc blur for (H, W, C) array on CPU via cv2.filter2D."""
    kernel = _make_disc_kernel(radius)
    if kernel.shape == (1, 1):
        return img
    return cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REFLECT_101)


def _tile_starts(length, tile, stride):
    """1-D tile start offsets covering [0, length) with the given tile size
    and stride, snapping the last tile flush with the far edge."""
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def _feather_weights(start, length, total, overlap):
    """1-D tent-feathered blend weight for a tile spanning [start, start+length)
    within an axis of size total.

    Ramps 0→1 over the tile's leading `overlap` pixels and 1→0 over its
    trailing `overlap` pixels — but only on sides that have a neighbouring
    tile to hand off to; a tile flush with the image border holds full
    weight right up to that border, since nothing blends in from outside it.
    """
    w = np.ones(length, dtype=np.float32)
    if overlap <= 0:
        return w
    idx   = np.arange(length)
    left  = idx if start > 0 else np.full(length, overlap, dtype=np.int64)
    right = (length - 1 - idx) if start + length < total else np.full(length, overlap, dtype=np.int64)
    return np.clip(np.minimum(left, right) / overlap, 0.0, 1.0).astype(np.float32)


def _tile_refine(rgb, infer_fn, reference, native_tile, footprint, overlap_frac):
    """One pass of tiled depth refinement against `reference`: crops
    overlapping tiles sized to `footprint` (downsampled to `native_tile` for
    inference if larger), each least-squares aligned — scale + shift — to
    `reference`'s own values in its footprint, then blended together with
    tent-feathered edges. `reference` need not be a single global pass; a
    multi-pass cascade can chain calls, each refining the previous call's
    output at a narrower (higher-resolution, lower-context) footprint.
    Returns a depth map the same (H, W) shape as `reference`.
    """
    H, W = reference.shape
    if W <= footprint and H <= footprint:
        return reference   # whole image already fits in one tile's footprint

    overlap = round(footprint * overlap_frac)
    stride  = footprint - overlap
    xs = _tile_starts(W, footprint, stride)
    ys = _tile_starts(H, footprint, stride)

    print(f"  [depth] tiled refinement: {len(xs)}x{len(ys)} tiles, "
          f"footprint={footprint}px, native={native_tile}px, overlap={overlap_frac}...",
          file=sys.stderr, flush=True)

    depth_acc  = np.zeros((H, W), dtype=np.float32)
    weight_acc = np.zeros((H, W), dtype=np.float32)

    n_tiles = len(xs) * len(ys)
    for yi, y0 in enumerate(ys):
        wy = _feather_weights(y0, footprint, H, overlap)
        for xi, x0 in enumerate(xs):
            crop = rgb[y0:y0 + footprint, x0:x0 + footprint]
            # Downsample the (possibly larger-than-native) footprint crop to
            # the model's own input resolution for inference, then upsample
            # the result back to the crop's own footprint size.
            crop_native = (cv2.resize(crop, (native_tile, native_tile), interpolation=cv2.INTER_AREA)
                          if footprint != native_tile else crop)
            tile_depth = cv2.resize(infer_fn(crop_native), (footprint, footprint),
                                    interpolation=cv2.INTER_LINEAR)

            # Least-squares affine fit to the reference map's values in this
            # tile's footprint (subsampled for speed) — aligns this tile's
            # arbitrary scale/shift to the reference's scale without erasing
            # the fine detail the tile alone can see.
            target = reference[y0:y0 + footprint, x0:x0 + footprint]
            src = tile_depth[::4, ::4].ravel().astype(np.float64)
            dst = target[::4, ::4].ravel().astype(np.float64)
            (a, b), *_ = np.linalg.lstsq(
                np.stack([src, np.ones_like(src)], axis=1), dst, rcond=None)
            aligned = (a * tile_depth + b).astype(np.float32)

            wx     = _feather_weights(x0, footprint, W, overlap)
            weight = wy[:, None] * wx[None, :]

            depth_acc[y0:y0 + footprint, x0:x0 + footprint]  += weight * aligned
            weight_acc[y0:y0 + footprint, x0:x0 + footprint] += weight

            print(f"\r  [depth]   tile {yi * len(xs) + xi + 1}/{n_tiles}",
                  end="", file=sys.stderr, flush=True)
    print(file=sys.stderr, flush=True)

    return depth_acc / np.maximum(weight_acc, 1e-6)


def _estimate_depth(image_bgr, model=_DEFAULT_MODEL,
                    tile_width_frac=_DEFAULT_TILE_WIDTH_FRACS, tile_overlap=0.5):
    """High-resolution monocular depth.

    Depth Anything V2 (the default) is a fixed-input-size model — its image
    processor resizes everything down to a small fixed size (e.g. 518px)
    regardless of source resolution — so a single whole-image ("global")
    inference alone destroys thin foreground structures (wires, masts,
    lattice towers) before the network ever sees them, and silently inherits
    whatever's behind them in the depth map. `_tile_refine` recovers that
    detail: it re-runs the same model on overlapping tiles cropped at
    (approximately) its own native input resolution, so each tile needs
    little or no further downsampling and thin structures survive as real
    edges. Because the model's depth output is only defined up to an
    unknown per-inference affine transform (it's trained with a
    scale/shift-invariant loss), each tile's result isn't on the same scale
    as its neighbours or the global map — tiles are least-squares fit
    (scale + shift) to a reference map before blending them together with
    tent-feathered edges. See `tile_width_frac` below for how multiple such
    passes at different footprints are combined.

    Apple's Depth Pro (`--model apple/DepthPro-hf`) doesn't have this
    limitation: it derives its patches from multiple downsamplings of the
    *original* image and fuses them internally, so a single whole-image
    inference already sees thin foreground structures at native resolution
    — confirmed empirically: it recovers a several-px-wide transmission
    tower from a single pass on a 5000px-wide source with no tiling at all
    (`skip_tiling` below detects it specifically and returns its global pass
    directly, since tiling measurably doesn't help it). It was the default
    here until it was found to occasionally misjudge large flat regions on
    stylised/painted art in a way Depth Anything V2 doesn't (see
    `--fix-sky`) — and with the multi-pass tiling below, Depth Anything now
    matches or exceeds it on thin-structure detail too, so it's no longer
    needed as the default for that either.

    `tile_width_frac` and `tile_overlap` are exposed for experimenting with
    a context-vs-resolution tradeoff: a tile cropped at the model's own
    native resolution ("100% zoom", `tile_width_frac=None`) sees maximum
    per-tile resolution but minimum surrounding context, which may be
    exactly what starves the model of the cues it needs to recognise "one
    rigid object" rather than reading a noisy mix of nearby scales. Setting
    `tile_width_frac` to e.g. `0.5` sizes each tile's *source crop* to that
    fraction of the image's width (so `0.5` = each tile's footprint spans
    half the original image) — larger than the model's native input size —
    and downsamples just that crop down to native resolution before
    inference, giving the model a wider field of view per tile at reduced
    (but still better-than-a-single-global-pass) resolution. This is a
    different, larger-footprint tile than the "100% zoom" case, not a
    resize of the whole image before a fixed-size tiling pass.

    `tile_width_frac` also accepts a list: each fraction gets its own
    independent tiled pass — every one aligned directly against the global
    map, not chained through each other — and the final result is the
    per-pixel *maximum* disparity (nearest reading) across the global map and
    every pass. Different footprints have different blind spots that don't
    overlap much, so this recovers each one's strengths rather than forcing
    a single tradeoff: a wide pass (e.g. `0.5`) gives smooth, internally
    consistent depth on solid objects (no more per-pixel mast noise) but
    blends soft, semi-transparent things (steam, smoke) into the sky behind
    them, and loses thin wires outright past roughly a 2.5x per-tile
    downsample; a narrow pass (e.g. `0.125`, close to native resolution)
    keeps fine detail — even a lattice tower's individual crossing struts —
    but is noisy on large solid objects the way a lone native-zoom tiling
    pass always was. Since max only ever pulls a value *nearer*, never
    farther, each pass can only contribute where it's more confident
    something is close, not accidentally erase another pass's correct
    detail — confirmed empirically across `[0.5, 0.25, 0.125]`: smooth masts
    (from the 0.5 pass), a fully-resolved lattice tower (from the 0.125
    pass), and steam clearly separated from sky (recovered by whichever pass
    read it nearest), all in one map, each in a few seconds.

    Returns (H, W) float32, disparity convention (higher = closer). Depth
    Pro outputs metric depth in metres (higher = *farther*); this is
    inverted to disparity so the rest of the pipeline (CoC formula,
    --focus, rowmax) can treat every model's output the same way.

    Requires: pip install transformers torch pillow torchvision
    """
    try:
        from transformers import pipeline as hf_pipeline
        import torch
        from PIL import Image as PILImage
    except ImportError as e:
        print(
            f"Error: {e}\n"
            "Depth estimation requires: pip install transformers torch pillow torchvision",
            file=sys.stderr,
        )
        sys.exit(1)

    rgb  = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    H, W = image_bgr.shape[:2]

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"  [depth] model={model}  device={device}", file=sys.stderr, flush=True)
    pipe = hf_pipeline("depth-estimation", model=model, device=device)
    is_metric = "DepthPro" in type(pipe.model).__name__
    # Depth Pro derives its patches from multiple downsamplings of the
    # *original* image and fuses them internally, rather than resizing the
    # whole image down to its image processor's nominal size and stopping
    # there (as fixed-input-size models like Depth Anything do) — confirmed
    # empirically: a single pass recovers a several-px-wide transmission
    # tower from a 5000px-wide source. So a global pass alone is sufficient
    # regardless of how that source resolution compares to the processor's
    # advertised `size`, unlike every other model this function has been
    # run against.
    skip_tiling = is_metric

    def _infer(rgb_arr):
        out = pipe(PILImage.fromarray(rgb_arr))
        raw = np.array(out["predicted_depth"], dtype=np.float32)
        if raw.ndim > 2:
            raw = raw.squeeze()
        if is_metric:
            raw = 1.0 / (raw + 1e-3)   # metres (higher = farther) -> disparity
        return raw

    # Pass 1 — global depth field, whole image in one inference. Note: the
    # HF depth-estimation pipeline's own postprocessing resizes its output
    # to match the input's resolution for every model tried here (Depth
    # Pro and Depth Anything alike) — that shape match is NOT a signal that
    # native detail survived, since Depth Anything's global pass demonstrably
    # loses it (see module docstring); only `skip_tiling` above encodes that.
    print("  [depth] global pass...", file=sys.stderr, flush=True)
    global_depth = cv2.resize(_infer(rgb), (W, H), interpolation=cv2.INTER_LINEAR)

    if skip_tiling:
        return global_depth

    # Native tile size = the model's own native input resolution. Each pass's
    # tile *footprint* (the region of the original image each tile actually
    # crops) defaults to that same size — a "100% zoom" tile needing ~no
    # further downsampling — but tile_width_frac can size it as a fraction of
    # the image's width instead, trading per-tile resolution for context.
    proc_size   = getattr(pipe.image_processor, "size", None) or {}
    native_tile = int(proc_size.get("height") or proc_size.get("shortest_edge") or 518)

    fracs = ([tile_width_frac] if isinstance(tile_width_frac, (int, float))
            else list(tile_width_frac) if tile_width_frac else [None])

    # Each fraction gets its own independent pass aligned against the global
    # map — not chained through each other, so one pass's blind spot (e.g. a
    # wide pass blending steam into the sky) can't propagate into the next —
    # and the final map is the per-pixel maximum across the global map and
    # every pass. Max only ever pulls a value nearer, never farther, so a
    # pass can only contribute where it's more confident something is close;
    # it can't erase detail another pass correctly captured.
    combined = global_depth
    for frac in fracs:
        footprint = max(native_tile, round(W * frac)) if frac else native_tile
        pass_result = _tile_refine(rgb, _infer, global_depth, native_tile, footprint, tile_overlap)
        combined = np.maximum(combined, pass_result)
    return combined


def _depth_blur(image, depth_raw, sigma_max, d_focus=0.0, n_levels=16):
    """Telecentric defocus blur via forward-scatter layered compositing.

    Divides the depth map into K slabs.  Each slab k forward-scatters its
    content by blurring (image × mask_k) with disc radius r_k — spreading that
    layer's colour outward by exactly r_k pixels, exactly as a real aperture
    does.  Slabs are then composited front-to-back with premultiplied alpha so
    that foreground occludes background and background bokeh circles appear at
    foreground edges without heuristic depth dilation.

    CoC formula (telecentric): r_k = sigma_max · |d_k − d_focus| / denom
    where denom = max(d_focus, 1 − d_focus).

    depth_raw : (H, W) float32 — raw disparity; higher = closer to camera
    sigma_max : max disc radius in pixels (CoC at the farthest depth from focus)
    d_focus   : normalised disparity of the in-focus plane
    n_levels  : number of depth slabs (controls smoothness; default 16)
    """
    d_min, d_max = float(depth_raw.min()), float(depth_raw.max())
    depth = ((depth_raw - d_min) / (d_max - d_min)
             if d_max - d_min > 1e-6 else np.zeros_like(depth_raw))

    img_lin = _srgb_to_linear(image.astype(np.float32) / 255.0)
    H, W    = image.shape[:2]

    denom = max(d_focus, 1.0 - d_focus)  # ≥ 0.5

    try:
        import mlx.core  # noqa: F401
        _blur = _disc_blur_mlx
        print("  [blur] device=mlx", file=sys.stderr, flush=True)
    except ImportError:
        _blur = _disc_blur_cpu
        print("  [blur] device=cpu (install mlx for GPU acceleration)",
              file=sys.stderr, flush=True)

    K       = n_levels
    centers = np.linspace(0.0, 1.0, K)           # slab centres; partition of unity
    spacing = 1.0 / (K - 1) if K > 1 else 1.0
    radii   = sigma_max * np.abs(centers - d_focus) / denom

    # Accumulation buffers (premultiplied alpha)
    color_acc  = np.zeros((H, W, 3), dtype=np.float32)
    weight_acc = np.zeros((H, W),    dtype=np.float32)

    # Front-to-back order: foreground (highest disparity) first so it occludes
    order = np.argsort(centers)[::-1]

    print(f"  [blur] scatter-compositing {K} layers, r_max={sigma_max:.1f}px …",
          file=sys.stderr, flush=True)
    for i in order:
        r_k = float(radii[i])
        # Tent basis: linear interpolation between neighbouring slab centres,
        # giving exact partition of unity (sum_k mask_k = 1 everywhere).
        mask = np.maximum(0.0, 1.0 - np.abs(depth - centers[i]) / spacing,
                          dtype=np.float32)

        # Stack premultiplied colour and mask as 4 channels, blur in one pass.
        premult = np.empty((H, W, 4), dtype=np.float32)
        premult[:, :, :3] = img_lin * mask[:, :, None]
        premult[:, :,  3] = mask
        blurred = _blur(premult, r_k)

        color_k  = blurred[:, :, :3]  # disc_blur(image × mask) — scattered colour
        alpha_k  = blurred[:, :,  3]  # disc_blur(mask)          — scattered coverage

        # Premultiplied front-to-back composite: each layer fills what the
        # front layers left uncovered.
        remaining   = 1.0 - weight_acc
        color_acc  += remaining[:, :, None] * color_k
        weight_acc += remaining * alpha_k
        # Guard against float accumulation error pushing weight above 1
        np.clip(weight_acc, 0.0, 1.0, out=weight_acc)

    # Un-premultiply (weight_acc ≈ 1 everywhere after all K layers)
    result_lin = color_acc / (weight_acc[:, :, None] + 1e-6)
    return np.clip(_linear_to_srgb(result_lin) * 255.0, 0, 255).astype(np.uint8)


def _otsu_sky_mask(disp, feather=3.0):
    """Segment the farthest, most tightly-clustered region of a disparity
    map (typically sky) via Otsu's threshold, which picks the split that
    minimises intra-class variance — a good fit here since sky tends to sit
    in a narrow low-disparity band clearly apart from the rest of a scene's
    much wider spread. Returns a (H, W) float32 mask in [0, 1] (1 = sky)
    with a lightly feathered edge so a hard per-pixel threshold doesn't
    stairstep the blend seam.
    """
    d_min, d_max = float(disp.min()), float(disp.max())
    d8 = np.clip((disp - d_min) / max(d_max - d_min, 1e-6) * 255, 0, 255).astype(np.uint8)
    thresh, _ = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = (d8 <= thresh).astype(np.float32)
    return cv2.GaussianBlur(mask, (0, 0), feather)


def _fix_sky_depth(depth_disp, image_bgr,
                   sky_model="depth-anything/Depth-Anything-V2-Small-hf"):
    """Corrects a Depth Pro failure mode seen on stylised/painted art: it's
    trained on real photographs, where a flat, desaturated, silhouette-like
    region is a strong learned cue for atmospheric distance — so it can
    read a plainly-painted structure that way and place it *farther* than
    the sky behind it (confirmed on a power-station painting, where Depth
    Pro placed the sky nearer than the building in front of it, while Depth
    Anything V2 got the same region right).

    Depth Anything's map is used to both locate the sky — via
    `_otsu_sky_mask`, since sky is its single farthest, most tightly
    clustered region — and to supply corrected values there. The two
    models' outputs are on different, arbitrary scales, so Depth Anything's
    map can't be pasted in directly: it's least-squares fit (scale + shift)
    to depth_disp's scale using the *non-sky* region, where the two models
    should broadly agree, before its (now depth_disp-scaled) sky values are
    blended in through the feathered mask. A final safety clamp caps the
    sky region at the minimum non-sky disparity, guaranteeing the corrected
    sky is at least as far as everything in front of it even if the fit
    isn't perfect.
    """
    print("  [depth] sky correction: running Depth Anything V2 for sky mask...",
          file=sys.stderr, flush=True)
    sky_disp = _estimate_depth(image_bgr, model=sky_model)

    mask = _otsu_sky_mask(sky_disp)

    non_sky = mask < 0.5
    src = sky_disp[non_sky][::4].astype(np.float64)
    dst = depth_disp[non_sky][::4].astype(np.float64)
    (a, b), *_ = np.linalg.lstsq(
        np.stack([src, np.ones_like(src)], axis=1), dst, rcond=None)
    aligned_sky = (a * sky_disp + b).astype(np.float32)

    corrected = depth_disp * (1.0 - mask) + aligned_sky * mask

    non_sky_min = float(depth_disp[non_sky].min())
    return np.where(mask > 0.01, np.minimum(corrected, non_sky_min), corrected).astype(np.float32)


def _flatten_thin_masts(depth_disp, image_bgr, sam_model="facebook/sam-vit-base",
                        target_width=1500, min_aspect=2.0, min_height_frac=0.05,
                        max_relative_spread=0.15, feather=2.0):
    """Enforces near-uniform depth on thin, tall, solid vertical structures
    (chimneys, masts) whose true depth barely varies across their height but
    which monocular depth models can render with substantial internal noise
    (confirmed: a power-station smokestack whose per-row disparity wobbled
    non-monotonically by ±15 with no plausible perspective explanation).

    Candidates are found via SAM's automatic mask generation (the
    `mask-generation` pipeline) on the *source image*, not any depth map —
    using a depth map to find its own errors is circular, and an Otsu split
    on either Depth Pro's or Depth Anything's map here just lumps the
    (fairly dark) smokestacks in with sky rather than isolating them (both
    models place the smokestacks near the low, "far" end of their own
    disparity range). An earlier classical approach (black-hat/top-hat
    contrast + connected components) worked but was unreliable — it missed
    a mast embedded in busy, high-contrast painted clouds entirely, since it
    can only separate an object from a *locally uniform* backdrop.

    The image is downsampled to `target_width` before running SAM: its ViT
    encoder resizes to a fixed internal resolution regardless of input
    size, so feeding it the full-resolution source doesn't improve mask
    quality (confirmed empirically: full-resolution and this downsampled
    pass produce the same mask overlap against the true mast shape) — it
    only makes every downstream step of automatic mask generation (per-point
    decoding, mask upsampling, NMS across ~1000 candidate masks) far more
    expensive for no benefit (~17 minutes for one 4500px-wide image vs.
    ~17 seconds downsampled). Each mask's own bounding-box aspect ratio
    (height ≫ width) is enough to keep just the mast-like ones — no
    solidity check is needed here the way the classical approach required
    one, since SAM's masks already respect real object boundaries: the
    transmission tower's wire lattice isn't returned as any tall/narrow
    mask in the first place (confirmed empirically: zero overlap between
    the tower region and any aspect-qualifying mask).

    Each surviving mask's depth is *not* flattened to a single value outright
    — a mast large enough to show some genuine perspective drift across its
    height would lose that real variation. Instead, each pixel's deviation
    from the mask's own median is clamped to a tolerance band (a soft
    winsorize, not a hard replace): deviations already inside that band pass
    through untouched, and only the excess beyond it — the part with no
    plausible perspective explanation — gets pulled in.

    That clamp is one-sided: only deviations reading *farther* than the
    mast's own median are pulled in; deviations reading *nearer* are left
    untouched no matter how large. A solid mast can't have something
    genuinely farther "through" it, so an excess-far pixel is unambiguous
    noise — but a pixel reading nearer could easily be something real
    crossing in front of it (confirmed: a transmission wire passing over a
    smokestack was getting its correct, nearer depth erased and pulled back
    to the smokestack's own median before this was one-sided).

    That tolerance band shrinks quadratically, not linearly, toward the
    background. Disparity is (proportional to) inverse distance, so for a
    mast of fixed real-world height, the disparity range its own top-to-
    bottom depth difference could plausibly produce shrinks with the square
    of its distance, not linearly — a mast twice as far away should be
    allowed a quarter of the tolerance, not half. `allowed` is therefore
    `max_relative_spread * median_val * (median_val / scene_max)`: at
    `median_val == scene_max` (the nearest thing in the whole scene) it
    reduces to the old linear `max_relative_spread * median_val`, and it
    falls away quadratically for anything farther back. The result is
    blended in through a lightly feathered mask.
    """
    try:
        from transformers import pipeline as hf_pipeline
        import torch
        from PIL import Image as PILImage
    except ImportError as e:
        print(
            f"Error: {e}\n"
            "Mast flattening requires: pip install transformers torch pillow",
            file=sys.stderr,
        )
        sys.exit(1)

    h, w = depth_disp.shape
    device = "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"

    scale = min(1.0, target_width / w)
    small = cv2.resize(image_bgr, (max(1, int(w * scale)), max(1, int(h * scale))),
                       interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    print(f"  [depth] mast flattening: running SAM ({sam_model}) on "
          f"{small.shape[1]}x{small.shape[0]} downsample...", file=sys.stderr, flush=True)
    pipe = hf_pipeline("mask-generation", model=sam_model, device=device)
    out  = pipe(PILImage.fromarray(rgb), points_per_batch=64, points_per_side=32)

    corrected  = depth_disp.copy()
    min_height = min_height_frac * h
    scene_max  = max(float(depth_disp.max()), 1e-6)
    for m in out["masks"]:
        m = np.array(m).astype(np.uint8)
        ys, xs = np.where(m)
        if len(xs) == 0:
            continue
        bw, bh = xs.max() - xs.min(), ys.max() - ys.min()
        if bh < min_height or bh / max(bw, 1) < min_aspect:
            continue
        m_full = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        median_val = float(np.median(depth_disp[m_full]))
        allowed    = max_relative_spread * median_val * (median_val / scene_max)
        # One-sided: only pull in excess-far deviations (negative); a
        # nearer-reading pixel may be something real in front of the mast
        # (e.g. a wire), so that side is left uncapped.
        deviation  = np.maximum(depth_disp - median_val, -allowed)
        target     = median_val + deviation
        weight = cv2.GaussianBlur(m_full.astype(np.float32), (0, 0), feather)
        corrected = corrected * (1.0 - weight) + target * weight

    return corrected.astype(np.float32)


def _rank(values):
    """Average ("fractional") rank of each value, 0 = smallest.

    Tied values share the mean of the ranks they'd jointly occupy, rather
    than each grabbing a distinct ordinal rank via plain double-argsort.
    Plain ordinal ranking breaks ties by whatever order the values happen
    to appear in the input list — for a criterion like `connectedness`,
    where a large fraction of candidates genuinely tie at exactly 1.0 (a
    single coherent SAM mask), that tie-break is pure noise, not signal,
    and it can spread a tied block across nearly the whole rank range
    (confirmed: two candidates both at connectedness=1.0 landed at ranks 2
    and 12 out of 14 — a 10-rank, double-weighted swing from nothing but
    incidental list order). Averaging collapses every tied block to one
    shared rank, so equal values score equally.
    """
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values)
    ranks = np.empty(len(values))
    ranks[order] = np.arange(len(values))
    sorted_vals = values[order]
    i = 0
    n = len(values)
    while i < n:
        j = i
        while j + 1 < n and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    return ranks


class _UnionFind:
    """Minimal union-find (path compression, no union-by-rank — the region
    counts here are far too small for that to matter)."""

    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def _ranges_overlap_50(lo1, hi1, lo2, hi2):
    """True if [lo1,hi1] and [lo2,hi2] overlap by >50% of *both* ranges'
    own extents — symmetric, so a small range fully nested in the first
    quarter of a much larger one does not count (it covers >50% of itself
    but not of the larger range)."""
    inter = min(hi1, hi2) - max(lo1, lo2)
    if inter <= 0:
        return False
    return inter / max(hi1 - lo1, 1e-9) > 0.5 and inter / max(hi2 - lo2, 1e-9) > 0.5


def _chunkiness(mask, area, erosion_frac=0.1):
    """Fraction of `mask`'s area that survives erosion by a disk of radius
    `erosion_frac` × its own equivalent-circle radius `sqrt(area / pi)`.

    This replaces bounding-box aspect ratio ("squareness") as the
    tolerant-of-elongation compactness criterion: aspect ratio can't tell a
    solid, tall structure (a cooling tower, a chimney — substantial cross-
    section at every height) apart from a thin sliver of the same
    elongation (a wire, an edge-detection noise strip, one leg of a
    lattice tower) — both score near 0 on min(w,h)/max(w,h) even though
    only one of them is a real compact object. Erosion tests the thing
    that actually distinguishes them: local *thickness*. A thin sliver's
    width is small in absolute terms, so eroding by even a modest radius
    wipes it out almost entirely; a solid shape's cross-section survives a
    proportionally-sized erosion largely intact no matter how tall it is,
    because erosion only eats a fixed-radius margin off *every* boundary,
    and a long solid shape's boundary is almost all sides, not ends.

    The erosion radius scales with the *region's own* size (via its
    equivalent-circle radius) rather than a fixed pixel count, so this
    stays scale-invariant across differently-sized candidates and
    differently-sized images — the same fraction test applies whether the
    candidate is a 200px or 2000px structure.
    """
    r = max(1, round(erosion_frac * np.sqrt(area / np.pi)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    eroded = cv2.erode(mask.astype(np.uint8), kernel)
    return float(eroded.sum()) / area


def _detect_figure_focus(depth_raw, image_bgr, sam_model="facebook/sam-vit-base",
                         target_width=1500, min_area_frac=0.0005, max_area_frac=0.5):
    """Locate the principal "figure" — a compact foreground subject — via
    SAM segmentation of the source image, and return its median depth,
    normalised the same way `_depth_blur` normalises `depth_raw`, ready to
    use as `d_focus`.

    Candidates come from SAM's automatic mask generation (the
    `mask-generation` pipeline) on the *source image*, not from classical
    edge detection on the depth map — the same trade this module already
    made for `_flatten_thin_masts`, and for the same reason. An earlier
    version of this function ran Canny on the depth map and connected-
    component-labelled the non-edge pixels, exactly as `_flatten_thin_masts`
    originally did before it switched to SAM. It worked on photos with
    genuinely sharp depth discontinuities, but failed on a stylised power-
    station painting: Canny traced the cooling tower's silhouette as a
    clearly visible gradient, but the boundary had gaps too small to see by
    eye and too large for a fixed-size dilation to close at that image's
    resolution — enough for the "inside" of the tower to leak into the
    surrounding sky and merge into one connected blob covering ~90% of the
    frame (confirmed by rendering that raw connected component: sky,
    mountains, tower, and smokestacks all one undifferentiated region). The
    tower was never even considered as its own candidate — no amount of
    downstream ranking can recover from a segmentation stage that never
    isolated it. SAM has no such failure mode here: it segments from the
    image's own visual boundaries (brushwork, colour, contrast), which are
    complete and unambiguous on this image even where the depth map's
    inferred discontinuity is not — and this is the same image
    `_flatten_thin_masts`'s docstring already documents SAM succeeding on
    where a classical contrast-based approach missed a mast entirely
    against busy painted clouds.

    The image is downsampled to `target_width` before running SAM, exactly
    as `_flatten_thin_masts` does and for the same reason: SAM's ViT
    encoder resizes to a fixed internal resolution regardless of input
    size, so the full-resolution source doesn't improve mask quality, only
    the cost of mask upsampling and NMS across the ~1000 candidate masks
    (~17 minutes at full 4500px resolution vs. ~17 seconds downsampled).
    Each mask is resized back to `depth_raw`'s resolution with nearest-
    neighbour interpolation (preserving hard mask edges) before anything
    below measures it.

    A single object can still come back as several SAM masks — SAM's
    output is hierarchical (whole object and sub-parts both proposed) and
    a mask can be split by an occluder crossing in front of it — even
    though every piece sits at the same depth. Those pieces are merged back
    together before
    ranking: any two regions whose depth ranges (10th-90th percentile of
    their own pixels, robust to a few stray boundary pixels) overlap by
    more than 50% of *both* ranges (`_ranges_overlap_50`) are treated as
    one figure. This is a graph problem, not a single pairwise pass — three
    regions can chain together (A-B and B-C both overlap enough even if A-C
    doesn't) — so it's resolved with union-find over all pairs, and each
    resulting connected group's pixel masks are OR-ed into one candidate
    mask.

    Each merged candidate is scored on five criteria and combined by a
    **weighted rank sum** (each candidate's rank — 0 = worst — on one
    criterion, scaled by that criterion's weight, summed across all five)
    rather than a weighted or min-max-normalised score built from the raw
    quantities: the quantities live on incomparable scales (pixel counts
    vs. ratios vs. raw disparity), so weighting the *ranks* (already a
    common, comparable 0..N-1 scale) sidesteps having to calibrate a
    trade-off between raw units while still letting some criteria dominate
    the others. Ties are shared, not arbitrarily split — `_rank` averages
    the ranks of equal values rather than breaking ties by list order,
    which matters here because a criterion like connectedness genuinely
    ties at 1.0 for most candidates (a single coherent SAM mask), and an
    unbroken tie shouldn't swing a double-weighted criterion based on
    nothing but incidental ordering.

    Area carries weight 6 — enough to overcome losing on any two of the
    double-weighted shape criteria, not enough to overcome all three plus
    depth. Chunkiness, convexity, and connectedness carry weight 2; depth
    carries weight 1. Shape still matters more than merely being the
    nearest thing in frame (a near, ragged, scattered blob is exactly the
    false-positive pattern — a foreground clutter of unrelated same-depth
    objects — this ranking needs to reject), but not more than being
    substantially the largest candidate: confirmed necessary when a
    correctly-segmented cooling tower (257,310px) lost outright to an
    incidental dark-bush blob (23,824px, 10.8x smaller) that merely had
    marginally better chunkiness/solidity/depth, before area's weight was
    raised from 1.
        - size          — pixel count; larger wins. Weight 6.
        - chunkiness    — `_chunkiness`: fraction of the mask's area
                          surviving erosion by a radius proportional to its
                          own size; closer to 1 wins. A proxy for "one
                          compact, *solid* subject" that — unlike bounding-
                          box aspect ratio — doesn't penalise a shape merely
                          for being tall or elongated, only for being thin:
                          a cooling tower or chimney is exactly as
                          "chunky" as a round subject of the same cross-
                          sectional substance, while a wire, a lattice-
                          tower strut, or a thin sliver of edge-detection
                          noise erodes away almost entirely. Weight 2.
        - convexity     — mask area / convex-hull area (solidity); closer
                          to 1 wins. A real solid subject is mostly convex;
                          a merge that stitched together far-apart same-
                          depth pieces produces a hull that balloons out to
                          cover the gap between them, tanking this score.
                          Weight 2.
        - depth         — median raw disparity within the mask; higher
                          (nearer) wins. Weight 1.
        - connectedness — area of the merged mask's largest single spatial
                          blob / the merged mask's total area; closer to 1
                          wins. Depth-range merging is deliberately blind
                          to spatial position, so this is what actually
                          penalises a merge of two same-depth but
                          spatially disjoint objects (e.g. two unrelated
                          background elements at a similar distance) —
                          solidity penalises the *gap* between such pieces
                          but not fragmentation with no gap-spanning hull
                          cost (pieces already near each other). Weight 2.

    Candidates smaller than `min_area_frac` of the image (SAM proposes
    masks down to a few pixels — fine detail, not noise, but too small to
    be "the figure") are dropped before merging; merged candidates larger
    than `max_area_frac` (background spanning most of the frame) are
    dropped after. `min_area_frac` defaults small (0.05%) rather than to a
    size that would already read as "a subject" on its own: a real figure
    can be genuinely small in frame (a single piece of fruit among many, a
    distant animal), and rejecting it here would just leave the border/
    solidity/ranking criteria below with nothing legitimate to find —
    whereas an over-loose min_area_frac costs only a little wasted work on
    small candidates that the later criteria (especially chunkiness and
    solidity, weighted double) reject on shape anyway.

Any region touching the image border is discarded outright before
    merging, regardless of how well it otherwise scores: a region cut off
    by the frame is a partial view of whatever it belongs to (cropped by
    the photo, not by a real object boundary), so its size and shape can't
    be measured from it at all — the standard "exclude on edges" rule from
    classical particle/blob analysis. This has to happen *before* the
    depth-range merge, not after: a large, otherwise-clean interior
    candidate can end up unioned with a small, unrelated, border-touching
    mask on nothing more than a coincidental shared depth range (confirmed
    — a 5,669px sliver of sky merged into, and disqualified, an otherwise
    clean 348,847px cooling-tower group when this check ran post-merge).
    Filtering border-touching regions first sidesteps that entirely: a
    union of masks that each avoid the border can never touch the border
    either.

    Falls back to 0.0 (background/infinity focus, this module's existing
    default) if segmentation finds no surviving candidate at all.
    """
    try:
        from transformers import pipeline as hf_pipeline
        import torch
        from PIL import Image as PILImage
    except ImportError as e:
        print(
            f"Error: {e}\n"
            "Figure detection requires: pip install transformers torch pillow",
            file=sys.stderr,
        )
        sys.exit(1)

    h, w = depth_raw.shape
    d_min, d_max = float(depth_raw.min()), float(depth_raw.max())
    if d_max - d_min < 1e-6:
        return 0.0, None

    device = "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
    scale = min(1.0, target_width / w)
    small = cv2.resize(image_bgr, (max(1, int(w * scale)), max(1, int(h * scale))),
                       interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    print(f"  [focus] figure segmentation: running SAM ({sam_model}) on "
          f"{small.shape[1]}x{small.shape[0]} downsample...", file=sys.stderr, flush=True)
    pipe = hf_pipeline("mask-generation", model=sam_model, device=device)
    out = pipe(PILImage.fromarray(rgb), points_per_batch=64, points_per_side=32)

    total_px = h * w

    # First pass: raw SAM masks, filtered only for size.
    regions = []
    for m in out["masks"]:
        m = np.array(m).astype(np.uint8)
        if m.sum() == 0:
            continue
        mask = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        area = int(mask.sum())
        if area < min_area_frac * total_px:
            continue
        # Drop border-touching regions *before* merging, not just after:
        # a union of masks that each avoid the border can never itself
        # touch the border, but the converse isn't safe to rely on — a
        # single small, unrelated, border-touching SAM mask (e.g. a sliver
        # of sky) that happens to share a large interior candidate's depth
        # range would otherwise merge into it and disqualify the whole
        # group on the strength of a coincidence (confirmed: a 5,669px
        # border-touching mask merged into, and discarded, an otherwise
        # clean 348,847px cooling-tower group). Filtering here makes a
        # merged group's own border check below unreachable in practice,
        # but it's kept as a defensive backstop.
        if mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any():
            continue
        d_lo, d_hi = np.percentile(depth_raw[mask], [10, 90])
        # A region of near-uniform depth (common — flat surfaces, solid-
        # colour painted objects) has a near-zero-width percentile range.
        # Left unpadded, comparing two such regions gives an `inter` of
        # exactly 0 even when their ranges are identical, so they'd never
        # be judged to overlap. Pad every range's width by a small fraction
        # of the whole map's spread so equal (or near-equal) flat regions
        # register as fully overlapping instead of falling through the
        # inter<=0 no-overlap guard.
        d_hi = max(float(d_hi), float(d_lo) + (d_max - d_min) * 1e-4)
        regions.append({"mask": mask, "lo": float(d_lo), "hi": d_hi})

    if not regions:
        print("  [focus] no figure candidate found; falling back to d_focus=0.0",
              file=sys.stderr, flush=True)
        return 0.0, None

    # Merge same-depth regions (transitively, via union-find over the
    # pairwise >50%-both-ways overlap graph) back into single figures.
    uf = _UnionFind(len(regions))
    for i in range(len(regions)):
        for j in range(i + 1, len(regions)):
            if _ranges_overlap_50(regions[i]["lo"], regions[i]["hi"],
                                  regions[j]["lo"], regions[j]["hi"]):
                uf.union(i, j)

    groups = {}
    for i in range(len(regions)):
        groups.setdefault(uf.find(i), []).append(i)

    candidates = []
    for idxs in groups.values():
        mask = np.zeros((h, w), dtype=bool)
        for i in idxs:
            mask |= regions[i]["mask"]

        # A region touching the image border is only a partial view of
        # whatever it belongs to — cut off by the crop, not by a real
        # object boundary — so nothing measured from it (size, chunkiness,
        # convexity, connectedness) can be trusted: the true object may
        # continue, in any shape, beyond the frame. This is the standard
        # "exclude on edges" rule from classical particle/blob analysis
        # (e.g. ImageJ's "Exclude on edges"). Border-touching *regions* are
        # already dropped before merging above, so no group reaching this
        # point can touch the border either (a union of border-avoiding
        # masks can't touch the border) — this check is now unreachable in
        # practice, kept only as a defensive backstop.
        if mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any():
            continue

        area = int(mask.sum())
        if area > max_area_frac * total_px:
            continue

        mask_u8 = mask.astype(np.uint8)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        hull_area = cv2.contourArea(cv2.convexHull(np.concatenate(contours, axis=0)))
        if hull_area < 1:
            continue

        n_cc, _, cc_stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
        largest_blob = float(cc_stats[1:, cv2.CC_STAT_AREA].max()) if n_cc > 1 else float(area)

        candidates.append({
            "area": area,
            "chunkiness": _chunkiness(mask, area),
            "solidity": area / hull_area,
            "median_depth": float(np.median(depth_raw[mask])),
            "connectedness": largest_blob / area,
            "mask": mask,
        })

    if not candidates:
        print("  [focus] no figure candidate found; falling back to d_focus=0.0",
              file=sys.stderr, flush=True)
        return 0.0, None

    # Area carries the heaviest weight: a large size advantage should be
    # able to overcome losing on any two of the double-weighted shape
    # criteria (6 > 2+2), though not all three of them plus depth combined
    # (6 <= 2+2+2). Confirmed necessary on a real case, not just tuned to
    # one: SAM correctly segmented a cooling tower as a clean, 257,310px,
    # non-border-touching mask, but a small (23,824px) incidental dark-bush
    # blob nearby was *shaped* slightly better (higher chunkiness,
    # solidity, and nearer depth) and used to win outright at area weight
    # 1-4 despite being 10.8x smaller — clearly the wrong pick for "the
    # figure" of the image. Weight 5 is the exact tipping point for that
    # case; 6 leaves a real margin rather than sitting on the edge of it.
    total = (6 * _rank([c["area"] for c in candidates])
             + 2 * _rank([c["chunkiness"] for c in candidates])
             + 2 * _rank([c["solidity"] for c in candidates])
             + 1 * _rank([c["median_depth"] for c in candidates])
             + 2 * _rank([c["connectedness"] for c in candidates]))
    best = candidates[int(np.argmax(total))]

    d_focus = (best["median_depth"] - d_min) / (d_max - d_min)
    print(f"  [focus] figure detected: area={best['area']}px "
          f"chunkiness={best['chunkiness']:.2f} solidity={best['solidity']:.2f} "
          f"connectedness={best['connectedness']:.2f} -> d_focus={d_focus:.3f}",
          file=sys.stderr, flush=True)
    return d_focus, best["mask"]


def _focus_type(s):
    """argparse type for --focus: a normalised-disparity float, or 'auto'."""
    return s if s == "auto" else float(s)


def main():
    parser = argparse.ArgumentParser(
        description="Crop and apply depth-guided variable blur for wallpaper preparation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input",  help="Input image path")
    parser.add_argument("-o", "--output", required=True, help="Output image path")

    # Crop
    parser.add_argument("--no-crop", action="store_true",
                        help="Skip aspect-ratio cropping")
    parser.add_argument("--aspect", default="16:9", metavar="W:H",
                        help="Crop aspect ratio")
    parser.add_argument("--align", default="center",
                        choices=["left", "center", "right", "top", "bottom"],
                        help="Crop alignment")

    # Depth
    parser.add_argument("--model", default=_DEFAULT_MODEL, metavar="MODEL",
                        help="HuggingFace depth estimation model ID")
    parser.add_argument("--depth-only", action="store_true",
                        help="Save blur-strength map as greyscale and exit (useful for tuning)")
    parser.add_argument("--save-depth", metavar="PATH",
                        help="Also save the normalised depth map (white=near, black=far) "
                             "alongside the main output")
    parser.add_argument("--fix-sky", action="store_true",
                        help="Correct sky/foreground depth inversions (seen on stylised art) "
                             "using an Otsu-segmented Depth Anything V2 sky mask")
    parser.add_argument("--flatten-masts", action="store_true",
                        help="Flatten thin, tall, solid vertical structures (chimneys, masts) "
                             "to their own median depth, reducing internal depth noise")

    # Blur
    parser.add_argument("--blur", type=float, default=2.0, metavar="PCT",
                        help="Max disc radius as %% of image height (circle of confusion at "
                             "the farthest point from the focus plane)")
    parser.add_argument("--levels", type=int, default=16, metavar="N",
                        help="Number of depth slabs for scatter compositing")
    parser.add_argument("--focus", type=_focus_type, default=0.0, metavar="D",
                        help="Focus plane as normalised disparity: 0.0=background/infinity "
                             "(default), 1.0=foreground, or 'auto' to detect the principal "
                             "figure via classical edge detection/segmentation of the depth "
                             "map and focus on its median depth")
    parser.add_argument("--save-figure-mask", metavar="PATH",
                        help="With --focus auto, also save the winning figure region as a "
                             "greyscale mask (white=figure) alongside the main output")

    args = parser.parse_args()

    image = cv2.imread(args.input)
    if image is None:
        print(f"Error: cannot read image '{args.input}'", file=sys.stderr)
        sys.exit(1)

    if not args.no_crop:
        ratio_w, ratio_h = (int(x) for x in args.aspect.split(':'))
        image = _crop_to_aspect(image, ratio_w, ratio_h, align=args.align)

    depth_raw = _estimate_depth(image, model=args.model)

    if args.fix_sky:
        depth_raw = _fix_sky_depth(depth_raw, image)

    if args.flatten_masts:
        depth_raw = _flatten_thin_masts(depth_raw, image)

    if args.save_depth:
        d_min, d_max = float(depth_raw.min()), float(depth_raw.max())
        depth_norm = (depth_raw - d_min) / (d_max - d_min) if d_max > d_min else np.zeros_like(depth_raw)
        cv2.imwrite(args.save_depth, np.clip(depth_norm * 255, 0, 255).astype(np.uint8))
        print(f"Saved depth map '{args.save_depth}'")

    if args.focus == "auto":
        d_focus, figure_mask = _detect_figure_focus(depth_raw, image)
    else:
        d_focus, figure_mask = args.focus, None

    if args.save_figure_mask:
        mask_img = (figure_mask.astype(np.uint8) * 255 if figure_mask is not None
                   else np.zeros(depth_raw.shape, dtype=np.uint8))
        cv2.imwrite(args.save_figure_mask, mask_img)
        print(f"Saved figure mask '{args.save_figure_mask}'")

    if args.depth_only:
        d_min, d_max = float(depth_raw.min()), float(depth_raw.max())
        depth_norm = (depth_raw - d_min) / (d_max - d_min) if d_max > d_min else np.zeros_like(depth_raw)
        denom = max(d_focus, 1.0 - d_focus)
        vis   = np.abs(depth_norm - d_focus) / denom  # white = max blur, black = in focus
        ok    = cv2.imwrite(args.output, np.clip(vis * 255, 0, 255).astype(np.uint8))
    else:
        h        = image.shape[0]
        sigma_px = max(1.0, args.blur / 100.0 * h)
        result   = _depth_blur(image, depth_raw,
                               sigma_max=sigma_px,
                               d_focus=d_focus,
                               n_levels=args.levels)
        ok = cv2.imwrite(args.output, result)

    if not ok:
        print(f"Error: cannot write image '{args.output}'", file=sys.stderr)
        sys.exit(1)
    print(f"Saved '{args.output}'")


if __name__ == "__main__":
    main()
