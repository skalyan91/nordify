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


def _estimate_depth(image_bgr, model=_DEFAULT_MODEL):
    """Run depth estimation on a BGR uint8 image.

    Returns (H, W) float32.  Output follows the disparity convention used by
    Depth Anything / MiDaS: higher values = closer to the camera (foreground).
    Values are NOT yet normalised — normalisation is deferred to after smoothing.

    Requires: pip install transformers torch pillow
    """
    try:
        from transformers import pipeline as hf_pipeline
        import torch
        from PIL import Image as PILImage
    except ImportError as e:
        print(
            f"Error: {e}\n"
            "Depth estimation requires: pip install transformers torch pillow",
            file=sys.stderr,
        )
        sys.exit(1)

    rgb     = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_img = PILImage.fromarray(rgb)

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"  [depth] model={model}  device={device}", file=sys.stderr, flush=True)

    pipe = hf_pipeline("depth-estimation", model=model, device=device)
    out  = pipe(pil_img)

    raw = np.array(out["predicted_depth"], dtype=np.float32)
    if raw.ndim > 2:
        raw = raw.squeeze()

    # Resize to input dimensions (model may process at a different resolution)
    return cv2.resize(raw, (image_bgr.shape[1], image_bgr.shape[0]),
                      interpolation=cv2.INTER_LINEAR)


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
                             "alongside the main output, e.g. for barycentre_crop.py")

    # Blur
    parser.add_argument("--blur", type=float, default=2.0, metavar="PCT",
                        help="Max disc radius as %% of image height (circle of confusion at "
                             "the farthest point from the focus plane)")
    parser.add_argument("--levels", type=int, default=16, metavar="N",
                        help="Number of depth slabs for scatter compositing")
    parser.add_argument("--focus", type=float, default=0.0, metavar="D",
                        help="Focus plane as normalised disparity: 0.0=background/infinity "
                             "(default), 1.0=foreground")

    args = parser.parse_args()

    image = cv2.imread(args.input)
    if image is None:
        print(f"Error: cannot read image '{args.input}'", file=sys.stderr)
        sys.exit(1)

    if not args.no_crop:
        ratio_w, ratio_h = (int(x) for x in args.aspect.split(':'))
        image = _crop_to_aspect(image, ratio_w, ratio_h, align=args.align)

    depth_raw = _estimate_depth(image, model=args.model)

    if args.save_depth:
        d_min, d_max = float(depth_raw.min()), float(depth_raw.max())
        depth_norm = (depth_raw - d_min) / (d_max - d_min) if d_max > d_min else np.zeros_like(depth_raw)
        cv2.imwrite(args.save_depth, np.clip(depth_norm * 255, 0, 255).astype(np.uint8))
        print(f"Saved depth map '{args.save_depth}'")

    if args.depth_only:
        d_min, d_max = float(depth_raw.min()), float(depth_raw.max())
        depth_norm = (depth_raw - d_min) / (d_max - d_min) if d_max > d_min else np.zeros_like(depth_raw)
        denom = max(args.focus, 1.0 - args.focus)
        vis   = np.abs(depth_norm - args.focus) / denom  # white = max blur, black = in focus
        ok    = cv2.imwrite(args.output, np.clip(vis * 255, 0, 255).astype(np.uint8))
    else:
        h        = image.shape[0]
        sigma_px = max(1.0, args.blur / 100.0 * h)
        result   = _depth_blur(image, depth_raw,
                               sigma_max=sigma_px,
                               d_focus=args.focus,
                               n_levels=args.levels)
        ok = cv2.imwrite(args.output, result)

    if not ok:
        print(f"Error: cannot write image '{args.output}'", file=sys.stderr)
        sys.exit(1)
    print(f"Saved '{args.output}'")


if __name__ == "__main__":
    main()
