#!/usr/bin/env python3
"""Crop and apply depth-guided variable blur for wallpaper preparation.

Estimates monocular depth, smooths and normalises the map, then blurs each
pixel according to its depth value: foreground (high disparity) receives the
most blur, background the least.  Blur is applied in linear light via a
Gaussian pyramid with GPU acceleration through MLX when available.

Requirements:
    pip install transformers torch pillow
    pip install mlx   # optional — used for GPU-accelerated blur pyramid
"""

import argparse
import sys

import cv2
import numpy as np

_N_BLUR_LEVELS = 50
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


def _gaussian_blur_mlx(img_lin, sigma):
    """Separable Gaussian blur on GPU via MLX depthwise convolution.

    Pre-pads with NumPy reflect mode so boundary pixels are mirrored rather
    than treated as black; the paired convolutions consume that padding exactly.
    """
    import mlx.core as mx
    radius = round(3 * sigma)
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k = (k / k.sum()).astype(np.float32)

    padded = np.pad(img_lin, [(radius, radius), (radius, radius), (0, 0)], mode='reflect')
    t = mx.array(padded[None])  # (1, H+2r, W+2r, 3)

    kh = mx.array(np.tile(k[None, None, :, None], (3, 1, 1, 1)))
    t = mx.conv2d(t, kh, groups=3)

    kv = mx.array(np.tile(k[None, :, None, None], (3, 1, 1, 1)))
    t = mx.conv2d(t, kv, groups=3)

    mx.eval(t)
    return np.array(t[0])  # (H, W, 3)


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


def _depth_blur(image, depth_raw, sigma_max, n_levels=_N_BLUR_LEVELS, smooth_sigma=10.0,
                power=2.0):
    """Apply depth-guided variable Gaussian blur in linear light.

    depth_raw  : (H, W) float32 — raw depth (any scale); higher = more blur
    sigma_max  : maximum Gaussian sigma (pixels)
    n_levels   : blur pyramid depth (higher = smoother level transitions)
    smooth_sigma: Gaussian sigma (pixels) applied to the depth map before use
    power      : exponent applied to normalised depth before level mapping;
                 2.0 = quadratic (keeps middle ground relatively sharp)

    Pipeline: blur depth → normalise to [0,1] → depth^power → pyramid level.
    """
    # Blur the depth map, then normalise to [0, 1]
    if smooth_sigma > 0:
        ks    = int(smooth_sigma * 6) | 1
        depth = cv2.GaussianBlur(depth_raw, (ks, ks), smooth_sigma)
    else:
        depth = depth_raw.copy()

    d_min, d_max = float(depth.min()), float(depth.max())
    if d_max - d_min > 1e-6:
        depth = (depth - d_min) / (d_max - d_min)
    else:
        depth = np.zeros_like(depth)

    img_lin = _srgb_to_linear(image.astype(np.float32) / 255.0)

    try:
        import mlx.core  # noqa: F401
        _blur = lambda s: _gaussian_blur_mlx(img_lin, s)
        print("  [blur] device=mlx", file=sys.stderr, flush=True)
    except ImportError:
        _blur = lambda s: cv2.GaussianBlur(
            img_lin, (int(s * 6) | 1, int(s * 6) | 1), s,
            borderType=cv2.BORDER_REFLECT_101,
        )
        print("  [blur] device=cpu (install mlx for GPU acceleration)",
              file=sys.stderr, flush=True)

    N = n_levels
    print(f"  [blur] building {N}-level pyramid, sigma_max={sigma_max:.1f}px …",
          file=sys.stderr, flush=True)
    levels     = [img_lin] + [_blur(sigma_max * (i + 1) / N) for i in range(N)]
    levels_arr = np.stack(levels, axis=0)  # (N+1, H, W, 3)

    # Map normalised depth [0, 1] → continuous pyramid level [0, N]
    level    = np.clip(depth ** power * N, 0.0, float(N))            # (H, W)
    level_lo = np.clip(np.floor(level).astype(np.int32), 0, N - 1)
    alpha    = (level - level_lo)[..., None]                         # (H, W, 1)

    h, w  = image.shape[:2]
    ll    = np.broadcast_to(level_lo[..., None], (h, w, 3))
    rows  = np.arange(h)[:, None, None]
    cols  = np.arange(w)[None, :, None]
    chans = np.arange(3)[None, None, :]
    lo    = levels_arr[ll,     rows, cols, chans]
    hi    = levels_arr[ll + 1, rows, cols, chans]

    result_lin = (1.0 - alpha) * lo + alpha * hi
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
    parser.add_argument("--invert-depth", action="store_true",
                        help="Invert depth map so background is blurred instead of foreground")
    parser.add_argument("--depth-only", action="store_true",
                        help="Save depth map as greyscale and exit (useful for tuning)")

    # Blur
    parser.add_argument("--blur", type=float, default=2.0, metavar="PCT",
                        help="Max blur sigma as %% of image height")
    parser.add_argument("--smooth", type=float, default=1.0, metavar="PCT",
                        help="Depth-map smoothing sigma as %% of image height")
    parser.add_argument("--levels", type=int, default=_N_BLUR_LEVELS, metavar="N",
                        help="Blur pyramid levels")
    parser.add_argument("--power", type=float, default=2.0, metavar="P",
                        help="Depth-to-blur curve exponent: 1=linear, 2=quadratic (default)")

    args = parser.parse_args()

    image = cv2.imread(args.input)
    if image is None:
        print(f"Error: cannot read image '{args.input}'", file=sys.stderr)
        sys.exit(1)

    if not args.no_crop:
        ratio_w, ratio_h = (int(x) for x in args.aspect.split(':'))
        image = _crop_to_aspect(image, ratio_w, ratio_h, align=args.align)

    depth_raw = _estimate_depth(image, model=args.model)

    if args.invert_depth:
        depth_raw = depth_raw.max() - depth_raw

    if args.depth_only:
        d_min, d_max = float(depth_raw.min()), float(depth_raw.max())
        vis = (depth_raw - d_min) / (d_max - d_min) if d_max > d_min else np.zeros_like(depth_raw)
        ok  = cv2.imwrite(args.output, (vis * 255).astype(np.uint8))
    else:
        h          = image.shape[0]
        sigma_px   = max(1.0, args.blur   / 100.0 * h)
        smooth_px  = max(0.0, args.smooth / 100.0 * h)
        result     = _depth_blur(image, depth_raw,
                                 sigma_max=sigma_px,
                                 n_levels=args.levels,
                                 smooth_sigma=smooth_px,
                                 power=args.power)
        ok = cv2.imwrite(args.output, result)

    if not ok:
        print(f"Error: cannot write image '{args.output}'", file=sys.stderr)
        sys.exit(1)
    print(f"Saved '{args.output}'")


if __name__ == "__main__":
    main()
