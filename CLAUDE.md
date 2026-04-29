# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**nord-lab** is a Python CLI tool that converts images to the [Nord colour palette](https://www.nordtheme.com/) using perceptually-uniform Oklab colour snapping, with optional dithering, palette mixing, and wallpaper preparation.

## Setup and Usage

```bash
source venv/bin/activate
python3 nordify.py <input> -o <output> [--dither fs] [--mix] [--wallpaper [--margin PX]]
```

Core dependencies: `numpy`, `opencv-python-headless` (in `venv/`).  
Optional: `mlx` (Apple Silicon) — required for `--mix`.

## Architecture

Everything lives in `nordify.py`. No other source files.

### Colour pipeline

- **`_srgb_to_linear` / `_linear_to_srgb`** — piecewise sRGB gamma (IEC 61966-2-1).
- **`_linear_rgb_to_oklab` / `_oklab_to_linear_rgb`** — Björn Ottosson's Oklab matrices via the LMS cube-root intermediate.
- **`_bgr_to_oklab` / `_oklab_to_bgr`** — OpenCV BGR ↔ Oklab, handling channel order and float normalisation.

### Palette

**`PALETTE_BGR`** — 17 colours: black `(0,0,0)` plus nord0–nord15 in BGR order (Polar Night × 4, Snow Storm × 3, Frost × 4, Aurora × 5). Black extends the lightness range for `--mix`.

**`build_lookup()`** — converts each palette entry to Oklab `(L, a, b)` and Oklch hue `H = arctan2(b, a)`; called once at startup.

### Conversion (`convert`)

For every pixel: BGR uint8 → Oklab `(L, a, b)` → Oklch chroma `C = sqrt(a²+b²)`.

The hue is snapped: the palette entry chosen by the dithering mode determines `H_out`; the output colour is reconstructed as `(L, C·cos(H_out), C·sin(H_out))` — preserving the pixel's own lightness and chroma while adopting a palette hue.

### Dithering modes

**No dithering (default)** — nearest palette entry by 3D Oklab distance; its hue is applied.

**`--dither fs`** — Floyd-Steinberg error diffusion in `(a, b)` space. Sequential pixel scan; error weights 7/16, 3/16, 5/16, 1/16. A `(thresh − 0.5) × scale` offset (scale = RMS palette chroma spread) is added to `(a_eff, b_eff)` before each palette lookup, seeding the diffusion with blue-noise statistics.

### Blue-noise threshold texture

**`_gen_blue_noise(size=64)`** — void-and-cluster algorithm (Ulichney 1993) using FFT convolution. Generated once on first use; cached in `_BN_TEXTURE`.

**`_threshold_texture(rows, cols)`** — tiles the blue-noise texture to image size.

### Palette mixing (`--mix`)

**`_halfspace_eqs(points)`** — computes the half-space representation of the convex hull of `points` (N, 3). O(N⁴) brute force; fine for N ≤ 20 entries.

**`mix_convert(image)`** — entry point for `--mix`. Clamps image pixel chroma to the palette's Oklab chroma range `[C_min, C_max]` as preprocessing (scaling `a`/`b` in Oklab, converting back to linear RGB), then builds RGB hull equations once and processes the image in 256-row strips via `_mix_strip`.

**`_mix_strip(strip_lin, hull_eqs)`** — optimises out-of-hull pixels in linear RGB space via three sequential Adam phases:
- **Phase 1 (lightness):** minimises `(L − L_target)²`
- **Phase 2 (hue):** minimises cross-product hue error + 1000 × L penalty
- **Phase 3 (chroma):** minimises `(C − C_target)²` + 1000 × hue + 1000 × L penalties

After every Adam step, `snap()` projects the colour onto the RGB convex hull via 20 iterations of half-space projection (no GPU sync — fully fused MLX graph). In-hull pixels are returned unchanged. All GPU computation runs on MLX (Apple Silicon).

### Wallpaper preparation (`--wallpaper`)

**`_crop_16_9(image)`** — center-crops to 16:9 aspect ratio. Applied before nordification so fewer pixels are processed.

**`_edge_blur(image, margin=200)`** — blends a Gaussian-blurred version at the edges using a smoothstep ramp of width `margin` pixels (identical on all four sides). Sigma = `margin / 3`. Applied after nordification. Controlled via `--margin PX`.
