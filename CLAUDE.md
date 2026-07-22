# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**nord-lab** is a Python CLI pipeline that prepares wallpapers in the [Nord colour palette](https://www.nordtheme.com/). `depth_blur.py` handles cropping and depth-guided blur; `nordify.py` handles palette conversion. `entropy_crop.py` is a standalone utility for choosing a crop offset algorithmically instead of via `--align`.

## Setup and Usage

```bash
source venv/bin/activate

# Step 1 — crop + depth-guided blur
python3 depth_blur.py <input> -o <blurred> [--aspect W:H] [--align ...] [--blur PCT] [--focus D]

# Step 2 — nordify
python3 nordify.py <blurred> -o <output> [--dither fs] [--mix [spectral|additive]]

# Optional — pick a crop offset by minimising cut-edge entropy, apply to N images at once
python3 entropy_crop.py <edge-source> --aspect W:H <in1> <out1> [<in2> <out2> ...]
```

Core dependencies: `numpy`, `opencv-python-headless` (in `venv/`).  
Depth estimation: `transformers`, `torch`, `pillow` (downloaded on first run from HuggingFace).  
Optional: `mlx` (Apple Silicon) — required for `--mix`; accelerates blur pyramid in `depth_blur.py`.

## Architecture

Two scripts: `depth_blur.py` (preprocessing) and `nordify.py` (palette conversion).

### Colour pipeline

- **`_srgb_to_linear` / `_linear_to_srgb`** — piecewise sRGB gamma (IEC 61966-2-1).
- **`_linear_rgb_to_oklab` / `_oklab_to_linear_rgb`** — Björn Ottosson's Oklab matrices via the LMS cube-root intermediate.
- **`_bgr_to_oklab` / `_oklab_to_bgr`** — OpenCV BGR ↔ Oklab, handling channel order and float normalisation.

### Palette

**`PALETTE_BGR`** — 17 colours: black `(0,0,0)` plus nord0–nord15 in BGR order (Polar Night × 4, Snow Storm × 3, Frost × 4, Aurora × 5). Black extends the lightness range for `--mix`.

**`_fit_palette_ks()`** — fits a reflectance spectrum to each palette colour. Spectral colours (single dominant channel) get a single Gaussian `R_base + A·exp(−(λ−λ₀)²/2σ²)`. Extra-spectral colours (purple/magenta: `g < r AND g < b`) use a **bi-Gaussian** with one red lobe (λ ∈ [560, 700] nm) and one blue/violet lobe (λ ∈ [400, 490] nm) — physically necessary because purple cannot be produced by any single wavelength.

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

Two interchangeable models, both MLX-only and both leaving in-hull/in-simplex pixels unchanged.

**`--mix spectral` (default) — `mix_convert_spectral(image)`.** Fits a Gaussian (or bi-Gaussian) reflectance spectrum to each palette colour via `_fit_palette_ks()`, then for every pixel optimises a simplex of weights (Σcᵢ = 1) over the palette's K/S spectra to minimise Oklab distance to the target, mixing via Kubelka-Munk. Builds an augmented palette (pure colours + all pairwise 50/50 K/S mixtures), snaps each pixel to the nearest augmented entry, blurs, and re-snaps to seed the simplex weights before a phased Adam optimisation (`_mix_strip_spectral`) run strip-by-strip; `_simplex_project_mlx` projects onto the probability simplex after every step.

**`--mix additive` — `mix_convert_additive(image)`.** Convex hull of the palette in **linear RGB** (light/additive mixing), represented as half-spaces via `_halfspace_eqs(points)` (O(N⁴) brute force; fine for N ≤ 20 entries). Per pixel (`_mix_strip_additive`), out-of-hull colours are first clamped onto the hull, then walked back toward the pixel's original colour in Oklab space via a **two-phase Adam optimisation**:
- **Phase 1 (luminance):** minimises `(L − L_target)²`
- **Phase 2 (chrominance):** minimises `(a − a_target)² + (b − b_target)²` (hue and chroma jointly) + 1000 × luminance penalty

After every Adam step, the candidate is re-projected onto the hull (20 iterations of half-space projection) — so the effective, gamut-clamped step is `projected_candidate − colour`, not the raw Adam update. Adam's per-coordinate adaptivity is needed because phase 2's penalty term makes the loss ill-conditioned; plain gradient descent stalls on it. (An earlier three-phase variant — luminance, then hue, then chroma sequentially — converges to a similar result but needs far more iterations per pixel, since splitting hue and chroma sharpens that ill-conditioning further.) In-hull pixels are skipped entirely (zero gradient at the target). All GPU computation stays on MLX (Apple Silicon) — no host/device round-trips inside the optimisation loop beyond the scalar loss used for convergence/progress checks.

## depth_blur.py

Preprocessing script: crop → depth estimation → depth-guided blur.

### Cropping

**`_crop_to_aspect(image, ratio_w, ratio_h, align='center')`** — crops to the given aspect ratio. `align` is `'left'`/`'center'`/`'right'` when width is cropped, `'top'`/`'center'`/`'bottom'` when height is cropped. Run by default; skip with `--no-crop`.

### Depth estimation

**`_estimate_depth(image_bgr, model)`** — runs Depth Anything V2 Small (`depth-anything/Depth-Anything-V2-Small-hf`) via the HuggingFace `transformers` pipeline. Uses MPS on Apple Silicon, CPU otherwise. Returns raw `(H, W)` float32 depth map (disparity convention: higher = closer/foreground). Resizes model output back to input resolution.

### Depth-guided blur

**`_make_disc_kernel(radius)`** — builds a normalised uniform disc (pillbox) kernel with a 1-pixel anti-aliased edge ramp (`clip(radius − dist + 0.5, 0, 1)`). Returns a 1×1 identity when `radius < 0.5`.

**`_disc_blur_mlx(img, radius)`** / **`_disc_blur_cpu(img, radius)`** — disc blur for `(H, W, C)` arrays (any C) on GPU via MLX 2-D depthwise `conv2d` (weight shape `(C, kH, kW, 1)`, `groups=C`) or CPU via `cv2.filter2D`. Both reflect-pad at boundaries.

**`_depth_blur(image, depth_raw, sigma_max, d_focus=0.0, n_levels=16)`** — **layered forward-scatter compositing**. Divides depth into K=`n_levels` slabs with tent-function membership (exact partition of unity). For each slab k (front-to-back order, foreground first): packs `image × mask_k` and `mask_k` as a 4-channel array, blurs with disc radius `rₖ = sigma_max · |dₖ − d_focus| / max(d_focus, 1−d_focus)` (telecentric CoC), then composites with premultiplied alpha (`color_acc += remaining · color_k`, `weight_acc += remaining · alpha_k`). Background bokeh circles at foreground edges emerge naturally from the forward scatter — no depth dilation needed. Un-premultiplies at the end (`result / weight_acc`). `--depth-only` saves the blur-strength map `|d − d_focus| / denom` (white = max blur, black = in-focus plane).

## entropy_crop.py

Standalone utility: picks a crop offset by minimising the entropy of the edges it would cut, then applies that offset to one or more pixel-aligned images. No saliency/subject model — it only avoids cutting through wherever the image is locally busiest.

**`_gradient_magnitude(image_bgr)`** — Sobel gradient magnitude (`sqrt(gx² + gy²)`) on a lightly Gaussian-blurred (`σ=1.5`) grayscale image. Continuous, not thresholded: a binary detector like Canny goes nearly empty on an already palette-snapped image (mostly large flat colour regions), which ties candidate offsets pathologically.

**`find_crop_offset(image_bgr, ratio_w, ratio_h)`** — crops whichever axis the target ratio requires narrowing (same width-vs-height test as `depth_blur.py`'s `_crop_to_aspect`). For every candidate offset, sums the gradient magnitude along the two boundary lines (rows for a width crop, columns for a height crop) into a per-row/column profile, normalises it to a probability distribution, and scores it by Shannon entropy — `0` bits if the offset cuts no edges at all (the ideal case), low when cut weight is concentrated in a few rows/columns, high when smeared evenly across many (i.e. slicing through many different objects). Returns the offset with lowest entropy, ties broken by lowest total edge weight cut (`np.lexsort`).

**`main()`** — reads one edge-detection source image, computes the offset once via `find_crop_offset`, then applies `_apply_crop` identically to every `<in> <out>` pair given (asserting each input is pixel-aligned with the edge source) — so a day/night pair of the same wallpaper crops in lock-step. `--visualize` optionally dumps the edge source with the chosen boundary lines drawn on it, for sanity-checking.
