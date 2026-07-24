# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**nord-lab** is a Python CLI pipeline that prepares wallpapers in the [Nord colour palette](https://www.nordtheme.com/). `depth_blur.py` handles cropping and depth-guided blur; `nordify.py` handles palette conversion. `barycentre_crop.py` is a standalone utility for choosing a crop offset algorithmically instead of via `--align`.

## Setup and Usage

```bash
source venv/bin/activate

# Step 1 — crop + depth-guided blur
python3 depth_blur.py <input> -o <blurred> [--aspect W:H] [--align ...] [--blur PCT] [--focus D]

# Step 2 — nordify
python3 nordify.py <blurred> -o <output> [--dither fs] [--mix [spectral|additive]]

# Optional — pick a crop offset by centreing the depth map's barycentre, apply to N images at once
python3 depth_blur.py <input> -o <blurred> --no-crop --save-depth <depth.png>
python3 barycentre_crop.py <depth.png> --aspect W:H <in1> <out1> [<in2> <out2> ...]
```

Core dependencies: `numpy`, `opencv-python-headless` (in `venv/`).  
Depth estimation: `transformers`, `torch`, `torchvision` (required by `DepthProImageProcessor`), `pillow` (downloaded on first run from HuggingFace).  
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

**`_estimate_depth(image_bgr, model, tile_width_frac, tile_overlap)`** — runs Depth Anything V2 Small (`depth-anything/Depth-Anything-V2-Small-hf`, the default) via the HuggingFace `transformers` pipeline (MPS on Apple Silicon, CPU otherwise). Depth Anything is a fixed-input-size model — its image processor resizes everything down to a small fixed size (e.g. 518px) regardless of source resolution — so a single whole-image ("global") pass destroys thin foreground structures (wires, masts, lattice towers) before the network ever sees them, and silently inherits whatever's behind them in the depth map. `_estimate_depth` recovers that detail with **`_tile_refine`**, run multiple times at different footprints and combined by taking the per-pixel maximum:

**`_tile_refine(rgb, infer_fn, reference, native_tile, footprint, overlap_frac)`** — one tiled refinement pass. Crops overlapping tiles sized to `footprint` (downsampled to the model's native input resolution — `native_tile`, e.g. 518px — if `footprint` is larger), each least-squares fit (`np.linalg.lstsq`, subsampled 4×4) — scale + shift — to `reference`'s own values in its footprint, since a depth model's output is only defined up to an unknown per-inference affine transform (it's trained with a scale/shift-invariant loss). **`_tile_starts(length, tile, stride)`** computes 1-D tile offsets snapped flush to the far edge, and **`_feather_weights(start, length, total, overlap)`** produces 1-D tent weights (ramping 0→1/1→0 only on sides with a neighbouring tile, full weight to the image border otherwise) whose 2-D outer product blends each tile's contribution into a shared accumulator — the same premultiplied-accumulate-then-normalise pattern `_depth_blur` uses for its slabs.

`_estimate_depth`'s default `tile_width_frac = [0.5, 0.25, 0.125]` runs three such passes, each sizing its tile *footprint* as that fraction of the image's width (larger than native resolution, so each tile is downsampled before inference) and each aligned directly against the global pass, not chained through each other. The final map is `np.maximum` across the global pass and all three — **not** an average or a cascade. Different footprints have different blind spots that don't overlap much, and max lets each pass contribute only where it's more confident something is close, without any pass's mistake erasing another's correct detail:

- A wide pass (`0.5` — footprint spans half the image) gives smooth, internally consistent depth on solid objects — confirmed: a power-station smokestack's per-row disparity that wobbled non-monotonically by ±15/255 with a single native-zoom pass became visually uniform top to bottom. But it blends soft, semi-transparent things (steam, smoke) into the sky behind them, and past roughly a 2.5x per-tile downsample it loses thin wires outright.
- A narrow pass (`0.125`, close to native resolution) keeps fine detail that wide passes lose — confirmed: a lattice transmission tower's individual crossing struts, an unresolved blur at `0.5`, are fully resolved at `0.125` — but is noisy on large solid objects the same way a lone native-zoom pass always was.
- Averaging or chaining (tried first) doesn't work: two sources with the *same* blind spot (e.g. global and a wide pass both failing to separate steam from sky) just average to another version of that blind spot, and a narrow pass forced to align against an already-wrong wide-pass reference inherits the mistake instead of correcting it. Max sidesteps both, since a correct, nearer reading from any single pass always wins regardless of what the others say.

Earlier, Apple's Depth Pro (`apple/DepthPro-hf`) was the default — a multi-scale ViT that derives patches from multiple downsamplings of the *original* image internally, so a single pass sees fine detail without any tiling at all. It's still available via `--model apple/DepthPro-hf` and `_estimate_depth` still special-cases it (`"DepthPro" in type(pipe.model).__name__`, converting its metric depth in metres to disparity via `1 / depth` and skipping the tiling path entirely, since tiling measurably doesn't help it). It was dropped as the default because on stylised/painted art specifically it can misjudge large flat regions (see `--fix-sky` below) in a way Depth Anything V2 doesn't, and — with the multi-pass tiling above — Depth Anything now matches or exceeds it on thin-structure detail too.

Output is always `(H, W)` float32, disparity convention (higher = closer/foreground).

### Sky depth correction (`--fix-sky`)

No longer needed by the default pipeline — this was specifically a Depth Pro correction, and Depth Anything V2 (the new default) doesn't make the mistake it corrects. Still available and functional via `--fix-sky` for anyone using `--model apple/DepthPro-hf`.

Depth Pro, trained on real photographs, can read a flat, desaturated, silhouette-like region as *near* — a strong learned cue for atmospheric haze in photos — even when it's the sky sitting behind a much nearer, plainly-painted structure. Confirmed on a power-station painting: Depth Pro placed the sky at disparity 0.11 and the building in front of it at 0.08 (sky nearer — backwards), while Depth Anything V2 placed them at 0.11 and 0.40 respectively (correct) on the same image. This is an out-of-distribution failure specific to stylised/painted content, not a general flaw in Depth Pro (it's still clearly better for thin structures) or in this pipeline's disparity handling (the same maps put foreground grass at the correct, unambiguous near end in both cases).

**`_otsu_sky_mask(disp, feather=3.0)`** — segments the farthest, most tightly-clustered region of a disparity map (typically sky) via Otsu's threshold, which picks the split minimising intra-class variance — effective here since sky sits in a narrow low-disparity band clearly apart from a scene's much wider spread (measured: sky std ≈ tight cluster around a mean of ~11/255 vs. a scene spanning 14–232). Returns a feathered `(H, W)` float32 mask (1 = sky) so a hard per-pixel threshold doesn't stairstep the blend seam.

**`_fix_sky_depth(depth_disp, image_bgr, sky_model=...)`** — runs Depth Anything V2 (via `_estimate_depth`, so it goes through the tiled-refinement path above) purely to locate and correct the sky: `_otsu_sky_mask` segments its map, then Depth Anything's sky-region values are least-squares fit (scale + shift) to `depth_disp`'s scale using the *non-sky* region — never the sky region itself, since fitting sky-to-sky would just reproduce whatever's wrong with `depth_disp`'s own sky values — before being blended into `depth_disp` through the feathered mask. A final safety clamp caps the corrected sky at the minimum non-sky disparity, guaranteeing it reads as at least as far as everything in front of it even if the fit is imperfect. Wired into `depth_blur.py`'s CLI as `--fix-sky`; doubles depth-estimation cost (Depth Anything's tiled pass runs in addition to Depth Pro's global pass) but that's still negligible next to `nordify.py --mix`'s per-image runtime.

### Mast depth flattening (`--flatten-masts`)

Superseded by `_estimate_depth`'s own multi-pass max combination for the default model — a `0.5`-footprint pass already gives smokestacks smooth, internally consistent depth directly, without needing SAM segmentation afterward. Still available and functional via `--flatten-masts`.

Both Depth Pro and Depth Anything V2 can render a thin, tall, rigid vertical structure (a power-station smokestack) with substantial internal depth noise along its height — confirmed non-monotonic wobbles of ±15/255 with no plausible perspective explanation for an object whose true depth barely varies top to bottom.

**`_flatten_thin_masts(depth_disp, image_bgr, sam_model=..., target_width=1500, min_aspect=2.0, min_height_frac=0.05, max_relative_spread=0.15, feather=2.0)`** — finds candidate masts via SAM's automatic mask generation (the `mask-generation` pipeline) on the *source image*, not any depth map: using a depth map to find its own errors is circular, and an Otsu split on either model's map here lumps the (fairly dark) smokestacks in with sky rather than isolating them. An earlier classical approach (black-hat/top-hat contrast + connected components, filtered by aspect ratio and solidity) worked but was unreliable — it could only separate a mast from a *locally uniform* backdrop, missing one entirely when it was embedded in busy, high-contrast painted clouds.

Each surviving mask's depth is *not* flattened to a single value outright — a mast large enough to show genuine perspective drift across its height would lose that real variation. Each pixel's deviation from the mask's own median is instead clamped to a tolerance band (a soft winsorize, not a hard replace): deviations already inside the band pass through untouched, and only the excess beyond it — the part with no plausible perspective explanation — gets pulled in. That band shrinks quadratically, not linearly, toward the background: disparity is proportional to inverse distance, so for a mast of fixed real-world height, the disparity range its own top-to-bottom depth difference could plausibly produce shrinks with the square of distance, not linearly — a mast twice as far away should be allowed a quarter of the tolerance, not half. `allowed = max_relative_spread * median_val * (median_val / scene_max)`: at `median_val == scene_max` (the nearest thing in the whole scene) it reduces to the old linear `max_relative_spread * median_val`, and falls away quadratically for anything farther back (measured: full 15% tolerance at the scene's nearest point, ~2% for a background-level smokestack).

The clamp is also one-sided: only deviations reading *farther* than the mask's own median are pulled in (`np.maximum(depth_disp - median_val, -allowed)`, no upper bound); a mask can't have something genuinely farther "through" it, so an excess-far pixel is unambiguous noise, but a pixel reading *nearer* could be something real crossing in front of it — confirmed necessary: a transmission wire passing in front of a smokestack was getting its correct, nearer depth erased and pulled back to the smokestack's own median before this was one-sided.

The image is downsampled to `target_width` before running SAM. SAM's ViT encoder resizes to a fixed internal resolution regardless of input size, so running it on the full-resolution source doesn't improve mask quality at all — confirmed empirically, full-resolution and a 1500px-wide downsample of the same image produce the *same* mask overlap against the true mast shape (~0.6) — it only makes every downstream step of automatic mask generation (per-point decoding, mask upsampling, NMS across ~1000 candidate masks) far more expensive for no benefit: ~17 minutes for one 4500px-wide image vs. ~17 seconds downsampled, a ~60x difference for identical output. Each mask's own bounding-box aspect ratio is enough to keep just the mast-like ones — no solidity check is needed the way the classical approach required one, since SAM's masks already respect real object boundaries: the transmission tower's wire lattice isn't returned as any tall/narrow mask in the first place (confirmed empirically — zero overlap between the tower region and any aspect-qualifying mask). This version fixed *both* smokestacks tested (including the one against busy clouds that defeated the classical approach), each in a few seconds.

### Depth-guided blur

**`_make_disc_kernel(radius)`** — builds a normalised uniform disc (pillbox) kernel with a 1-pixel anti-aliased edge ramp (`clip(radius − dist + 0.5, 0, 1)`). Returns a 1×1 identity when `radius < 0.5`.

**`_disc_blur_mlx(img, radius)`** / **`_disc_blur_cpu(img, radius)`** — disc blur for `(H, W, C)` arrays (any C) on GPU via MLX 2-D depthwise `conv2d` (weight shape `(C, kH, kW, 1)`, `groups=C`) or CPU via `cv2.filter2D`. Both reflect-pad at boundaries.

**`_depth_blur(image, depth_raw, sigma_max, d_focus=0.0, n_levels=16)`** — **layered forward-scatter compositing**. Divides depth into K=`n_levels` slabs with tent-function membership (exact partition of unity). For each slab k (front-to-back order, foreground first): packs `image × mask_k` and `mask_k` as a 4-channel array, blurs with disc radius `rₖ = sigma_max · |dₖ − d_focus| / max(d_focus, 1−d_focus)` (telecentric CoC), then composites with premultiplied alpha (`color_acc += remaining · color_k`, `weight_acc += remaining · alpha_k`). Background bokeh circles at foreground edges emerge naturally from the forward scatter — no depth dilation needed. Un-premultiplies at the end (`result / weight_acc`). `--depth-only` saves the blur-strength map `|d − d_focus| / denom` (white = max blur, black = in-focus plane); `--save-depth` saves the plain normalised depth map (no focus-relative transform) alongside the main blurred output, for `barycentre_crop.py`.

## barycentre_crop.py

Standalone utility: picks a crop offset that centres the depth map's barycentre, then applies that offset to one or more pixel-aligned images. No edge/saliency model — reuses the depth map `depth_blur.py` already computes.

**`find_crop_offset(depth_map, ratio_w, ratio_h)`** — crops whichever axis the target ratio requires narrowing (same width-vs-height test as `depth_blur.py`'s `_crop_to_aspect`). Sums the depth map's pixel values (mass) along the other axis into a marginal per-row/column profile, takes its weighted mean coordinate as the barycentre, then centres the crop window on it: `offset = round(barycentre − new_length/2)`, clamped to `[0, excess]`. A depth map with ~zero total mass (degenerate/flat) falls back to a plain centred crop.

**`main()`** — reads one depth-map image, computes the offset once via `find_crop_offset`, then applies `_apply_crop` identically to every `<in> <out>` pair given (asserting each input is pixel-aligned with the depth map) — so a day/night pair of the same wallpaper crops in lock-step. `--visualize` optionally dumps the depth map with the barycentre and chosen boundary lines drawn on it, for sanity-checking.
