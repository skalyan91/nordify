# nordify

Convert any image to the [Nord colour palette](https://www.nordtheme.com/). Colour snapping and dithering operate in the perceptually-uniform [Oklab](https://bottosson.github.io/posts/oklab/) colour space; palette mixing maps pixels onto a convex hull whose geometry varies by mixing model (spectral Kubelka-Munk pigment mixing, or linear-RGB additive light mixing).

## Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

For depth estimation and palette mixing, also install:

```bash
pip install transformers torch pillow  # depth_blur.py
pip install mlx                        # --mix (Apple Silicon required)
```

## Pipeline

The pipeline is two scripts run in sequence:

```
depth_blur.py  →  nordify.py
(crop + blur)     (palette conversion)
```

### Step 1 — `depth_blur.py`

Crops to a target aspect ratio, estimates monocular depth, and applies a depth-guided defocus (bokeh) blur via layered forward-scatter compositing.

```
python3 depth_blur.py <input> -o <blurred> [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--aspect W:H` | `16:9` | Crop aspect ratio |
| `--align` | `center` | Crop alignment: `left`/`center`/`right`/`top`/`bottom` |
| `--no-crop` | — | Skip aspect-ratio cropping |
| `--blur PCT` | `2.0` | Max disc radius (circle of confusion) as % of image height |
| `--levels N` | `16` | Number of depth slabs for scatter compositing |
| `--focus D` | `0.0` | Focus plane as normalised disparity: `0.0`=background/infinity, `1.0`=foreground |
| `--depth-only` | — | Save the blur-strength map and exit (useful for tuning) |
| `--model MODEL` | Depth Anything V2 Small | HuggingFace depth model ID |

### Step 2 — `nordify.py`

Maps every pixel to a Nord colour. Optionally applies a nighttime pre-processing pass before conversion.

```
python3 nordify.py <input> -o <output> [options]
```

| Flag | Description |
|------|-------------|
| `--dither fs` | Floyd-Steinberg dithering with blue-noise seeding |
| `--mix [spectral\|additive]` | Palette mixing (requires MLX); see [Methods](#palette-mixing---mix) below. |
| `--night` | Nighttime pre-processing: darken and cool the image before palette conversion |

### Utility — `entropy_crop.py`

An alternative to `depth_blur.py`'s `--align left`/`center`/`right`: picks a crop offset by minimising the entropy of the edges the crop boundary would cut, then applies that same offset to one or more images — useful for re-cropping an already-finished wallpaper to a second aspect ratio (e.g. deriving a 3:2 variant from a 16:9 one) while keeping a day/night pair aligned. See [Methods](#entropy-minimising-crop-entropy_croppy) below.

```
python3 entropy_crop.py <edge-source> --aspect W:H <in1> <out1> [<in2> <out2> ...]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--aspect W:H` | `16:9` | Target aspect ratio |
| `--visualize PATH` | — | Save a copy of `edge-source` with the chosen crop boundaries drawn on it |

`edge-source` is only used to choose the crop offset (via its own edges) — it is not implicitly cropped; include it in the `<in> <out>` pairs if it should be too. Every input must be pixel-aligned with `edge-source` (i.e. the same dimensions).

### Examples

```bash
# Plain colour snapping
python3 nordify.py photo.jpg -o photo_nord.png

# Floyd-Steinberg dithering
python3 nordify.py photo.jpg -o photo_nord.png --dither fs

# Spectral palette mixing
python3 nordify.py photo.jpg -o photo_nord.png --mix

# Additive (linear-light) palette mixing
python3 nordify.py photo.jpg -o photo_nord.png --mix additive

# Nighttime version with spectral mixing
python3 nordify.py photo.jpg -o photo_night.png --night --mix

# Full wallpaper pipeline: 16:9 crop + depth blur, then nordify
python3 depth_blur.py photo.jpg -o blurred.png
python3 nordify.py blurred.png -o wallpaper.png --mix

# 3:2 wallpaper, keeping left side, heavier blur
python3 depth_blur.py photo.jpg -o blurred.png --aspect 3:2 --align left --blur 4
python3 nordify.py blurred.png -o wallpaper.png --mix

# Derive a 3:2 crop from an already-finished 16:9 wallpaper (and its night
# variant) via minimum-entropy cropping, keeping both pixel-aligned
python3 entropy_crop.py wallpaper.png --aspect 3:2 \
    wallpaper.png wallpaper_3x2.png \
    wallpaper_night.png wallpaper_night_3x2.png
```

## Samples

Original photo by [Philippe Gauthier](https://unsplash.com/photos/orange-fruits-under-blue-sky-during-daytime-eaOjEz8746k) on Unsplash.

| | |
|---|---|
| **Original** | **Colour snapping** |
| ![Original](samples/original.jpg) | ![Snapped](samples/snapped.png) |
| **Floyd-Steinberg dithering** | **Nighttime (`--night`)** |
| ![Dithered](samples/dithered.png) | ![Night](samples/night.png) |

**Palette mixing (`--mix`):**

| `--mix spectral` (default) | `--mix additive` |
|---|---|
| ![Mixed spectral](samples/mixed.png) | ![Mixed additive](samples/mixed_additive.png) |

**Wallpaper crop + depth-guided defocus blur + spectral palette mixing (`depth_blur.py` → `nordify.py --mix`):**

![Wallpaper](samples/wallpaper.png)

## Methods

### Colour snapping

Each pixel's colour is converted to [Oklab](https://bottosson.github.io/posts/oklab/) — a perceptually uniform space — and its hue is snapped to the nearest palette hue while its lightness and chroma are left unchanged. This preserves the tonal contrast of the original image.

### Floyd-Steinberg dithering with blue noise

[Floyd-Steinberg error diffusion](https://en.wikipedia.org/wiki/Floyd%E2%80%93Steinberg_dithering) in the Oklab (a, b) plane: the palette mismatch at each pixel is propagated to its neighbours with weights 7/16, 3/16, 5/16, 1/16. Before each palette lookup the effective colour is offset by a [blue-noise](https://en.wikipedia.org/wiki/Blue_noise) value (generated via the void-and-cluster algorithm), which breaks up the banding that plain error diffusion can produce in smooth gradients.

### Palette mixing (`--mix`)

Two mixing models, both minimising Oklab distance to each pixel's original colour and both leaving already-reachable pixels unchanged.

**`--mix spectral` (default)** fits a Gaussian reflectance spectrum to each Nord colour, then for every pixel optimises a simplex (Σcᵢ = 1) over the 17 palette K/S spectra to minimise Oklab distance to the target. Mixing in spectral K/S space follows Kubelka-Munk theory: convex combinations of K/S spectra correspond to physically realised opaque paint mixtures.

The pipeline:

1. **Augmented palette** — the 17 pure colours plus all N(N−1)/2 pairwise 50/50 K/S mixtures are assembled. Each pixel is snapped to the nearest augmented entry; pairwise-mixture entries ensure boundary pixels receive interior-simplex starting weights rather than one-hot corners.
2. **Random diversification** — the snapped weights are blended 50/50 with a Dirichlet-sampled random field, then re-projected onto the simplex. This prevents pixels near palette boundaries from stalling in local optima.
3. **Spatial blur** — weights are Gaussian-smoothed across neighbours and re-projected, giving spatially coherent mixing in flat regions.
4. **Adam optimisation** — cosine-decayed Adam refines the simplex weights per strip to minimise Oklab distance to the target.

**`--mix additive`** treats the palette as a set of light sources instead of pigments: the reachable gamut is the convex hull of the 17 colours (plus black and white) in **linear RGB**, which — unlike the K/S spectral gamut — includes ordinary additive colour mixing. Out-of-gamut pixels are first clamped onto this hull, then walked back toward their original colour in Oklab space in two sequential phases — luminance, then chrominance (hue and chroma matched jointly) — each phase run with Adam. Every step's candidate is re-projected onto the hull, so the effective step is always the gradient clamped to the gamut boundary rather than a raw unconstrained step. Because the additive gamut is much larger than the pigment gamut, this model changes photographs more subtly than spectral mixing — visible mainly on strongly saturated or overexposed pixels.

For a more detailed discussion of the algorithms and their artistic rationale, see [BACKGROUND.md](BACKGROUND.md).

### Depth-guided blur (`depth_blur.py`)

Estimates monocular depth via [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) and applies a telecentric defocus (bokeh) model via layered forward-scatter compositing. The depth map is divided into `--levels` slabs with tent-function membership (an exact partition of unity); each slab forward-scatters its own colour and coverage outward by a disc (pillbox) kernel sized to that slab's circle of confusion — `r = sigma_max · |d − d_focus| / denom` — exactly as a real aperture spreads light from an out-of-focus point. Slabs are then composited front-to-back with premultiplied alpha, so nearer slabs occlude farther ones. Because blur is scattered from each source pixel outward rather than gathered into each output pixel from a neighbourhood sized by its own depth, background bokeh naturally bleeds up to (and is naturally clipped by) sharp foreground edges, and a blurred foreground naturally bleeds semi-transparently over a sharp background — with no heuristic depth dilation needed. Blur runs in linear light with MLX GPU acceleration on Apple Silicon when available.

### Entropy-minimising crop (`entropy_crop.py`)

Picks a crop offset without any notion of saliency or subject detection — only of *disruption*. For each candidate offset along the axis that needs narrowing, the crop's two boundary lines are laid over a Sobel gradient-magnitude map of the source image (continuous, not thresholded — a binary edge detector like Canny goes nearly empty on an already palette-snapped image, which is mostly large flat colour regions, and ties pathologically). Summing that gradient magnitude along both boundary lines gives an unnormalised distribution over rows (or columns, for a height crop): how much of an edge each row's slice of the boundary cuts through. The Shannon entropy of that distribution is the score for the offset — low when the cuts are concentrated in a few rows (the boundary mostly threads through flat sky, water, or background, only clipping something in a narrow band) and high when they're smeared evenly across many rows, i.e. slicing through many different objects all over the frame. An offset that cuts no edges at all scores zero entropy, the best case. The offset with the lowest entropy wins (ties broken by lowest total edge weight cut).

This has no idea what a moon or a smokestack *is* — it just avoids drawing the crop line through wherever the image is busiest, which is often, but not always, the same thing an eye would pick.

### Nighttime pre-processing (`--night`)

Transforms pixel colours in Oklab before palette conversion:

- **Luminance** — `L → 1 − √(1 − L)`: a curve that darkens mid-tones and highlights while keeping shadows from crushing to black.
- **Yellow/blue axis** — `b` is shifted toward blue in inverse proportion to luminance: dark pixels receive the full shift (warming tones become cool), bright pixels are left unchanged (artificial lights and highlights keep their original colour temperature).

## Nord Palette

| Group | Colours |
|-------|---------|
| Polar Night | nord0 `#2E3440` · nord1 `#3B4252` · nord2 `#434C5E` · nord3 `#4C566A` |
| Snow Storm | nord4 `#D8DEE9` · nord5 `#E5E9F0` · nord6 `#ECEFF4` |
| Frost | nord7 `#8FBCBB` · nord8 `#88C0D0` · nord9 `#81A1C1` · nord10 `#5E81AC` |
| Aurora | nord11 `#BF616A` · nord12 `#D08770` · nord13 `#EBCB8B` · nord14 `#A3BE8C` · nord15 `#B48EAD` |

---

*Developed with [Claude Code](https://claude.ai/code)*
