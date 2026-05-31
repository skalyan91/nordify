# nordify

Convert any image to the [Nord colour palette](https://www.nordtheme.com/). Colour snapping and dithering operate in the perceptually-uniform [Oklab](https://bottosson.github.io/posts/oklab/) colour space; palette mixing maps pixels onto a convex hull whose geometry varies by mixing model (additive, pigment, or opaque paint).

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

Crops to a target aspect ratio, estimates monocular depth, and applies a depth-guided Gaussian blur (foreground blurred most, background least).

```
python3 depth_blur.py <input> -o <blurred> [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--aspect W:H` | `16:9` | Crop aspect ratio |
| `--align` | `center` | Crop alignment: `left`/`center`/`right`/`top`/`bottom` |
| `--no-crop` | — | Skip aspect-ratio cropping |
| `--blur PCT` | `2.0` | Max blur sigma as % of image height |
| `--smooth PCT` | `1.0` | Depth-map smoothing sigma as % of image height |
| `--power P` | `2.0` | Depth-to-blur curve exponent (`1`=linear, `2`=quadratic) |
| `--invert-depth` | — | Blur background instead of foreground |
| `--depth-only` | — | Save normalised depth map and exit |
| `--model MODEL` | Depth Anything V2 Small | HuggingFace depth model ID |

### Step 2 — `nordify.py`

Maps every pixel to a Nord colour. Optionally applies a nighttime pre-processing pass before conversion.

```
python3 nordify.py <input> -o <output> [options]
```

| Flag | Description |
|------|-------------|
| `--dither fs` | Floyd-Steinberg dithering with blue-noise seeding |
| `--mix [MODEL]` | Palette mixing gamut mapping (requires MLX). `MODEL` is `additive` (default), `pigment`, or `opaque`; see [Methods](#palette-mixing---mix) below. |
| `--night` | Nighttime pre-processing: darken and cool the image before palette conversion |

### Examples

```bash
# Plain colour snapping
python3 nordify.py photo.jpg -o photo_nord.png

# Floyd-Steinberg dithering
python3 nordify.py photo.jpg -o photo_nord.png --dither fs

# Palette mixing — additive (default), pigment, or opaque
python3 nordify.py photo.jpg -o photo_nord.png --mix
python3 nordify.py photo.jpg -o photo_nord.png --mix pigment
python3 nordify.py photo.jpg -o photo_nord.png --mix opaque

# Nighttime version with palette mixing
python3 nordify.py photo.jpg -o photo_night.png --night --mix

# Full wallpaper pipeline: 16:9 crop + depth blur, then nordify (opaque paint model)
python3 depth_blur.py photo.jpg -o blurred.png --smooth 7
python3 nordify.py blurred.png -o wallpaper.png --mix opaque

# 3:2 wallpaper, keeping left side, heavier blur
python3 depth_blur.py photo.jpg -o blurred.png --aspect 3:2 --align left --blur 4
python3 nordify.py blurred.png -o wallpaper.png --mix opaque
```

## Samples

Original photo by [Philippe Gauthier](https://unsplash.com/photos/orange-fruits-under-blue-sky-during-daytime-eaOjEz8746k) on Unsplash.

| | |
|---|---|
| **Original** | **Colour snapping** |
| ![Original](samples/original.jpg) | ![Snapped](samples/snapped.png) |
| **Floyd-Steinberg dithering** | **Nighttime (`--night`)** |
| ![Dithered](samples/dithered.png) | ![Night](samples/night.png) |

**Palette mixing — `--mix [additive|pigment|opaque]`:**

| `--mix additive` | `--mix pigment` | `--mix opaque` |
|---|---|---|
| ![Mixed additive](samples/mixed.png) | ![Mixed pigment](samples/mixed_pigment.png) | ![Mixed opaque](samples/mixed_opaque.png) |

**Wallpaper crop + depth blur + opaque palette mixing (`depth_blur.py --smooth 7` → `nordify.py --mix opaque`):**

![Wallpaper](samples/wallpaper.png)

## Methods

### Colour snapping

Each pixel's colour is converted to [Oklab](https://bottosson.github.io/posts/oklab/) — a perceptually uniform space — and its hue is snapped to the nearest palette hue while its lightness and chroma are left unchanged. This preserves the tonal contrast of the original image.

### Floyd-Steinberg dithering with blue noise

[Floyd-Steinberg error diffusion](https://en.wikipedia.org/wiki/Floyd%E2%80%93Steinberg_dithering) in the Oklab (a, b) plane: the palette mismatch at each pixel is propagated to its neighbours with weights 7/16, 3/16, 5/16, 1/16. Before each palette lookup the effective colour is offset by a [blue-noise](https://en.wikipedia.org/wiki/Blue_noise) value (generated via the void-and-cluster algorithm), which breaks up the banding that plain error diffusion can produce in smooth gradients.

### Palette mixing (`--mix`)

Palette mixing treats Nord colours like paints: any colour achievable by mixing them is the *palette gamut* — a [convex hull](https://en.wikipedia.org/wiki/Convex_hull) built in a colour space that matches the physical mixing model. Pixels already inside the gamut pass through unchanged; pixels outside are remapped to the nearest achievable colour.

Three mixing models are available via `--mix [MODEL]`:

| Model | Space | Physical meaning |
|-------|-------|-----------------|
| `additive` (default) | Linear RGB | Mixing light beams; convex combinations are intensity-weighted averages |
| `pigment` | log(linear RGB + ε) | Transparent pigments (watercolour, ink): Beer's law, T\_mix = T\_A^w · T\_B^(1−w) — midpoint mixtures are darker than the arithmetic mean |
| `opaque` | Kubelka-Munk K/S = (1−R)²/2R | Opaque paints: K/S ratios mix linearly, giving the muddy midpoints characteristic of physical paint |

Once the working space is chosen, out-of-gamut pixels are remapped via a three-phase [Adam](https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Adam) optimisation in Oklab, mirroring how an artist builds up a painting:

1. **Lightness first** — establish the light/dark structure (the *value sketch*).
2. **Hue second** — lay in colour temperature and hue relationships.
3. **Chroma last** — push or pull colour intensity while holding the value and hue already achieved.

After every Adam step the colour is snapped back onto the hull boundary.

#### Opaque model: phased optimisation

Raw K/S values span 0–499 (pure white to pure black), making the hull poorly conditioned for gradient-based optimisation. Each phase therefore runs in two halves:

1. **First half — log(1+K/S) space.** The reparameterisation compresses the range to [0, 6.2], giving the POCS snap and Adam well-conditioned steps throughout the reflectance range.
2. **Second half — raw K/S space.** The point is converted to K/S, re-snapped onto the raw-K/S hull, and refined with strict Kubelka-Munk mixing as the final constraint.

This gives smooth convergence (no posterisation) while ensuring the result satisfies exact K/S mixing theory.

For a more detailed discussion of the algorithms and their artistic rationale, see [BACKGROUND.md](BACKGROUND.md).

### Depth-guided blur (`depth_blur.py`)

Estimates monocular depth via [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) and builds a Gaussian blur pyramid. Each pixel is blended between pyramid levels according to its (smoothed, normalised) depth value raised to `--power`, so foreground objects receive strong blur while the background stays sharp. Blur and pyramid construction use MLX GPU acceleration on Apple Silicon when available.

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
