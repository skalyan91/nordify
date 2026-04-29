# nordify

Convert any image to the [Nord colour palette](https://www.nordtheme.com/). Colour snapping and dithering operate in the perceptually-uniform [Oklab](https://bottosson.github.io/posts/oklab/) colour space; palette mixing snaps colours in linear RGB using a convex-hull model.

## Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

For palette mixing (`--mix`), also install:

```bash
pip install mlx   # Apple Silicon (required for --mix)
```

## Usage

```
python3 nordify.py <input> -o <output> [--dither fs] [--mix] [--wallpaper] [--margin PX]
```

| Flag | Description |
|------|-------------|
| `--dither fs` | Floyd-Steinberg dithering with blue-noise seeding |
| `--mix` | Palette mixing gamut mapping |
| `--wallpaper` | Crop to 16:9 and apply gradient blur at edges |
| `--margin PX` | Blur ramp width in pixels for `--wallpaper` (default: 200) |

### Examples

```bash
# Plain colour snapping
python3 nordify.py photo.jpg -o photo_nord.png

# Floyd-Steinberg dithering
python3 nordify.py photo.jpg -o photo_nord.png --dither fs

# Palette mixing
python3 nordify.py photo.jpg -o photo_nord.png --mix

# Wallpaper (16:9 crop + edge blur, combinable with any mode)
python3 nordify.py photo.jpg -o wallpaper.png --mix --wallpaper

# Wallpaper with wider blur ramp
python3 nordify.py photo.jpg -o wallpaper.png --mix --wallpaper --margin 350
```

## Samples

Original photo by [v2osk](https://unsplash.com/photos/1Z2niiBPg5A) on Unsplash.

| | |
|---|---|
| **Original** | **Colour snapping** |
| ![Original](samples/original.jpg) | ![Snapped](samples/snapped.png) |
| **Floyd-Steinberg dithering** | **Palette mixing** |
| ![Dithered](samples/dithered.png) | ![Mixed](samples/mixed.png) |

**Palette mixing + wallpaper crop:**

![Wallpaper](samples/wallpaper.png)

## Methods

### Colour snapping

Each pixel's colour is converted to [Oklab](https://bottosson.github.io/posts/oklab/) — a perceptually uniform space — and its hue is snapped to the nearest palette hue while its lightness and chroma are left unchanged. This preserves the tonal contrast of the original image.

### Floyd-Steinberg dithering with blue noise

[Floyd-Steinberg error diffusion](https://en.wikipedia.org/wiki/Floyd%E2%80%93Steinberg_dithering) in the Oklab (a, b) plane: the palette mismatch at each pixel is propagated to its neighbours with weights 7/16, 3/16, 5/16, 1/16. Before each palette lookup the effective colour is offset by a [blue-noise](https://en.wikipedia.org/wiki/Blue_noise) value (generated via the void-and-cluster algorithm), which breaks up the banding that plain error diffusion can produce in smooth gradients.

### Palette mixing (`--mix`)

Palette mixing treats Nord colours like paints: any colour achievable by mixing them is in the *palette gamut* — a [convex hull](https://en.wikipedia.org/wiki/Convex_hull) in linear RGB space. Pixels already inside the gamut pass through unchanged; pixels outside are remapped to the nearest achievable colour via a three-phase optimisation in Oklab that mirrors how an artist builds up a painting:

1. **Lightness first** — establish the light/dark structure (the *value sketch*).
2. **Hue second** — lay in colour temperature and hue relationships.
3. **Chroma last** — push or pull colour intensity while holding the value and hue already achieved.

Each phase runs [Adam gradient descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Adam) with the colour snapped back onto the hull boundary after every step. Image chroma is pre-clamped to the palette's chroma range before optimisation.

For a more detailed discussion of the algorithms and their artistic rationale, see [BACKGROUND.md](BACKGROUND.md).

### Wallpaper preparation (`--wallpaper`)

Center-crops the image to 16:9, then applies a [smoothstep](https://en.wikipedia.org/wiki/Smoothstep) gradient blur over a fixed-pixel margin on all four edges. The blurred region accommodates a menu bar (top) and dock (bottom) without visual clash. Blur strength is controlled via `--margin`.

## Nord Palette

| Group | Colours |
|-------|---------|
| Polar Night | nord0 `#2E3440` · nord1 `#3B4252` · nord2 `#434C5E` · nord3 `#4C566A` |
| Snow Storm | nord4 `#D8DEE9` · nord5 `#E5E9F0` · nord6 `#ECEFF4` |
| Frost | nord7 `#8FBCBB` · nord8 `#88C0D0` · nord9 `#81A1C1` · nord10 `#5E81AC` |
| Aurora | nord11 `#BF616A` · nord12 `#D08770` · nord13 `#EBCB8B` · nord14 `#A3BE8C` · nord15 `#B48EAD` |

---

*Developed with [Claude Code](https://claude.ai/code)*
