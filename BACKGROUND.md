# Background

## The Nord palette and its gamut

[Nord](https://www.nordtheme.com/) is a 16-colour palette — four dark greys (Polar Night), three near-whites (Snow Storm), four teal-blues (Frost), and five muted Aurora tones. Because it has so few colours, not every real-world colour can be expressed as a pure Nord colour. The set of colours that *can* be expressed — either as a palette entry or as a [convex combination](https://en.wikipedia.org/wiki/Convex_combination) (a weighted average) of palette entries — forms the *palette gamut*: a small, closed [convex hull](https://en.wikipedia.org/wiki/Convex_hull) inside the RGB cube. Colours inside the hull can be reproduced by mixing; colours outside cannot, and must be remapped.

## Perceptual colour space: Oklab

Ordinary RGB is a poor basis for colour arithmetic. Equal numerical distances in RGB do not correspond to equal perceived differences, which means that a nearest-neighbour search in RGB gives visually uneven results.

[Oklab](https://bottosson.github.io/posts/oklab/) (Björn Ottosson, 2020) is a perceptually uniform colour space: Euclidean distance in Oklab correlates well with how different two colours look to a human observer. It has three axes:

- **L** — lightness (0 = black, 1 = white)
- **a**, **b** — opponent colour channels (together encoding hue and chroma)

Two derived quantities are useful throughout:

- **Chroma** C = √(a² + b²) — how saturated the colour is
- **Hue** H = arctan2(b, a) — the colour's position on the colour wheel

All colour comparisons and optimisations in nordify run in Oklab.

## Colour snapping

For colours already close to a palette hue, the simplest mapping preserves as much of the original image as possible: keep the pixel's own lightness L and chroma C, and only change the hue H to the nearest palette hue. The output colour is reconstructed as (L, C·cos(H_out), C·sin(H_out)) in Oklab.

This is analogous to adjusting the colour temperature of a photograph without altering its exposure: the light/dark relationships — what painters call *values* — survive intact.

## Floyd-Steinberg dithering with blue noise

The idea behind dithering is as old as printmaking: when you have only a few colours (or only black and white), you can simulate intermediate tones by mixing small marks of different colours close together, letting the eye average them at a distance.

[Pointillist](https://en.wikipedia.org/wiki/Pointillism) painters like Seurat and Signac exploited exactly this — building luminous colour from tiny dots of pure pigment placed side by side. The same principle appears in the engraved portraits on banknotes: a master engraver renders shadow, skin tone, and fabric using nothing but lines and stippled dots of a single ink colour, relying on their density and spacing to suggest a full tonal range.

[Error diffusion](https://en.wikipedia.org/wiki/Error_diffusion), and [Floyd-Steinberg](https://en.wikipedia.org/wiki/Floyd%E2%80%93Steinberg_dithering) in particular, automates this process. When a pixel cannot be reproduced exactly with the available palette, the mismatch (the *quantisation error*) is spread forward to neighbouring pixels so that the local average colour remains correct. The result is a fine-grained pattern of palette colours whose density varies with the underlying image, just as an engraver varies the density of hatching to model form.

With Nord's 16 colours this works well in areas of high contrast, but in smooth gradients the sequential scan pattern of raw Floyd-Steinberg can produce worm-like streaks — the digital equivalent of a printer whose hand is too steady, laying down ink in mechanical rows rather than natural variation.

Seeding the diffusion with a [blue-noise](https://en.wikipedia.org/wiki/Blue_noise) texture (generated via the void-and-cluster algorithm, Ulichney 1993) breaks those streaks up. Blue noise has energy concentrated at high spatial frequencies — it looks random at the scale of a few pixels while remaining evenly distributed overall, like the stipple of a skilled engraver rather than a mechanical grid. The result is dithering that reads as organic texture rather than digital artefact.

## Palette mixing (`--mix`): spectral Kubelka-Munk

### Physical basis

When a painter mixes two opaque paints, the result is not a simple average of their colours. Each pigment absorbs and scatters light differently at every wavelength; mixing them combines those physical processes, not their RGB numbers. The correct model for opaque paint mixtures was worked out by Paul Kubelka and Franz Munk in 1931. Their key insight: for a mixture of paints, the ratio K/S (absorption coefficient divided by scattering coefficient) at each wavelength combines *linearly* by weight, even though the resulting reflectance is a non-linear function of K/S.

In practice, the K/S ratio for a measured reflectance R at a single wavelength is:

> K/S = (1 − R)² / 2R

and to convert back: R = 1 / (1 + K/S + √(K/S² + 2·K/S)).

### Spectral representation

Rather than storing palette colours as single RGB triples, `--mix` fits each Nord colour with a 31-band reflectance spectrum (380–700 nm in 10 nm steps). Each band's reflectance is modelled as a clamped Gaussian:

> R(λ) = R_base + A · exp(−(λ − λ₀)² / 2σ²)

The four parameters (R_base, A, λ₀, σ) are fitted by minimising the CIE XYZ distance between the integrated spectrum and the palette colour under D65 illumination. Working in spectral K/S space means that any convex combination of palette K/S spectra corresponds to a physically realisable opaque paint mixture.

### Optimisation

For each pixel, the algorithm finds simplex weights c (cᵢ ≥ 0, Σcᵢ = 1) such that the K/S mixture ΣcᵢKSᵢ integrates to a colour as close as possible to the target in Oklab. This is solved with cosine-decayed [Adam](https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Adam) gradient descent, with the weights re-projected onto the probability simplex after every step.

### Initialisation and diversity

Good initialisation is important: Adam can stall in local optima if weights start too close to a palette-edge one-hot vector. The pipeline addresses this in two steps:

1. **Augmented palette snap** — the optimiser is initialised not just from the 17 pure palette colours but from an augmented set that includes all N(N−1)/2 pairwise 50/50 K/S mixtures. Pixels near colour-boundary regions are directly assigned interior-simplex starting weights rather than one-hot corners, avoiding the sharpest local optima from the start.

2. **Random diversification** — the snapped weights are blended 50/50 with a Dirichlet-sampled random weight field, then re-projected onto the simplex. This scatters the initialisation away from the nearest palette entry, giving Adam room to explore a wider region of the loss landscape.

The weights are then Gaussian-smoothed across image neighbours and re-projected, producing spatially coherent mixing in flat regions.
