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

All colour comparisons and optimisations in palettize.py run in Oklab.

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

## Palette mixing (`--mix additive`): linear-light convex hull

### Physical basis

Kubelka-Munk mixing models paint: pigments that absorb and scatter light, mixed by physically stirring them together. But not every medium works that way. Point two coloured stage lights at the same patch of wall and their illuminance simply adds — the physics of *additive* colour mixing, the same process at work when a screen's red, green, and blue subpixels blend into a full-colour image, or when overlapping torch beams wash out to white. In linear RGB — light intensity before the sRGB gamma curve is applied — a convex combination of colours corresponds exactly to this kind of physically realisable light mixture. So `--mix additive` builds its gamut as the convex hull of the 17 Nord colours (plus black and white, to extend the reachable lightness range) directly in linear RGB, rather than in Kubelka-Munk's spectral K/S space.

This gamut is a much more generous one than the pigment gamut of `--mix spectral`: mixing light rarely darkens or desaturates the way mixing paint does, so a far greater share of an ordinary photograph already falls inside it. In practice this means `--mix additive` tends to leave more of an image untouched and shifts the remainder more subtly than the spectral model — visible mainly on strongly saturated or overexposed pixels.

### Gamut as half-spaces

A convex hull in three dimensions can be represented as the intersection of half-spaces — one inequality nx·x + ny·y + nz·z + d ≤ 0 per face, satisfied by every point inside the hull. `_halfspace_eqs` builds this representation by brute force: every triple of palette points defines a candidate plane, kept only if every other point lies on one side of it. With 19 points (17 colours plus black and white) this is a few thousand triples — cheap to compute once and reuse for every pixel.

Projecting an arbitrary colour onto the hull is then an iterative process: find the most-violated face, push the point back onto it, and repeat. Twenty iterations is enough to converge for a shape this simple, and the whole loop runs as ordinary MLX tensor arithmetic — no per-pixel branching, no host round-trips.

### Optimisation: two-phase, gamut-clamped

Simply projecting each out-of-gamut pixel onto its nearest point on the hull would work, but "nearest" in linear RGB is not perceptually meaningful — Euclidean distance there does not track how different two colours *look*. Instead, `--mix additive` first projects onto the hull (a reasonable starting point, and a no-op for pixels already inside), then spends its optimisation budget walking that starting point back toward the original colour's appearance in Oklab, split into two phases:

1. **Luminance** — minimise (L − L_target)², matching the target's lightness.
2. **Chrominance** — minimise (a − a_target)² + (b − b_target)² (hue and chroma matched jointly, since both live in the same (a, b) plane), with a 1000× penalty on drifting away from the luminance already achieved in phase 1.

An earlier attempt split chrominance further, into a hue phase and then a chroma phase run sequentially — mirroring the phase structure historically used for Kubelka-Munk mixing above. It converges to a similar-looking result, but far more slowly: separating hue from chroma means each phase's loss carries its own 1000× penalty term guarding the phase(s) before it, and that layered ill-conditioning needs many more optimiser steps to settle. Matching (a, b) jointly sidesteps the problem, since hue and chroma are just polar coordinates of the same two numbers — there is no reason to solve for them one at a time.

Each phase runs [Adam](https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Adam), whose per-coordinate adaptivity is what makes the penalty term tractable at all — plain gradient descent, tried first, stalled badly on phase 2's ill-conditioned loss. But an unconstrained Adam step could easily propose a colour outside the gamut again. So after every step, the candidate colour is re-projected onto the hull via the same half-space projection used for initialisation — meaning the *effective* step taken is not the raw Adam update but (projected candidate − current colour): whatever part of the update would have left the gamut is silently clipped away, and optimisation continues from wherever that leaves the pixel. This is the same principle as projected gradient descent in convex optimisation: take an ordinary step, then project back onto the feasible set, repeat.

## Automatic figure detection: a parafoveal acuity model

### The problem with hand-picked criteria

`depth_blur.py --focus auto` has to pick a focus plane without being told what the subject of a photo is. An early version did this by segmenting the image and scoring each candidate region on five hand-picked criteria — size, a compactness measure, convexity, spatial connectedness, depth — combined by weights tuned reactively against whichever test image had just broken. That is a genuinely *underdetermined* procedure: nothing but trial and error against a couple of photos justified any particular weight, and there was no way to know whether it would generalise.

The fix was to stop scoring proxies for "looks like a subject" and instead model something concrete and measurable: how sharply a human viewer would actually resolve each part of the image. "The figure" becomes whichever candidate region has the greatest total *resolvable* area under that model — one principled quantity instead of five arbitrary ones.

### Viewing geometry, without knowing the screen

Modelling visual acuity first requires knowing how far a point on the image sits from where the viewer is looking, in degrees of visual angle rather than pixels. That requires an assumed viewing distance — but the pipeline has no idea what physical display the output will end up on, or at what resolution.

The way out: define the viewing distance as a multiple of the image's *own* diagonal (1.5×, a typical comfortable viewing ratio for a display filling much of the field of view) rather than as an absolute physical distance. A pixel `r` pixels from the frame's centre — the assumed fixation point, the only defensible default with no gaze data — then subtends a visual angle of

> eccentricity = arctan(r / (1.5 · diagonal_px))

Both `r` and the diagonal are measured in the same pixel units, so the unknown physical size of a real pixel cancels out of the ratio completely. The formula needs nothing about the eventual display beyond the assumption that the image will be viewed comfortably, filling a consistent fraction of the visual field regardless of its actual size.

### Cortical magnification

Visual acuity is sharpest at the point of fixation and falls off with eccentricity — not linearly, and not with a hard cutoff at some "foveal radius," but smoothly, in a way well characterised by vision science. The cortical magnification factor — roughly, how much retinal/cortical area is devoted to processing a given patch of visual field — follows an inverse relationship with eccentricity first described by [Rovamo & Virsu (1979)](https://doi.org/10.1007/BF00236627), and the same falloff underlies gaze-contingent computer graphics (e.g. Guenter et al., [*Foveated 3D Graphics*](https://doi.org/10.1145/2366145.2366183), 2012):

> acuity(e) = 1 / (1 + e / E2)

where `E2` is the eccentricity at which acuity has fallen to half its foveal value — taken here as ≈2.3°, a standard value in that literature. At `e = 0` this is 1; it decays smoothly and asymptotically thereafter, never reaching exactly zero. There is no free radius parameter to hand-pick — every quantity in the model (viewing distance ratio, `E2`) has an independent, citable meaning.

### Combining with depth

Acuity alone can't tell a large, cleanly-segmented but *distant* region (a patch of sky) from an equally large, equally central *near* one — eccentricity says nothing about depth. Nearness is folded into the same per-pixel weight as a second multiplicative factor: normalised disparity, 0 at the scene's own farthest point and 1 at its nearest. "The figure" is then the segmented region (candidates from [SAM](https://github.com/facebookresearch/segment-anything)'s automatic mask generation, chosen over classical edge detection because it segments from the image's own visual boundaries rather than the depth map's often-incomplete inferred ones) with the greatest total `acuity × nearness`-weighted area.

## Minimum-entropy cropping, reusing the same model

`entropy_crop.py` needs a related but inverted quantity: not "how well would a viewer see this," but "how much would be lost if this position were cut off" by a candidate crop boundary. That is the complement of acuity in the same model — `e / (e + E2)`, 0 at a candidate crop window's own centre and rising toward 1 at its edge — replacing an earlier, purely geometric parabola of the same shape but with no arbitrary exponent to have picked.

One consequence of the change: a parabola is a polynomial in position, so a weighted sum over every candidate crop offset could be computed in closed form from a handful of prefix sums. The acuity-derived weight involves an arctangent and a division, so that shortcut no longer applies — the weighted sum for every offset is instead computed as a single batched FFT correlation, restoring the same practical speed for an arbitrary (non-polynomial) weight shape.

## Detecting lights at night: scale-space blob detection

### What actually makes something read as "a light"

`palettize.py --night` darkens and cools an image to suggest nighttime, but a real light source — a lit window, a streetlamp, the moon — should stay bright rather than dim along with everything else. The natural question, "is this pixel bright?", turns out to be the wrong one: a white shirt in daylight is bright, but so is everything around it, and it isn't "a light." What actually makes something read as a light source is standing out sharply from its *immediate surroundings* — local contrast, not any absolute brightness or colour value. That signal isn't visible to a rule operating on one pixel's own colour at a time; it requires looking at a neighbourhood.

### Difference-of-Gaussians as a blob detector

Subtracting two different-width Gaussian blurs of the same image — a Difference-of-Gaussians, or DoG — approximates the Laplacian of a Gaussian and responds strongly at blob-like features whose size matches the scale spanned by the two blur radii: at the centre of a bright spot on a dark background, a narrower blur still mostly reflects the spot's own bright value, while a wider blur has averaged in more of the dark surround and reads lower, so the difference is large and positive there. This is the same core mechanism behind [SIFT](https://en.wikipedia.org/wiki/Scale-invariant_feature_transform) keypoint detection — a Gaussian scale-space pyramid, with feature points found as local extrema of the DoG response.

Running this at several scales at once (rather than picking one fixed blur radius) lets both a small, sharp streetlight and a larger, softer glow register as blobs, each at whichever scale matches its own size — a local maximum in the DoG response, both across neighbouring pixels and across neighbouring scales, marks a detected light. Requiring a genuine local maximum — not merely an elevated DoG value — is what excludes plain edges and gradients, which have elevated response over an extended region but no compact spatial peak.

This was chosen over a segmentation-based alternative (e.g. running SAM and comparing each proposed mask's brightness against a surrounding ring) for being far simpler while capturing exactly the same underlying signal: no ML model, no depth map, and no extra dependency — local contrast in the lightness channel alone is already the thing that defines a light.
