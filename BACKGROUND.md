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

## Palette mixing (`--mix`): thinking like an artist

The `--mix` mode is based on a paint-mixing model: any colour achievable by blending Nord palette entries (like pigments on a palette) lies inside the *convex hull* of those entries in linear RGB space. A pixel outside the hull cannot be reproduced by any mixture, so it must be approximated by the nearest in-hull colour.

Finding that nearest colour is a constrained optimisation problem, and the solution mirrors how an artist builds up a painting from scratch:

### Phase 1 — value (lightness)

A painter's first act is usually a *value sketch* or grisaille underpainting: getting the light/dark relationships right before committing to colour. A painting that reads well in greyscale will read well in colour. In nordify, Phase 1 minimises the difference in lightness L between the remapped colour and the original, while keeping the colour on the hull boundary. See: [value (art)](https://en.wikipedia.org/wiki/Lightness).

### Phase 2 — hue

With the values established, the painter lays in the colour temperature and hue relationships — the warm versus cool areas, the colour bias of shadows. Phase 2 minimises the angular difference in hue (the cross-product of the hue direction vectors in Oklab) while penalising any drift from the lightness achieved in Phase 1.

### Phase 3 — chroma (saturation)

Finally, the painter adjusts the intensity of each colour — pushing it more saturated in the lights, graying it down in the shadows. Phase 3 minimises the chroma difference while holding both the hue and lightness achieved in the earlier phases.

Each phase runs [Adam gradient descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Adam) until convergence, with the colour projected back onto the hull boundary after every step (via iterative half-space projection). The three-phase decomposition ensures that the most perceptually important attribute — value — is locked in before hue and chroma are adjusted, just as it is in traditional painting practice.
