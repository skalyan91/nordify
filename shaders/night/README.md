# Night-colouring shader

GLSL port of `palettize.py`'s `_nighttime` / `_dog_light_peaks` / `_light_protection_map`: darkens and cools an image for a night-time look, while identifying compact bright spots (lit windows, streetlights, the moon) via multi-scale Difference-of-Gaussians blob detection and protecting them from the effect (and boosting them slightly instead).

`NightPipeline` exposes two calls, split by what depends on `amount` and what doesn't:

- **`prepare(image, size)`** — everything independent of `amount`: oklab conversion and the (downsampled — see below) multi-scale DoG light-peak detection, ending in a full-resolution protection map. This is the expensive part.
- **`resolve(amount=1.0)`** — the cheap, single-pass darken/cool-or-brighten/warm transform, reading `prepare()`'s cached results. Measured ~0.1-0.3ms in isolation (see timing below) — cheap enough to call on every tick of a live slider without rerunning detection.
- **`run(image, size, amount=1.0)`** — `prepare()` + `resolve()` in one call, for one-shot callers that don't need to vary `amount` live (e.g. the Python validation driver, or a UI's initial render).

`amount` is one continuous, reversible parameter, `amount ∈ [-1, 1]`: `+1` is the original night effect, `0` leaves the image unchanged, and `-1` mirrors into an "extreme day" push — brightening and warming exactly where the night direction would have darkened and cooled most (the shadows). See `nighttime_resolve.frag`'s and `palettize.py`'s `_nighttime`'s own header comments for the exact power-law family (`L → 1-(1-L)^p(amount)`, `p=2^-amount`; `b`'s shift exponent `q=2^amount`, its reciprocal) that makes this one continuous formula rather than a crossfade between two effects. Light-peak protection itself is scaled by `max(amount, 0)` so it fades out approaching (and plays no role below) `amount=0` — there's nothing to protect a peak *from* on the brightening side.

## Files

- `common.glsl` — shared srgb/linear/Oklab conversion functions (including the Oklab→linear-RGB inverse, which none of the other shader families in this repo needed before now).
- `oklab_convert.frag` — sRGB → Oklab.
- `minmax_seed.frag` / `minmax_reduce.frag` — the Oklab `b` channel's min/max (needed for the cooling step's normalisation), copied from `wallpaper/`'s identical reduction pattern.
- `downsample2x.frag` — 2×2 box-filter downsample, run once before the detection stack (see "Downsampled detection" below).
- `gaussian_blur_h.frag` / `gaussian_blur_v.frag` — one reusable separable-blur pass pair, parametrised by `u_sigma`/`u_radius`/`u_normalize` at runtime rather than compiled per-sigma. Used both for the 6-level detection blur stack (normalised) and the 5 protection-bump blurs (unnormalised — see below).
- `dog_nms.frag` — one Difference-of-Gaussians level's peak detection (spatial 3×3 + cross-scale non-max suppression + threshold), run once per level (5 levels from 6 blur sigmas).
- `combine_max.frag` — combines the 5 per-level protection maps.
- `upsample_bilinear.frag` — bilinear-upsamples the (downsampled) protection map back to full resolution before `nighttime_resolve.frag` consumes it.
- `nighttime_resolve.frag` — the final darken/cool/protect colour transform; the only pass `resolve()` runs.
- `pipeline.py` — host driver (moderngl).

## Downsampled detection

Detection (every blur, DoG, and protection-map pass between `downsample2x.frag` and `upsample_bilinear.frag`) runs at half resolution, not the image's own. Profiling `run()` at 1600×821 found the 11 separable-Gaussian blur passes (radii up to ~90px at full resolution) dominating total cost, since separable-blur cost scales as `pixels × radius`. `prepare()` downsamples the oklab image by 2× once via a plain box filter, halves every one of the 11 blur sigmas to match (same *physical* detection scale — a light of a given real size still gets detected at the same scale, just computed over 1/4 the pixels with half the radius), and upsamples only the final protection map (a single small image, not the whole cascade) back to full resolution via bilinear filtering before `nighttime_resolve.frag` reads it. Measured a real (GPU-synced, not just CPU-dispatch-time) ~2.3× wall-clock reduction on the *whole* `run()` at 1600×821 (254ms → 110ms, moderngl/Metal) — less than the naive `2³=8×` a pure compute-cost argument would suggest, because a growing share of the remaining cost is fixed per-draw-call overhead (~24 passes total) that doesn't shrink with resolution; the blur passes' own compute dropped ~5.7× in isolation, closer to the theoretical figure.

The downsample factor (2, `DOWNSAMPLE` in `pipeline.py`/`night.js`) isn't exposed as a tunable: it was chosen and validated (both for speed and for peak-detection accuracy — confirmed no missed/shifted peaks on the same synthetic three-blob test used to validate the pipeline originally) as a safe default. A larger factor risks losing the smallest detection scale (`sigma=2.5`, already near the documented noise floor where sub-2px scales pick up specular glints instead of real lights) to the downsample blur entirely.

## Two deliberate simplifications (both documented in the shaders' own comments too)

1. **One generic blur pass, not eleven bespoke ones.** `gaussian_blur_h/v.frag` compute the Gaussian weight per tap on the fly from a runtime `u_sigma`, rather than the fixed literal-weight arrays `additive_mix.frag`/`wallpaper/luminance_blur_h.frag` use for their one fixed sigma — needed here since this pipeline uses **11 different sigmas** (6 detection scales + 5 protection-bump scales, `1.5×` each detection level's own scale) and hard-coding eleven shader variants would be needless duplication.
2. **Sum-then-clamp instead of max** for combining the 5 protection maps. `_light_protection_map` combines overlapping bumps via `maximum` so a light detected at more than one adjacent scale isn't over-protected. A 2-D Gaussian bump is separable, so *summing* bumps can reuse the same cheap separable blur pass (applied directly to each level's thresholded peak-amplitude field); *max* has no equivalently cheap separable form. The two agree exactly for isolated peaks (the common case) and only differ when several peaks' bumps overlap heavily, where sum-then-clamp saturates to the same `1.0` max would, just slightly earlier.

## Validated

Compiled and run for real (moderngl) on a synthetic scene (dark background, three compact bright blobs at known positions): detected peak positions matched the three synthetic lights exactly; full-image output matched `palettize._nighttime` almost exactly (mean abs BGR diff 0.002/255, max 1/255 — light centres and background pixels matched byte-for-byte). Visually confirmed on `samples/original.jpg`: sky and foliage darken and cool noticeably, the bright fruit stays warm. `run()` vs. `prepare()`+`resolve()` called separately produce byte-identical output (max diff 0.0), confirming the split is a pure refactor.

Re-validated end-to-end through the real web demo (Playwright, real photo, real browser) after both the downsample and the prepare()/resolve() split: no visual regression, and dragging `amount` at speed through the real UI (including with additive palette mixing layered on top, which reruns every tick since it depends on night's output) measured under 4ms per tick with zero WebGL warnings or errors.

One real bug was caught by that re-validation and is worth knowing about if you extend this: `night.js`'s `_minmaxB` helper (the log-step 2×2 reduction for the Oklab `b` channel's min/max) pushed *every* intermediate texture into its shared free-list, including the one that becomes its own final return value — harmless in the original single-call `run()` architecture, where that final value was consumed once and then the whole free-list (including it) was deleted at the very end of the same call anyway. Once `prepare()`/`resolve()` split that final value into **persistent** state (`this.bMinMax`, read by every subsequent `resolve()` call), the bug became real: `prepare()`'s own end-of-call cleanup deleted `this.bMinMax` the instant it was assigned, and every `resolve()` call after that bound an already-deleted WebGL texture (visible as repeated `bindTexture: attempt to use a deleted object` console warnings, one per `resolve()` call). `wallpaper.js`'s equivalent helper (`_reduce2dMinMax`) never had this bug — the fix (push the value about to be superseded, not the newly created one, matching that file) brought the two in line.

## Timing

800×533 (0.43 MP), Apple M5 via Metal, `prepare()`: ~15ms; `resolve()`: ~0.1-0.3ms. 1600×821, moderngl/Metal (GPU-synced): `run()` ~110ms total (down from ~254ms before downsampled detection); in the actual WebGL2 demo (browser driver overhead is substantially lower than moderngl's Python-level per-call overhead — confirmed directly, not assumed): `prepare()` ~9-15ms, `resolve()` ~0.2-0.3ms. Scales with resolution and with the largest blur radius (`3×24 = 72px` for the widest detection scale before downsampling, half that after) — like `subtractive_mix.frag`, `prepare()`'s cost is a "run once and cache" cost, not a per-frame one at large sizes, which is exactly why it's the piece kept out of the live `resolve()` path.
