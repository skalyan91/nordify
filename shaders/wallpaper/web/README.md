# Wallpaper Ricer (Wallricer) — web demo

A browser UI for `shaders/wallpaper/`'s pipeline: upload an image and,
optionally, a depth map and a segmentation map, then interactively
explore the cropped + figure-sensitive + depth-blurred result, with
`shaders/night/` and `palettize.js`'s palette-mixing shaders available as
optional colour filters. Palette mixing targets any scheme in
`palettize.PALETTES` (Nord, Solarized, Gruvbox, Everforest, Catppuccin,
Dracula, ...), selected from a dropdown under "Palette mixing" — see
"Palette selection" below for how that's precomputed.

Two independent things are computed, not one linear filter chain:
**geometry** (crop offset, figure identification, the figure's own median
depth as the default focal plane, and the GMM depth layers) is
decided once from the *original* uploaded image, depth map, and
segmentation map — before night colouring or palette mixing ever run, so a
colour filter can never shift where the frame crops or what counts as
the figure. **Colour** (night, then palette mixing, in that order) is
applied to that same original image to produce whatever the final
composite actually draws. See "Filter chain..." below for why this split
exists.

**Depth map and segmentation map are optional.** They're needed
together (`wallpaper.js`'s geometry stage takes both or neither) — upload
both to get cropping and depth-guided focus/blur; upload only an image
and the "Focus" section disables itself, cropping is skipped, and the
image (after any colour filters) is shown at its native resolution.

## Running it

**Needs a local static server — it will not work opened directly via
`file://`.** ES module `import`s (used throughout this demo's JS, for
the same reasons any modern JS codebase uses them) are blocked by CORS
when loaded from `file://` in every major browser; this is a browser
platform restriction, not something fixable from the page's own code.
One line, from this directory:

```bash
python3 -m http.server 8000
# then open http://localhost:8000/index.html
```

(Any other static file server — `npx serve`, the VS Code Live Server
extension, etc. — works equally well.)

Requires WebGL2 with the `EXT_color_buffer_float` extension (needed to
render to the floating-point textures every pass in these pipelines
uses) — both are standard on any browser from the last several years;
the page reports a clear error on load if either is missing rather than
failing silently.

## Upload contract

- **Image**: any browser-decodable image (PNG/JPEG/WebP).
- **Depth map**: grayscale (or RGB — only the R channel is read), pixel
  value / 255 = normalised disparity, **higher = closer** — the same
  convention `depth_blur.py --save-depth` writes.
- **Segmentation map**: grayscale (or RGB — only R is read), pixel value
  = integer candidate-region ID, `0` = background/unassigned — exactly
  `shaders/wallpaper/README.md`'s `u_segmentation` contract (the final,
  already-merged-and-split per-pixel labelling `_detect_figure_focus`
  would produce up through `candidates.append`, not raw SAM masks).

All three must be the same pixel dimensions; the page checks this after
upload and reports a clear error if they don't match. Depth map and
segmentation map are themselves optional — either upload both yourself,
or check "Figure out depth & focus automatically" to have the browser
compute them instead (see "In-browser depth & segmentation" below).

## In-browser depth & segmentation (optional, downloads models)

Checking "Figure out depth & focus automatically" (in the Upload
section) runs `inference.js`, which estimates depth and segments the
photo into candidate figure regions **entirely client-side**, via
[transformers.js](https://huggingface.co/docs/transformers.js) — no
backend, and the photo never leaves the browser. It replaces manual
depth/segmentation file uploads (the two file inputs disable themselves
while this is checked) with two real neural-network models running as
WASM (or WebGPU, if the browser supports it):

- **Depth**: Depth Anything V2 Small (`onnx-community/depth-anything-v2-small`,
  quantized) — the same model `depth_blur.py` uses by default, run
  through the *same* tiled multi-pass refinement `depth_blur.py`'s own
  `_estimate_depth` uses (`_tile_starts`/`_feather_weights`/`_tile_refine`,
  ported to `inference.js` as plain JS — one global pass plus one
  independent tiled pass per `tile_width_frac` in `[0.5, 0.25, 0.125]`,
  each least-squares aligned to the global pass and combined by per-pixel
  maximum), not a single whole-image pass — see that function's own
  docstring for why: Depth Anything resizes any input down to 518px
  internally regardless of source resolution, so a single pass alone
  loses thin structures (wires, masts, lattice towers). This is
  thorough but slow — confirmed in testing at ~5 minutes on a
  1600×821 photo (43 tile inferences plus the global pass, all
  sequential, on CPU/WASM) — the UI's own disclosure text says so
  upfront rather than leaving the checkbox looking hung.
- **Segmentation**: DETR ResNet-50 panoptic segmentation
  (`Xenova/detr-resnet-50-panoptic`, quantized), via transformers.js's
  `image-segmentation` pipeline. This is a deliberately simpler
  substitute for `depth_blur.py`'s own SAM-based automatic mask
  generation + depth-guided region merging — that process (~1000
  candidate masks, NMS, a union-find merge against the depth map) is both
  too slow for interactive in-browser use and depends on Python-side
  logic with no ready browser equivalent. Panoptic segmentation is a
  fair swap for what the segmentation map actually gets used for
  downstream: `wallpaper.js`'s `u_segmentation` contract only needs a
  per-pixel region ID per candidate object (0 = background) so its own
  GPU-side scoring pass can rank whichever is most prominent — it
  doesn't care how the regions were produced, only that each is
  spatially contiguous. `inference.js` rasterises each returned instance
  mask into one combined ID map, ranked by the model's own confidence
  score and capped at `wallpaper.js`'s `MAX_SEGMENTS` (64).

**This downloads real model weights, not just a few KB of JS** — quantized,
that's about 27MB (depth) + 44MB (segmentation) ≈ 72MB, confirmed via the
actual Hugging Face file sizes, plus a few more MB for the transformers.js
runtime itself and the WASM backend it loads. The page's own UI states
this upfront (a fixed disclosure next to the checkbox, not something you
only discover after triggering it) rather than assuming everyone reads
this file first. The browser's own HTTP cache means this is a one-time
cost per browser, not per page load. Both models load in parallel and
report live download progress through the same `#status` element
everything else in this demo uses.

## Palette selection (precomputed)

The palette dropdown (`#paletteSelect`, under "Palette mixing" in the
Filters section) picks which named scheme in `palettize.PALETTES` the
mixing shaders target. GLSL array sizes are compile-time constants, and
different palettes have different convex-hull facet counts (confirmed:
`NUM_FACES` ranges 14-29, `N_PAL` ranges 12-17 across this repo's current
schemes) — so each palette still needs its own compiled program
(`palettize.js`'s `AdditivePipeline`/`SubtractivePipeline`, substituting
the palette's own counts into the canonical `additive_mix.frag`/
`subtractive_mix.frag` source text before compiling).

The hull/K-S geometry every palette's shader needs is computed **offline**,
once, by `build_palette_data.py`: it calls `export_hull_uniforms
.build_uniforms(name)` / `export_km_uniforms.build_uniforms(name)` — the
*exact* Python functions the CLI and moderngl validation paths use — for
every scheme in `palettize.PALETTES`, and writes the result to a static
`palette_data.js`. The page's `warmup()` then compiles one additive and
one subtractive program *per palette* from that precomputed data (and
exercises each with a throwaway draw — see "GL drivers defer real
compile" below), all up front at page load; `onPaletteChange()` afterwards
is just picking which already-compiled program to use, with no async
work and no shader recompile on the critical path.

This replaces an earlier version of this demo that computed palette
geometry **live in the browser** via [Pyodide](https://pyodide.org/)
(real CPython + numpy compiled to WASM, running the same `build_uniforms()`
functions unmodified against `palettize.py`/`export_hull_uniforms.py`/
`export_km_uniforms.py` fetched as plain text): that let any palette added
to `palettize.PALETTES` work immediately with no web-specific build step,
at the cost of a multi-second Pyodide+numpy CDN fetch on every page load
and a fresh compute (and shader recompile) on every palette switch. Once
the palette *set* settled into a fixed, known list, that cost stopped
buying anything — `build_palette_data.py` keeps the same single source of
truth (`build_uniforms()`, unchanged) but runs it once, offline, with the
real CPython/numpy already installed for the CLI, rather than paying for
it live in every visitor's browser. Regenerate `palette_data.js` with
`python3 build_palette_data.py` whenever `palettize.PALETTES` changes.

Depth estimation and segmentation are, and always were, **not** run
client-side by this demo regardless of the above — those are PyTorch
models (Depth Anything V2, SAM), for which there's no equivalent
"precompute offline, ship static data" trick — so depth/segmentation stay
user-supplied (optional) uploads, per the contract above.

`cv2` (opencv-python) is imported lazily inside the handful of
`palettize.py` functions that actually need it (blur steps, image I/O),
not at module level — a leftover of the Pyodide era (Pyodide has no
opencv-python build) that's harmless to keep regardless of how the web
demo gets its geometry.

Switching palettes (`onPaletteChange()`) swaps both mixing pipelines'
active program (not just whichever mode is currently selected), then
reruns `rebuildColor()` — visually instant, since every palette's
programs were already compiled and exercised during `warmup()`.

## Geometry vs. colour, and why almost everything ends up live

`wallpaper.setup()` (`rebuildGeometry()` in `demo.js`) is given the
**original**, unfiltered source image, and decides crop offset, figure
identification, the figure's median depth, and the GMM depth layers
from it. It's deliberately never given a night/palette-mixing-filtered
image: those filters change what a pixel looks like, not where the
figure is or how the frame should crop, and letting a filter run first
would make crop/focus decisions depend on which filters happen to be
enabled right now — palette mixing in particular measurably changes the
image's own gradient magnitude, which is exactly what the entropy-based
crop search scores candidates on. `wallpaper.render()` takes its colour
source as an explicit parameter on every call rather than one fixed at
setup() time, so night/palette-mixing filters never need to touch
geometry at all.

That split makes three different things live simultaneously, each for
its own reason:

- **Focal depth** re-runs only `wallpaper.render()` (sub-millisecond to a
  few ms at the resolutions tested — see `shaders/wallpaper/README.md`'s
  timing table) — unaffected either way, since it never depended on
  colour filters even before this split.
- **Night colouring's `amount` slider** (`-1` extreme day … `0`
  unchanged … `+1` full night, see `shaders/night/README.md`) goes
  through `onNightAmountInput()`, not `rebuildColor()`: it calls
  `night.resolve(amount)` directly, reusing whatever `rebuildColor()`
  last `prepare()`d, then reruns palette mixing on top if it's enabled (real GPU
  cost for either mode measured at low single-digit ms — see
  `shaders/README.md`'s timing section — cheap enough to rerun on every
  tick too, since palette mixing's input changes whenever night's output does).
  `night.js`'s own `prepare()`/`resolve()` split (see
  `shaders/night/README.md`) is what makes this possible: `prepare()` is
  the expensive multi-scale DoG detection stage (tens of ms), a property
  of the image alone; `resolve()` is a single cheap pass (~0.2-0.3ms
  measured) that only reads `amount`. Skipping `prepare()` on every tick
  is the actual point of this fast path — see the note below on why an
  earlier version of this same slider *looked* live without actually
  being so.
- **Toggling night/palette mixing on or off, or changing mixing mode**
  reruns the full `rebuildColor()` (night's `prepare()`+`resolve()`, then
  palette mixing — a few ms total at demo resolutions) but *never*
  `rebuildGeometry()` — crop/figure/focus stay exactly as they were,
  since colour was never part of that decision.

**A measurement pitfall worth knowing about if you extend this:** an
earlier version of this demo called `night.js`'s single-call `run()` on
every `amount` tick and reported it as already "live" based on
`performance.now()` bracketing around that call — but WebGL draw calls
are asynchronous by default (`gl.drawArrays` enqueues work and returns
immediately; it doesn't block until the GPU finishes), so that
measurement only captured CPU-side command *submission* time, not actual
GPU execution time. Forcing a real sync (`gl.finish()`, or — as
`page.screenshot()` does in this repo's own Playwright tests, which is
how this was originally caught — anything that requires the result)
revealed `run()`'s true GPU cost at demo resolutions was ~100ms+, not the
sub-millisecond figure the CPU-side timing suggested: dragging the
slider at that ordering would have been genuinely janky in practice
(GPU falling behind), invisible to a timing method that never actually
waited for the GPU. The `prepare()`/`resolve()` split above, plus timing
every real change from here on with a forced sync, are both direct
responses to that finding.

(Two earlier versions of this demo got progressively closer to this: the
first applied night/palette-mixing *after* the wallpaper composite and
debounced them, since they had to rerun on every focal-depth change
under that ordering. The second moved them *before* the crop/composite
but still ran full geometry decisions on the *filtered* image, meaning a
filter toggle still had to rerun crop search and GMM every time — and
left no way for a colour filter's own parameter, like night's `amount`,
to be live at all. Splitting geometry and colour into two genuinely
independent stages, as described above, removes both remaining costs.)

## Downloading the shaders

"Download shaders (.zip)" (Export section) packages up exactly the GLSL
`.frag`/`.vert` files the demo is actually using **right now** — read
fresh off the current UI state at click time, so toggling a filter after
loading changes what a later export contains too — not the whole
`shaders/` directory regardless of what's in use:

- Palette mixing (if enabled): only the active mode's shader
  (`additive_mix.frag` or `subtractive_mix.frag`), pre-compiled for the
  selected palette (`NUM_FACES`/`N_PAL` substituted via `palettize.js`'s
  own `withConstInts()` — GLSL array sizes are compile-time constants, so
  this file isn't reusable for a different palette without recompiling),
  alongside a `..._geometry.json` with that palette's own numeric hull/
  K-S data (straight from `palette_data.js`) — everything needed to
  actually drive the shader, since the raw `.frag` file alone declares
  `uniform`s but has no values for them.
- Night colouring (if enabled): the full `shaders/night/` pass set.
- Crop/focus (if a depth map + segmentation map are actually driving the
  current render): the full `shaders/wallpaper/` pass set.

Every exported file is already fully self-contained — `build_shaders.py`
inlines `common.glsl` and the `#version`/precision header into each
shader string at build time, so nothing in the zip needs further
assembly before it'll compile. A `README.txt` in the zip explains what's
included and points at the driver code (`pipeline.py` / `wallpaper.js` /
`night.js` / `palettize.js`) needed to actually run the passes. Built
client-side via [fflate](https://github.com/101arrowz/fflate) (`zipSync`,
loaded from a CDN on demand, in `shader_export.js`) — no server involved.
If nothing is currently active (nothing loaded, no filters on), the
button reports that instead of downloading a near-empty zip.

## GPU resource cleanup

Every pass in these pipelines renders into a fresh floating-point
texture + FBO — `wallpaper.js`'s `setup()` alone allocates several dozen
per call. None of that is free on the GPU, so `wallpaper.js`, `night.js`,
`palettize.js`, and `demo.js` all explicitly `gl.deleteTexture`/
`gl.deleteFramebuffer` their *transient* intermediates (via `gl-utils.js`'s
`deleteTexAndFbo`/`deleteTexsAndFbo`) before returning, and free the
*previous* generation's persistent state (`wallpaper.js`'s
`depthMinMax`/`figureInfo`/`figureMedianDepth`/`cropOffset`/
`layerCenters`, each `render()` call's output, and `demo.js`'s
filter-chain output) exactly one call after it stops being referenced —
this was found and fixed after toggling a filter repeatedly showed
multi-second stalls that grew with each toggle, despite each pipeline's
own reported timing staying in the tens-of-ms range; profiling the GPU
driver's own behaviour (not JS timing) pointed at accumulating
un-freed GPU objects, confirmed by the stalls disappearing entirely once
cleanup was added (repeated toggling on a 1600×821 test image stayed at
~0.1s per toggle across many rounds, vs. 4.5-4.8s and growing before the
fix).

## Architecture

- `build_shaders.py` — generates `shaders.js` (embedded GLSL ES 300
  source strings) from the canonical `.frag`/`.vert`/`.glsl` files under
  `shaders/`, `shaders/wallpaper/`, and `shaders/night/`. Regenerate
  whenever those sources change.
- `build_palette_data.py` — generates `palette_data.js` (precomputed
  hull/K/S-triangle geometry for every scheme in `palettize.PALETTES`)
  from `export_hull_uniforms.py`'s / `export_km_uniforms.py`'s own
  `build_uniforms(name)` — see "Palette selection" above. Regenerate
  whenever `palettize.PALETTES` changes.
- `gl-utils.js` — small WebGL2 helper layer, the JS analogue of
  moderngl's convenience API used throughout the Python drivers (texture
  creation, FBOs incl. MRT, per-draw texture-unit-safe uniform binding —
  see its own comments for the texture-unit bug this specifically
  guards against, hit for real while building the Python side).
- `wallpaper.js`, `night.js`, `palettize.js` — mechanical ports of
  `shaders/wallpaper/pipeline.py`, `shaders/night/pipeline.py`, and
  `additive_mix.frag`'s / `subtractive_mix.frag`'s (both single-pass)
  usage, respectively.
- `inference.js` — optional client-side depth estimation + segmentation
  via transformers.js, loaded from a CDN on demand (never on page load) —
  see "In-browser depth & segmentation" above.
- `shader_export.js` — builds the "Download shaders" zip from whatever's
  currently active, via fflate (also loaded from a CDN on demand) — see
  "Downloading the shaders" above.
- `demo.js` / `index.html` — the UI.
- `test_harness.html` — a minimal page (no UI) that exposes the
  pipeline classes on `window` for scripted testing (Playwright etc.),
  used to validate this port against the Python/moderngl reference.

## Validated

Compiled and ran every pass for real in headless Chromium (Playwright)
against the Python/moderngl reference on identical synthetic inputs:
figure identification, crop offset, and GMM layer centres matched
closely (small differences are expected and correct — this demo
uploads depth as an 8-bit texture per the contract above, while the
Python-side reference test used full float precision; the crop offset
decision itself, a discrete choice, matched exactly). Night and both
mixing filters were run against the same photo crop used to validate
their Python/moderngl counterparts and produced visually matching
output. End-to-end through the actual page with real file uploads:
focal-depth dragging shows correct sharp/blurred depth-of-field
behaviour, click-to-focus correctly samples the clicked point's own
depth, and the night→palette-mixing→wallpaper chain renders correctly.

Also run against a real photo at demo scale — `samples/callide_demo_image.png`
(1600×821), its depth map, and a real 27-region SAM segmentation map
(`samples/callide_demo_segmentation.png`, generated via `depth_blur.py
--save-segmentation`) — not just the synthetic test scenes above: initial
load+geometry ~54ms (well under the earlier ~2.6s figure from before the
GPU-leak fix and the SAM-segmentation-map generation being the dominant
initial cost, not this pipeline), each filter toggle a few ms, and both
10 in-page focal-depth ticks and 7 in-page night-`amount` ticks measured
directly (not through Playwright's own per-call IPC overhead, which
dominates if you time slider drags from the test driver instead — e.g.
`page.fill()` + `dispatch_event()` reads several seconds per tick purely
from Playwright's own interaction cost, vs. sub-millisecond when driven
in-page) at well under 1ms each — confirming both sliders really are
cheap enough to be fully live at a realistic resolution, not just on the
small synthetic inputs used for the numerical cross-checks above.
Visually confirmed on the same photo: `amount=+1` darkens/cools with the
power-station's own lit windows and the moon staying bright and crisp,
`amount=-1` brightens/warms past the original into a washed-out pastel
daylight look with no light-protection artifacts (protection correctly
fades to zero below `amount=0`), and `amount=0` reproduces the
unfiltered original.

**Palette selection**: end-to-end through the real page (Playwright, real
Chromium): page load compiles and exercises all 12 (6 palettes × 2 modes)
precompiled programs and reaches "Ready" in well under a second — no CDN
fetch on the critical path any more; switching the dropdown to
`gruvbox-dark` *before* any image is loaded is instant and error-free
(no async work, nothing to render yet); loading the real Callide photo
and enabling palette mixing renders correctly in both additive and
subtractive mode; switching to `catppuccin-mocha` with an image already
loaded and mixing enabled re-renders live with no shader-compile stutter
(every palette's programs were already exercised during `warmup()`), and
no console errors or warnings beyond the usual benign `GL_CLOSE_PATH_NV`
driver notice. Visually confirmed the palettes tested (Nord, Gruvbox
Dark, Catppuccin Mocha) produce clearly distinct, correctly-tinted output
on the same photo — not a shared or stale geometry silently reused
across the switch. A real timing bug was caught this way during the move
to precompiled programs: `warmup()`'s throwaway draw originally only
exercised the *default* palette's programs, so a GL driver's "real
compile deferred to first draw" cost (see below) still landed live the
first time a user switched to any *other* palette; fixed by looping
`warmup()`'s dummy-texture draw over every precompiled palette, not just
the default one.

**Optional depth/segmentation**: loading only an image (no depth/seg
files chosen) correctly skips `wallpaper.setup()`/`render()` entirely,
disables the Focus fieldset and the crop-related Upload controls
(confirmed via each control's own `.disabled` state), renders the
image at its native, uncropped resolution with night/palette-mixing
filters still applying live, and the Download button still works.
Loading image + depth + segmentation afterward re-enables those controls
and produces the same cropped/focus-blurred output as before this
change — confirmed as a direct regression check against the same
Callide photo/depth/segmentation triple used elsewhere in this file.

Two other real bugs were caught this way and are worth knowing about if you
extend this. `wallpaper.js`'s `sampleDepthAt` (used by click-to-focus)
read back an FBO wrapper object without unwrapping its `.fbo` field, and
separately read a texture back as `FLOAT` unconditionally even when it
had been uploaded as 8-bit — both silent-until-run mistakes that only
showed up once actually exercised through the real page, not from
reading the code. Separately, none of `wallpaper.js`/`night.js`/
`palettize.js`/`demo.js` originally freed any of the textures/FBOs their
multi-pass pipelines allocate — every `setup()`/`run()`/`render()` call
leaked its intermediates, which didn't affect single-pass correctness
checks at all but caused toggling a filter repeatedly to get
progressively slower (multi-second stalls after a handful of toggles)
purely from GPU memory/driver pressure, invisible to each pipeline's own
JS-side timing. See "GPU resource cleanup" above.
