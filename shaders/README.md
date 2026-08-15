# Shaders

GLSL ports of palettize.py's palette-mixing modes, plus a real-time GLSL port
of the wallpaper pipeline (`depth_blur.py` + `entropy_crop.py`) in
[`wallpaper/`](wallpaper/README.md).

## Files

- `fullscreen.vert` — trivial fullscreen-triangle vertex shader (no vertex buffer needed; draw 3 vertices). Shared by both mixing shaders below and by every "one fragment per output texel" pass in `wallpaper/`.
- `additive_mix.frag` — port of `palettize.py --mix additive` (`_face_newton_closest` / `_mix_strip_additive`). Fully self-contained and per-pixel: every fragment is resolved independently via Levenberg-Marquardt-damped Gauss-Newton on each palette convex-hull facet's own 2-parameter surface (linear-RGB light mixing), so a single pass over the image is enough.
- `export_hull_uniforms.py` — generates `additive_mix.frag`'s hull-geometry uniforms directly from `palettize.py`'s own `_halfspace_eqs` / `_face_geometry`, so the shader and the Python `--mix additive` path share one source of truth for the palette hull. Regenerate (and update `NUM_FACES` in the shader) whenever the palette changes.
  - `python3 export_hull_uniforms.py` — prints GLSL `const` array literals, if you'd rather bake the (fixed) hull into the shader source than set it at runtime.
  - `python3 export_hull_uniforms.py --upload` — prints a PyOpenGL upload snippet; `build_uniforms()` is also importable directly for other bindings (moderngl, etc.).
- `subtractive_mix.frag` — real-time approximation of `palettize.py --mix spectral` (default, Kubelka-Munk pigment mixing). Same Gauss-Newton-on-candidate-triangles technique as `additive_mix.frag`, but the triangles are palette K/S-spectrum sub-simplices (their Oklab-hull triangulation, not a linear-RGB one) and the chain rule carries one more nonlinear link (`_km_to_lin`'s closed-form derivative). See the shader's own header comment for why this is a heuristic approximation of `mix_convert_spectral`'s full 17-dimensional simplex optimiser, not a direct port. Fully self-contained, single pass.
- `export_km_uniforms.py` — the K/S-spectrum analogue of `export_hull_uniforms.py`, built from `palettize.py`'s own `_fit_palette_ks`.

A 4-pass "smoothed" variant of `subtractive_mix.frag` (`subtractive_snap.frag` → two separable-blur passes on the per-pixel simplex weight vector → `subtractive_resolve.frag`) existed briefly, meant to fix visible per-pixel colour jitter in smooth gradients by blurring the weight vector before reconstructing colour (mirroring `mix_convert_spectral`'s own snap→blur→reproject→evaluate structure). Removed: the blur radius needed to meaningfully smooth that jitter also visibly softened real image detail. Confirmed numerically before removing it that `subtractive_snap.frag`'s own per-pixel output, fed straight into `subtractive_resolve.frag` with no blur in between, is byte-identical (max diff ~1e-5, float noise) to `subtractive_mix.frag`'s single-pass output — `subtractive_resolve.frag`'s simplex reprojection is a no-op on weights already sitting on the simplex, so the whole 4-pass split had become a slower way to compute exactly what the single-pass shader already computes, once blur was gone. Retired rather than kept around for that.

## Uniform contract

`u_image` — the input texture, sRGB, straight (non-premultiplied) alpha ignored, `[0, 1]`. All three colour-pipeline matrix uniforms below (in both shaders) are set with `transpose=GL_FALSE` and data from `A.T.astype(np.float32)` (already transposed by both export scripts' `build_uniforms()`) — this sidesteps GLSL's column-major matrix layout entirely rather than relying on `glUniformMatrix3fv`'s `transpose=GL_TRUE`, which isn't available on GL ES / WebGL.

**`additive_mix.frag`**: `u_rgb2lms`, `u_lms2oklab` (`mat3`); `u_hullEqs`, `u_V0/U/Wv/L0/P/Q` — per-facet hull geometry, `NUM_FACES` entries each (24 for the current palette). From `export_hull_uniforms.build_uniforms()`.

**`subtractive_mix.frag`**: `u_rgb2lms`, `u_lms2oklab`, `u_xyz2rgb` (`mat3`); `u_paletteRgb`/`u_paletteOklab` (`vec3[N_PAL]`, `N_PAL=17`); `u_kmTriangles` and `u_d65nCmf` — **textures**, not uniform arrays (an `RGB32F` `(NUM_BANDS, NUM_FACES)` texture and an `RGB32F` `(NUM_BANDS, 1)` texture respectively). This is deliberate, not a style choice: a plain `uniform float u_ks0[NUM_FACES*NUM_BANDS]` (and its `p`/`q` siblings) blew real hardware's `GL_MAX_FRAGMENT_UNIFORM_COMPONENTS` (4096) during validation, because GLSL pads every element of a default-block array to a full `vec4` regardless of its own type — 1674 scalars actually cost 6696 components that way. Packing the same data as texels sidesteps the padding rule. From `export_km_uniforms.build_uniforms()`.

## Validated

**`additive_mix.frag`**: compiled and run for real (moderngl, GL 4.1 core via Metal) against `palettize.mix_convert_additive()` on a 300×400 crop of `samples/original.jpg`: mean abs BGR difference 0.48/255, 99.9% of pixels within ±2/255, worst case 25/255 (a handful of pixels near facet-selection ties / the near-black instability the Python safety-net also guards against). The remaining gap is ordinary float32 transcendental-function variance between MLX's kernels and the GLSL compiler's `pow`, not an algorithmic difference.

**`subtractive_mix.frag`**: since it's a deliberately different algorithm from `mix_convert_spectral` (see the shader's header), the meaningful check isn't matching Python's output but matching a from-scratch numpy re-implementation of *this shader's own* triangle-Gauss-Newton algorithm and analytic Jacobian (including the `_km_to_lin` derivative, `dR/d(K/S) = -R/S`, checked against central-difference numerics first). Compiled and run for real against that reference on 10 varied target colours (saturated primaries, greys, near-black, near-white): **exact match** (linear-RGB output identical to displayed float32 precision on every one). Rendered on the same photo crop as `additive_mix.frag`'s test: visually correct palette mixing, no artifacts; some per-pixel colour jitter is visible in smooth gradients (adjacent, near-identical pixels occasionally landing on different candidate triangles) — the now-removed smoothed variant traded that jitter for visible spatial blur, confirmed on a real photo through the web demo, and was reverted for it.

## Timing

All numbers: Apple M5 (moderngl standalone context, GL 4.1 core via Metal), `samples/original.jpg` resized to each resolution, best-of-3 wall-clock including the GPU sync forced by reading the result back.

**These are moderngl/Python-driver numbers, not the web demo's own real-world cost** — don't use this table to predict how the WebGL2 demo (`shaders/wallpaper/web/`) feels. Confirmed directly (not assumed) while investigating a reported slowness in `shaders/night/`'s pipeline: the *same* shaders, run through the actual browser/WebGL2 driver with a real `gl.finish()`-forced sync (not just CPU-side `performance.now()` bracketing around an async draw call, which measures submission time, not GPU execution time), cost far less than this moderngl table suggests — e.g. `subtractive_mix.frag` (single pass) measured ~2.6ms in-browser at 1.3MP, nowhere near this table's moderngl figure at a comparable size; night's own full pipeline measured ~110-250ms via moderngl at 1600×821 but ~9-15ms in-browser at the same size. The gap is moderngl's own per-draw-call Python/driver overhead (each of these pipelines issues dozens of small draw calls), which the browser's WebGL2 implementation evidently amortises far better — it isn't a GPU compute difference, since both drivers run the identical GLSL. Treat this table as relative guidance for how the *shaders themselves* scale with resolution and complexity, not as absolute numbers for the deployed demo.

| Shader | 0.12 MP (400×300) | 1.08 MP (1200×900) | 4.32 MP (2400×1800) |
|---|---|---|---|
| `additive_mix.frag` | 4.7 ms | 15.1 ms | 63.4 ms |
| `subtractive_mix.frag` | 14.7 ms | 147.8 ms | 1105 ms |

`additive_mix.frag` scales at roughly a constant ~15 ms/MP once past small-size fixed overhead. `subtractive_mix.frag` is far more expensive per pixel (~15-20× additive's) and scales worse than linearly — expected, since each of its up to 18 candidate triangles does a 31-band loop with several `texelFetch` calls per band per Gauss-Newton iteration, and texture fetches are considerably costlier than the plain ALU work `additive_mix.frag` does per facet. At 4K+ resolutions this is clearly a "run once, cache the result" shader (or worth restricting to real-time use at more modest resolutions, per the in-browser numbers above), not a per-frame one.

| Wallpaper pipeline | 0.02 MP (192×128) | 0.39 MP (768×512) | 1.57 MP (1536×1024) |
|---|---|---|---|
| `setup()` (once per image/target-size change) | 34.5 ms | 93.8 ms | 241.6 ms |
| `render()` (per-frame, e.g. on focal-depth change) | 2.0 ms | 7.2 ms | 27.9 ms |

`setup()`'s cost is dominated by a large *fixed* component (even the smallest test image takes 34 ms) — the crop search alone issues on the order of 400 small draw calls (per-candidate weighted-magnitude + reduction passes), and draw-call overhead, not GPU compute, dominates at these image sizes. `render()` scales cleanly with resolution and stays cheap throughout, confirming the setup/per-frame split does what it's meant to: an interactive focal-depth change never pays the crop-search or figure-identification cost again, only the composite pass.
