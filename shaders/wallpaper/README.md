# Wallpaper-pipeline shaders

Real-time GLSL port of `depth_blur.py` + `entropy_crop.py`: given an image, a
depth map, a segmentation map, a target size, and an optional focal depth, it
(a) identifies "the figure" in the scene, (b) does figure-sensitive
entropy-based cropping to the target aspect, (c) simulates depth-of-field
bokeh, and (d) falls back to the figure's median depth when no focal depth is
given.

Unlike `../additive_mix.frag` (a single self-contained pass), this is a
**14-pass pipeline**, because figure identification, the crop search, and
median-of-figure depth are all *global reductions* — not something a single
fragment-shader invocation can do per-pixel. It's split into:

- **Setup** (`WallpaperPipeline.setup()`, passes 0–13) — rerun when the
  image/depth/segmentation/target-size changes. Not free (the crop search in
  particular issues ~400 small draw calls), but nothing here depends on focal
  depth.
- **Per-frame** (`WallpaperPipeline.render()`, pass 14) — the only pass that
  needs to rerun when focal depth changes, e.g. every frame if
  interactively focus-pulling.

Every value that crosses a pass boundary is a **GPU texture** — `setup()` and
`render()` never read a GPU result back to the CPU to feed a later pass, so
both are safe to call from a real-time loop.

## Segmentation-map contract (read this before wiring up real inputs)

`depth_blur.py`'s `_detect_figure_focus` runs SAM automatic mask generation,
then merges same-depth masks via union-find (`_ranges_overlap_50`) and splits
back into connected components (`cv2.connectedComponentsWithStats`) *before*
scoring candidates. None of that is fragment-shader-shaped work (an ML model
plus CPU graph algorithms), so it isn't reimplemented here.

**`u_segmentation` must already be that final per-pixel candidate-region
labelling**: a single-channel float texture where each texel holds an
integer region ID as a float (`0.0` = background/unassigned, `1.0..N` = one
already-merged-and-split candidate region each — i.e. everything through
`depth_blur.py:874`, `candidates.append(...)`, happens upstream). This
shader's job is exactly `depth_blur.py:891-897`: score each candidate by
foveal×depth-weighted mass, argmax, take its median depth — on GPU instead of
CPU/numpy.

## Usage

```python
import moderngl
from pipeline import WallpaperPipeline

ctx = moderngl.create_standalone_context()   # or moderngl.create_context() in a real GL app
pipe = WallpaperPipeline(ctx)

# image_tex: RGBA32F sRGB [0,1].  depth_tex, segmentation_tex: R32F, single channel.
pipe.setup(image_tex, depth_tex, segmentation_tex,
          image_size=(W, H), target_size=(out_W, out_H),
          ratio_w=16, ratio_h=9)          # target aspect, mirrors entropy_crop.py's args

frame = pipe.render(focal_depth=None)      # None -> falls back to the figure's median depth
frame = pipe.render(focal_depth=0.7)       # or drive it interactively every frame
```

Call `setup()` again whenever the image, depth map, segmentation map, or
target size changes. Call `render()` as often as you like in between —
it only re-touches pass 14.

## Pass pipeline

| # | File(s) | What |
|---|---|---|
| 0 | `minmax_seed.frag`, `minmax_reduce.frag` | Depth min/max (needed everywhere depth gets normalised) — log-step 2×2 reduction, seeded from the raw depth texture. |
| 1–2 | `segment_score.vert/.frag`, `argmax_1d.frag` | Figure identification: scatter foveal×depth weight per segment (`GL_POINTS` + additive blending, one point per source pixel), then argmax over the ≤`MAX_SEGMENTS` accumulator. Ports `depth_blur.py:851-891`. |
| 3–4 | `depth_hist_masked.vert/.frag`, `median_1d.frag` | Figure's median depth: masked 256-bin depth histogram, cumulative-sum to the 50th-percentile bin. Ports `depth_blur.py:882/893`'s `np.median`. |
| 5 | `luminance_blur_h/v.frag`, `gradient_mag.frag` | Sobel gradient magnitude of the image's luma (separable Gaussian pre-blur, `BORDER_REPLICATE`). Ports `entropy_crop.py`'s `_gradient_magnitude` exactly, including OpenCV's own BT.601 gray weights. |
| 6–7 | `figure_count_1d.vert/.frag`, `prefix_sum_1d.frag` | Figure-pixel count by crop-axis position, prefix-summed — feeds the O(1)-per-candidate figure-cut penalty below. |
| 8 | `weighted_mag_for_offset.frag`, `reduce_pairwise.frag`, `entropy_seed.frag`, `entropy_1d.frag` | Per candidate crop offset (`NUM_CANDIDATES`, host loop): risk-weighted gradient magnitude, reduced to a per-row profile, then to Shannon-entropy moments. Ports `entropy_crop.py`'s `_correlate_valid` + `_entropy`, computed by direct reduction per candidate rather than FFT (cheap since candidates are few and this only runs at setup time). |
| 9 | `combine_and_argmin.frag` | Combines each candidate's entropy with the figure-cut penalty (see below), argmins. Ports `entropy_crop.py:205`'s `np.lexsort` argmin/tiebreak. |
| 10–13 | `depth_hist_cropped.vert/.frag`, `gmm_init.frag`, `gmm_iterate.frag`, `sort_and_bounds.frag` | Adaptive depth layers: foveally-weighted depth histogram of the *cropped* region, a 1-D Gaussian Mixture Model (value-uniform init + EM iteration) fit to place `K` layer centres, sorted. |
| 14 | `composite.frag` | Per-frame: crop+resize to `u_targetSize`, then front-to-back premultiplied compositing of the `K` layers, each a Vogel-disk sparse-gather approximation of `_depth_blur`'s disc convolution. |

## Extensions beyond the Python pipeline

These three are *not* ports — `depth_blur.py`/`entropy_crop.py` don't do them
— added here for real-time use and per the user's request:

1. **Adaptive K depth layers (1-D Gaussian Mixture Model, via EM)** instead of
   `_depth_blur`'s fixed 16 uniform tent slabs. `K` (default 5, `em_iters`
   default 8) is a quality knob: fewer layers means fewer gather-samples per
   pixel in `composite.frag` (cost scales as `K × N_SAMPLES`).

   K-means was tried first and replaced: it assigns each histogram bin to
   its single nearest centroid and re-averages, which implicitly treats
   every cluster as the same "size" (an isotropic-equal-variance
   assumption) — a poor fit for depth histograms, which routinely mix a
   tight, compact foreground figure with a wide, smoothly-graded
   background in the *same* scene. A GMM instead fits each component's
   own variance and a soft, density-weighted responsibility per bin, so a
   tight spike and a diffuse spread can coexist without the wide one's
   scale distorting where the tight one's boundary should sit.

   Component means are seeded evenly **by value** across `[0, 1]`
   (`gmm_init.frag`), not by mass-quantile the way K-means's centroids
   were seeded. This isn't a stylistic choice: mass-quantile seeding was
   tried first and failed on exactly the histogram shape this pipeline
   sees constantly — a large, dominant, near-flat mode (sky, a wall) plus
   a much smaller figure and/or a gradient. Confirmed on a synthetic
   histogram shaped that way: quantile seeding placed most of `K`'s
   initial means already crowded on top of the dominant mode, and EM's
   soft responsibility never recovered — those components kept
   re-subdividing the same mode every iteration (each gets an outsized
   likelihood reward for precisely fitting a tall, tight peak) rather
   than covering the actual depth range, leaving the figure and any
   gradient to share whatever was left over (in the worst observed case,
   one badly-fit catch-all component covering both). Value-uniform
   seeding starts every component somewhere different regardless of
   where the mass concentrates, so EM has to earn each one's position
   rather than starting several stacked on the same easy local optimum —
   confirmed on the same histogram to converge to a spread closely
   matching (and, given each component's own fitted variance, more
   informative than) K-means's mass-proportional centroid placement, with
   no collapsed duplicates. See `gmm_init.frag`'s and `gmm_iterate.frag`'s
   own header comments for the full derivation.

   Not auto-selected for `K` itself — if a scene needs more or fewer
   components than `K`, EM either collapses extras onto the same real
   mode (harmless, just wasted layers — a repeated empty-component guard
   in `gmm_iterate.frag` keeps a starved component's previous parameters
   rather than letting it collapse to a degenerate zero-variance, zero-
   weight triple) or splits one real mode into overlapping sub-components
   (still harmless — `composite.frag`'s tent basis only ever consumes the
   sorted *means*, and is an exact partition of unity for any sorted
   `centers[]`, uniform or not).
2. **Figure-sensitive crop scoring.** `entropy_crop.py` is agnostic to scene
   content beyond raw gradients. `combine_and_argmin.frag` adds an explicit
   penalty: `figurePenaltyWeight * cutFraction * log2(other)` — `cutFraction`
   is the fraction of the figure's own pixels a candidate offset would crop
   away (O(1) via the prefix sum from pass 6–7), scaled by entropy's own max
   possible value (`log2(other)`) so the two terms are commensurate regardless
   of image size. `u_figurePenaltyWeight` defaults to **4.0** — large enough
   that avoiding the figure outright dominates ordinary entropy differences,
   while candidates that both avoid it fully (penalty 0) are still compared
   on entropy alone, same as the unmodified algorithm. This is a new,
   untested-in-the-wild heuristic (unlike the Python code's constants, which
   were each empirically confirmed against real photos) — treat it as a
   starting point to tune against your own content.
3. **Real-time bokeh via sparse Vogel-disk gather** instead of
   `_make_disc_kernel`'s exact O(r²) convolution. Convolution can be computed
   as a gather (`output(p) = Σ over kernel offsets of input(p+offset)`), so a
   fixed, sparse set of offsets (`N_SAMPLES=24`, `common.glsl`) approximates
   the same operation at a bounded, radius-independent per-pixel cost —
   standard practice for real-time depth of field. Because it's gather-based
   and `K` is now small, the entire layered composite collapses into
   `composite.frag`'s single pass (loop over `K` layers, inner loop over
   `N_SAMPLES`) — no per-layer blur+composite passes needed, unlike the
   Python version's necessarily-separate slabs.

   Used unrotated, `N_SAMPLES=24`'s fixed sample directions are too sparse
   to read as a smooth disc — the same 24-point pattern repeats identically
   at every pixel, so its polygonal outer ring (confirmed visually: an
   unrotated 24-point Vogel disk traces an approximate hexagon) shows up as
   a hard, consistent bokeh shape rather than a circle. `composite.frag`
   rotates the whole kernel by a per-pixel angle (`hash21()`, `common.glsl`)
   before use — same relative sample spacing (disc coverage unchanged), a
   different absolute orientation at every pixel, turning the coherent
   polygon into isotropic noise instead. A first attempt used Interleaved
   Gradient Noise (Jimenez 2014, the standard choice for this in real-time
   engines) and was reverted: confirmed on a synthetic point-light bokeh
   test that IGN's own linear-gradient construction produces a visible
   *directional* streak across the disc — fine when combined with TAA's
   temporal jitter accumulation (IGN's actual design target), useless here
   since this pipeline renders one frame per focal-depth change, not an
   accumulated sequence, so the directional bias never gets averaged away.
   A plain non-directional hash gives clean isotropic speckle instead. See
   `common.glsl`'s `VOGEL_OFFSETS`/`hash21` comments for the full
   before/after description.

## Tunable constants

| Constant | Default | Where | Trade-off |
|---|---|---|---|
| `MAX_SEGMENTS` | 64 | `segment_score.vert`, `argmax_1d.frag` | Upper bound on candidate figure regions. Raise if segmentation can produce more. |
| `NUM_CANDIDATES` | 32 | `pipeline.py`, `combine_and_argmin.frag` | Crop offsets sampled (evenly spaced across the excess range), not every integer offset — adjacent offsets differ by one row/column, a negligible perceptual difference. Raise for a finer search (setup-time cost only). |
| `K` (`k_layers`) | 5 | `pipeline.py`'s `setup()` arg | Depth layers. Lower = cheaper `composite.frag`, coarser bokeh depth resolution. |
| `em_iters` | 8 | `pipeline.py`'s `setup()` arg | GMM EM iterations. The value-uniform init (`gmm_init.frag`) usually converges in well under 8. |
| `u_figurePenaltyWeight` | 4.0 | `combine_and_argmin.frag` uniform | Higher = crop search avoids the figure more aggressively at the cost of otherwise-better (lower-entropy) crops. |
| `N_SAMPLES` | 24 | `common.glsl` | Gather samples per layer per pixel in `composite.frag`. Higher = smoother bokeh discs, linearly more expensive (`K × N_SAMPLES` texture reads per output pixel). |
| `u_sigmaMax` (`sigma_max`) | caller-provided | `pipeline.py`'s `render()` arg | Maximum circle-of-confusion radius, in *target* pixels (mirrors `depth_blur.py`'s `sigma_max`). |

## Validated

Full pixel-exact end-to-end validation (like `additive_mix.frag`'s) isn't
practical here — it would require actually running SAM to produce a real
segmentation map, a heavy dependency unrelated to what's being tested.
Instead, each piece carrying real algorithmic risk was compiled and run for
real (moderngl, GL 4.1 core via Metal) against a small Python/numpy reference
computed the same way, on a synthetic scene (a near "figure" disc, a farther
"clutter" disc placed to tempt a naive crop, textured background):

- **Figure argmax + median depth**: GPU picks the same segment as
  `depth_blur.py:851-897`'s formulas computed directly in numpy; median depth
  matches to within the 256-bin histogram's own quantisation (~0.0015 off
  a numpy `np.median` on this test case).
- **Crop search**: entropy values matched a numpy port of
  `entropy_crop.py`'s exact `_correlate_valid`/`_entropy` to 4–5 significant
  figures at every one of the 32 sampled candidates (`total` differs by a
  constant scale factor from operating on `[0,1]`-range pixels rather than
  Python's `[0,255]` — harmless, since it's scale-invariant for the entropy
  term and only ever compared against other GPU-computed totals for the
  tiebreak). The figure-sensitive penalty correctly shifted the chosen
  offset off plain-entropy's pick (which cut through the clutter disc) onto
  one that fully avoids it — and matched a direct exhaustive numpy search
  over *all* offsets (not just the 32 sampled), not merely the subsampled
  candidate set.
- **GMM layers**: compared directly against a plain K-means reference on a
  synthetic histogram shaped like a real scene (a dominant flat background
  mode, a small tight figure spike, a diffuse gradient) — K-means spread its
  5 centroids across background/gradient/figure sensibly; an initial
  mass-quantile-seeded GMM instead collapsed 3-4 of 5 components onto the
  background mode alone, confirming the seeding change described above was
  necessary, not cosmetic. With value-uniform seeding, the GMM converges to
  a comparable, non-degenerate spread (5 distinct means, no collapsed
  duplicates) on the same histogram, and to sensible visually-correct
  results end-to-end on a real photo through the actual web demo.
- **Composite**: rendered at multiple focal depths — visually correct
  depth-of-field (sharp figure/blurred background and vice versa, clean
  bokeh falloff at the figure's edge, no artifacts).
- **Bokeh disc shape**: a synthetic point-light-on-dark-background test
  (a small bright patch, thrown far out of focus at a large `sigma_max`)
  is the standard way to expose a sparse-gather DOF kernel's own shape —
  the unrotated `VOGEL_OFFSETS` kernel visibly traced a hexagon on this
  test; rotating it per pixel (see "Extensions" above) produces a clean
  circular outline instead, confirmed both in isolation and end-to-end
  on a real photo (`samples/callide_demo_image.png`) through the actual
  web demo, with visibly round, grainy — not polygonal — highlights in
  the out-of-focus background.

One real bug was caught this way and is worth knowing about if you extend
this: `pipeline.py`'s uniform-texture-unit allocator must reset its counter
at the start of *every* draw call, not just once — leaving it cumulative
across the crop search's ~400 draw calls silently exhausted the GPU's
texture units partway through and zeroed out every later candidate's score,
which only showed up once the loop ran for real (every single-call test of
each pass in isolation passed fine).
