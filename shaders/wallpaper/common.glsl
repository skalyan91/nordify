// Shared functions/constants for the wallpaper-pipeline shader passes.
// GLSL core profile has no cross-file #include, so pipeline.py prepends
// this source verbatim before every pass's own fragment/vertex shader
// before compiling. Keep this file free of `main()` / pass-specific
// uniforms — only generic helpers belong here.

const float PI = 3.14159265359;

// --- colour pipeline (matches palettize.py's _srgb_to_linear / _linear_to_srgb) ---

vec3 srgbToLinear(vec3 c) {
    return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), step(0.04045, c));
}

vec3 linearToSrgb(vec3 c) {
    c = max(c, 0.0);
    return mix(12.92 * c, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(0.0031308, c));
}

float luminance(vec3 rgbLin) {
    return dot(rgbLin, vec3(0.2126, 0.7152, 0.0722));
}

// --- parafoveal acuity model (depth_blur.py:_foveal_weight_map) ---
//
// 1/(1 + e/E2): e = eccentricity in degrees from the *image's own centre*
// (the only defensible fixation point with no gaze data), assuming a
// viewer at `viewingDistanceFactor` x the image's diagonal. `p` and
// `size` must be in the same pixel units; the unknown physical pixel
// pitch cancels out of the eccentricity ratio.
float fovealWeight(vec2 p, vec2 size, float viewingDistanceFactor, float e2Degrees) {
    vec2  c        = (size - 1.0) * 0.5;
    float diagPx   = length(size);
    float viewDist = viewingDistanceFactor * diagPx;
    float rPx      = length(p - c);
    float eccDeg   = degrees(atan(rPx / viewDist));
    return 1.0 / (1.0 + eccDeg / e2Degrees);
}

// --- crop edge-risk model (entropy_crop.py:_edge_risk_kernel), pointwise ---
//
// e/(e+E2): the complement of fovealWeight's falloff — 0 at the crop
// window's own centre (losing it costs most), rising toward 1 at the
// window edges (already at the limit of clear vision, costs little to
// lose). `k` is a crop-axis position within the window, `newLength`/
// `other` the window's own two final dimensions (crop-axis / perpendicular).
float edgeRiskWeight(float k, float newLength, float other,
                     float viewingDistanceFactor, float e2Degrees) {
    float diagPx   = length(vec2(newLength, other));
    float viewDist = viewingDistanceFactor * diagPx;
    float half_    = newLength * 0.5;
    float eccPx    = abs(k - half_);
    float eccDeg   = degrees(atan(eccPx / viewDist));
    return eccDeg / (eccDeg + e2Degrees);
}

// --- Shannon entropy (entropy_crop.py:_entropy), in bits ---
//
// Reduction passes hand this the running (sum, sum of p*log2(p)) pair
// rather than a full profile array, so it's expressed as a finishing
// step: given `total` (sum of the unnormalised profile) and `plogp`
// (sum of profile[i] * log2(profile[i]), computed while the profile
// values are still unnormalised), entropy = log2(total) - plogp/total
// -- the standard identity for entropy of profile/total from moments
// of the unnormalised profile, avoiding a second full pass to normalise
// first. Returns 0 for a degenerate (all-zero) profile.
float entropyFromMoments(float total, float plogp) {
    if (total <= 1e-9) return 0.0;
    return log2(total) - plogp / total;
}

// x*log2(x), defined as 0 at x<=0 (matches Python's `p[p>0]` filtering —
// the x->0 limit of x*log2(x) is 0, not the log's own divergence).
float xlog2x(float x) {
    return x > 0.0 ? x * log2(x) : 0.0;
}

// --- Vogel-disk sample offsets (unit disc, radius <= 1) -------------------
//
// Real-time approximation of depth_blur.py's exact O(r^2) disc
// convolution (_make_disc_kernel / _disc_blur_mlx): convolution can be
// computed as a gather (output(p) = sum over kernel offsets of
// input(p+offset)), so a *fixed*, *sparse* set of offsets approximates
// the same operation at a bounded, radius-independent per-pixel cost --
// standard practice for real-time bokeh (a "Vogel disk": golden-angle
// spiral, near-uniform disc coverage, deterministic). Scale by a layer's
// own CoC radius at the call site.
//
// Used exactly as listed here (unrotated), N_SAMPLES=24 is too sparse to
// look like a smooth circle: connecting the 24 fixed directions traces
// out a visible polygon (an unrotated low-N Vogel/Fibonacci disk's outer
// ring is itself an approximately-regular polygon, and with no
// per-pixel variation the *same* polygon repeats identically at every
// pixel, so it reads as a hard, consistent hexagon-ish bokeh shape
// rather than pixel-to-pixel noise). The call site (composite.frag)
// rotates this whole pattern by a per-pixel angle from
// hash21() below before using it -- same sample
// directions relative to each other (so disc coverage stays uniform),
// different absolute orientation at every pixel, which turns the
// coherent polygon into much-less-objectionable high-frequency noise
// instead. Confirmed necessary, not just theoretical: unrotated, this
// exact 24-point kernel visibly reads as a hexagon in the composited
// bokeh; rotated, it doesn't.
const int N_SAMPLES = 24;
const vec2 VOGEL_OFFSETS[N_SAMPLES] = vec2[](
    vec2(0.144338, 0.000000),
    vec2(-0.184342, 0.168873),
    vec2(0.028217, -0.321513),
    vec2(0.232351, 0.303061),
    vec2(-0.426393, -0.075423),
    vec2(0.403917, -0.256939),
    vec2(-0.135102, 0.502574),
    vec2(-0.257655, -0.496099),
    vec2(0.559008, 0.204149),
    vec2(-0.581555, 0.240057),
    vec2(0.280348, -0.599087),
    vec2(0.207170, 0.660490),
    vec2(-0.624412, -0.361860),
    vec2(0.732507, -0.161040),
    vec2(-0.447038, 0.635865),
    vec2(-0.103276, -0.796974),
    vec2(0.634013, 0.534347),
    vec2(-0.853183, 0.035282),
    vec2(0.622332, -0.619303),
    vec2(-0.041636, 0.900426),
    vec2(-0.592151, -0.709594),
    vec2(0.938032, 0.126211),
    vec2(-0.794793, 0.552996),
    vec2(0.217183, -0.965401)
);

// Per-pixel hash in [0, 1), used by composite.frag to pick a rotation
// angle for VOGEL_OFFSETS (see that array's own comment above for why).
//
// Interleaved Gradient Noise (Jimenez 2014), the standard choice for
// dithering a real-time DOF kernel this same way, was tried first and
// reverted: confirmed on a synthetic point-light bokeh test (a bright
// point on a dark background, thrown far out of focus) that IGN produces
// a visible *directional* streak pattern across the disc, not clean
// noise -- expected once you look at its own construction (it's a
// linear gradient of a dot product with a fixed direction vector before
// the fract/hash step, hence "gradient" in the name), and fine in the
// AAA-engine real-time context it was designed for, where TAA
// accumulates many frames' worth of different camera jitter and
// averages the directional bias away. This pipeline renders one frame
// per focal-depth change, not an accumulated sequence, so that bias
// never gets averaged out -- it would just sit there as a visible static
// artefact. A hash with no directional structure (fract of a product of
// two already-fract'd, cross-mixed pixel-coordinate-derived terms, a
// standard cheap 2D hash) was confirmed on the same test to give
// isotropic speckle instead, with no streaking.
float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}
