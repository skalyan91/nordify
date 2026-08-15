#version 330 core

// Port of palettize.py's `--mix additive` (mix_convert_additive /
// _face_newton_closest): moves out-of-gamut pixels to the Oklab-nearest
// point on the selected palette's convex hull in linear RGB, via
// Levenberg-Marquardt-damped Gauss-Newton on each hull facet's own 2D
// (a, b) parametrisation. See palettize.py / CLAUDE.md for the derivation.
//
// This shader is fully self-contained and per-pixel — every input pixel
// is resolved independently, so a single fragment-shader pass is enough
// (no reduction/multi-pass machinery needed, unlike the wallpaper-crop
// shader).
//
// ---------------------------------------------------------------------
// Host contract — all facet/hull data below is generated on the CPU
// from the palette (see `export_hull_uniforms.py` next to this file,
// which reuses palettize.py's own `_halfspace_eqs` / `_face_geometry` so
// there is exactly one source of truth for the palette's hull geometry).
// NUM_FACES must match the array lengths that script prints — if the
// palette ever changes, regenerate and paste in the new arrays.
// ---------------------------------------------------------------------

in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D u_image;   // sRGB image, [0, 1], straight (non-premultiplied) alpha ignored

// RGB -> LMS -> Oklab pipeline matrices (Bjorn Ottosson's Oklab).
// Set from the host as mat3 uniforms — NOT hardcoded as GLSL literals,
// to sidestep GLSL's column-major matrix layout entirely: upload each
// as `A.T.astype(np.float32)` (numpy row-major -> GLSL column-major)
// with glUniformMatrix3fv(..., transpose=GL_FALSE, ...). See
// export_hull_uniforms.py for the exact upload call.
uniform mat3 u_rgb2lms;      // palettize._M_RGB_TO_LMS
uniform mat3 u_lms2oklab;    // palettize._M_LMS_TO_OKLAB

const int NUM_FACES = 24;    // must match export_hull_uniforms.py's output for the current palette

// Per-facet half-space equation (nx, ny, nz, d): dot(c, n) + d <= 0 inside the hull.
// Used only for the coarse in-hull test and the naive-clamp safety-net baseline.
uniform vec4 u_hullEqs[NUM_FACES];

// Per-facet triangle geometry. A point on facet f is
//   c(a, b) = V0[f] + a*U[f] + b*Wv[f]                      (linear RGB)
//   l(a, b) = L0[f] + a*P[f] + b*Q[f]  = u_rgb2lms * c(a,b)  (LMS, affine — precomputed)
uniform vec3 u_V0[NUM_FACES];
uniform vec3 u_U[NUM_FACES];
uniform vec3 u_Wv[NUM_FACES];
uniform vec3 u_L0[NUM_FACES];
uniform vec3 u_P[NUM_FACES];
uniform vec3 u_Q[NUM_FACES];

const float LM_LAMBDA = 1.0;     // Levenberg-Marquardt diagonal damping (validated fixed value)
const int   N_ITERS   = 10;      // Gauss-Newton iterations per facet (validated)
const float EPS_L     = 1e-6;    // LMS floor before cbrt (guards the near-black singular slope)
const int   N_PROJECT = 20;      // half-space POCS iterations for the naive-clamp baseline

// --- colour pipeline (mirrors palettize.py's _srgb_to_linear / _linear_to_srgb) -------------

vec3 srgbToLinear(vec3 c) {
    return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), step(0.04045, c));
}

vec3 linearToSrgb(vec3 c) {
    c = max(c, 0.0);
    return mix(12.92 * c, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(0.0031308, c));
}

vec3 rgbLinToOklab(vec3 rgbLin) {
    vec3 lms  = u_rgb2lms * rgbLin;
    vec3 lms_ = sign(lms) * pow(abs(lms), vec3(1.0 / 3.0));   // guarded cube root
    return u_lms2oklab * lms_;
}

// --- hull membership / naive clamp (mirrors _halfspace_eqs consumers in _mix_strip_additive) ---

bool inHull(vec3 c) {
    for (int i = 0; i < NUM_FACES; i++) {
        if (dot(c, u_hullEqs[i].xyz) + u_hullEqs[i].w > 1e-6) return false;
    }
    return true;
}

// POCS half-space projection — same algorithm as palettize.py's `_project`:
// each iteration steps back along the single most-violated constraint's
// normal, then a final hard [0,1] clamp. Only used as the safety-net
// baseline the Gauss-Newton result must never be worse than.
vec3 projectToHull(vec3 c) {
    for (int iter = 0; iter < N_PROJECT; iter++) {
        float worst = -1.0e30;
        int worstIdx = 0;
        for (int i = 0; i < NUM_FACES; i++) {
            float v = dot(c, u_hullEqs[i].xyz) + u_hullEqs[i].w;
            if (v > worst) { worst = v; worstIdx = i; }
        }
        if (worst > 0.0) c -= worst * u_hullEqs[worstIdx].xyz;
    }
    return clamp(c, 0.0, 1.0);
}

// --- barycentric simplex clamp (mirrors palettize.py's _simplex_clip_ab / _simplex_project_mlx,
//     specialised to N=3 since (a, b, 1-a-b) is always a 3-vector here) -----------------------

vec2 simplexClipAB(vec2 ab) {
    vec3 v = vec3(ab.x, ab.y, 1.0 - ab.x - ab.y);
    if (min(v.x, min(v.y, v.z)) >= 0.0) return ab;   // already feasible

    // Sort descending (3 elements — unrolled).
    vec3 u = v;
    if (u.x < u.y) { float t = u.x; u.x = u.y; u.y = t; }
    if (u.y < u.z) { float t = u.y; u.y = u.z; u.z = t; }
    if (u.x < u.y) { float t = u.x; u.x = u.y; u.y = t; }

    vec3 cs = vec3(u.x, u.x + u.y, u.x + u.y + u.z);     // cumulative sum
    int rho = 1;
    if (u.y - (cs.y - 1.0) / 2.0 > 0.0) rho = 2;
    if (u.z - (cs.z - 1.0) / 3.0 > 0.0) rho = 3;
    float theta = (rho == 1) ? (cs.x - 1.0) : (rho == 2) ? (cs.y - 1.0) / 2.0 : (cs.z - 1.0) / 3.0;

    v = max(v - theta, 0.0);
    return v.xy;
}

// --- per-facet Gauss-Newton search (mirrors palettize.py's _face_newton_closest) ----------------

vec3 faceNewtonClosest(vec3 targetLin, vec3 targetOk, out float bestDistOut) {
    vec3  bestColor = targetLin;
    float bestDist  = 1.0e30;

    for (int f = 0; f < NUM_FACES; f++) {
        vec3 V0 = u_V0[f], U = u_U[f], Wv = u_Wv[f];
        vec3 L0 = u_L0[f], P = u_P[f], Q = u_Q[f];

        // Flat initial guess: least-squares (a, b) in linear RGB, ignoring
        // the cube-root warp — closed form, cheap, a good Newton seed.
        float UU = dot(U, U), UV = dot(U, Wv), VV = dot(Wv, Wv);
        float det0 = UU * VV - UV * UV;
        vec3  diff = targetLin - V0;
        float Ud = dot(diff, U), Vd = dot(diff, Wv);
        float a = (VV * Ud - UV * Vd) / det0;
        float b = (UU * Vd - UV * Ud) / det0;
        vec2  ab = simplexClipAB(vec2(a, b));
        a = ab.x; b = ab.y;

        for (int it = 0; it < N_ITERS; it++) {
            vec3 l  = max(L0 + a * P + b * Q, EPS_L);
            vec3 w  = 1.0 / (3.0 * pow(l, vec3(2.0 / 3.0)));
            vec3 X  = u_lms2oklab * pow(l, vec3(1.0 / 3.0));
            vec3 Xa = u_lms2oklab * (w * P);
            vec3 Xb = u_lms2oklab * (w * Q);

            vec3 r = X - targetOk;
            float E = dot(Xa, Xa), F = dot(Xa, Xb), G = dot(Xb, Xb);
            float ga = dot(Xa, r), gb = dot(Xb, r);

            float Ed = E * (1.0 + LM_LAMBDA), Gd = G * (1.0 + LM_LAMBDA);   // LM diagonal damping
            float det = Ed * Gd - F * F;

            a -= (Gd * ga - F * gb) / det;
            b -= (Ed * gb - F * ga) / det;
            ab = simplexClipAB(vec2(a, b));
            a = ab.x; b = ab.y;
        }

        vec3  l     = max(L0 + a * P + b * Q, EPS_L);
        vec3  X     = u_lms2oklab * pow(l, vec3(1.0 / 3.0));
        vec3  color = V0 + a * U + b * Wv;
        vec3  rr    = X - targetOk;
        float d     = dot(rr, rr);

        if (d < bestDist) { bestDist = d; bestColor = color; }
    }

    bestDistOut = bestDist;
    return bestColor;
}

// --- main ---------------------------------------------------------------------------------

void main() {
    vec3 srgb = texture(u_image, v_uv).rgb;
    vec3 lin  = srgbToLinear(srgb);

    if (inHull(lin)) {
        fragColor = vec4(srgb, 1.0);   // in-gamut pixels pass through untouched
        return;
    }

    vec3 targetOk = rgbLinToOklab(lin);
    vec3 baseline = projectToHull(lin);

    float refinedDist;
    vec3  refined = faceNewtonClosest(lin, targetOk, refinedDist);

    vec3  baselineOk   = rgbLinToOklab(baseline);
    vec3  baselineDiff = baselineOk - targetOk;
    float baselineDist = dot(baselineDiff, baselineDiff);

    // Safety net: never let the facet search leave a pixel worse off than
    // the naive hull clamp (guards the same near-black instability the
    // Python implementation's safety net guards).
    vec3 outLin = (refinedDist > baselineDist) ? baseline : refined;

    fragColor = vec4(clamp(linearToSrgb(outLin), 0.0, 1.0), 1.0);
}
