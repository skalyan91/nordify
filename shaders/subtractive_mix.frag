#version 330 core

// Real-time approximation of palettize.py's `--mix spectral` (default,
// Kubelka-Munk pigment mixing): moves out-of-gamut pixels toward the
// Oklab-nearest reachable paint mixture, via Levenberg-Marquardt-damped
// Gauss-Newton on triangulated palette K/S-spectrum candidates — the
// same technique as additive_mix.frag, extended through one more link
// in the chain rule for the Kubelka-Munk nonlinearity.
//
// --------------------------------------------------------------------
// WHY THIS IS AN APPROXIMATION, NOT A DIRECT PORT (read before relying
// on close fidelity to `--mix spectral`'s output):
//
// mix_convert_spectral optimises over the *full* 17-dimensional palette
// simplex (Sigma c_i = 1, c >= 0) with ~300-600 Adam steps per pixel,
// each step evaluating a 31-band Kubelka-Munk mixture. That's not
// shader-shaped work at real-time cost. Instead, this shader restricts
// the search to *triangles* — 3-colour sub-simplices — exactly as
// additive_mix.frag restricts its search to hull facets rather than the
// whole polytope interior. The palette's genuine achievable-reflectance
// gamut is a curved manifold (Kubelka-Munk mixing does not follow
// straight lines in Oklab space), so unlike additive_mix.frag's hull
// facets — which *are* the exact gamut boundary in linear RGB — the
// candidate triangles here (the convex hull of the palette's own Oklab
// positions, computed by export_km_uniforms.py) are a *heuristic*
// candidate set, not an exact facet decomposition of the true gamut.
// It should closely approximate the boundary most out-of-gamut pixels
// actually snap to (mix_convert_spectral's own augmented-palette
// initialisation already observes that boundary pixels mostly land on
// nearby 2-3-colour mixtures), but won't reach every mixture the full
// simplex optimiser could in principle find.
// --------------------------------------------------------------------

in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D u_image;   // sRGB, [0, 1]

uniform mat3 u_rgb2lms;      // palettize._M_RGB_TO_LMS, transposed for GLSL (see export script)
uniform mat3 u_lms2oklab;    // palettize._M_LMS_TO_OKLAB, transposed
uniform mat3 u_xyz2rgb;      // palettize._M_XYZ_TO_RGB, transposed

const int NUM_BANDS = 31;    // CIE 400-700nm @ 10nm, matches palettize._CIE_CMF / _D65

// D65-normalised CMF, (NUM_BANDS, 1) RGB32F texture: texel(band, 0).rgb
// = (x-bar, y-bar, z-bar) * D65n at that band. A plain `uniform float[]`
// this size is fine on its own, but see u_kmTriangles below for why
// textures, not arrays, hold every per-band table in this shader.
uniform sampler2D u_d65nCmf;

const int NUM_FACES = 18;    // must match export_km_uniforms.py's output for the current palette

// Per-triangle K/S-spectrum geometry, packed as a (NUM_BANDS, NUM_FACES)
// RGB32F texture: texel(band, f) = (ks0, p, q) at that band for facet f.
// A point on triangle f is
//   ks(a, b) = ks0[f] + a*p[f] + b*q[f]                     (31-band K/S)
//
// This (and u_d65nCmf above) uses a texture rather than a plain
// `uniform float[...]` array deliberately: GLSL pads every element of a
// default-block array to a full vec4 (16 bytes) regardless of its own
// type, so NUM_FACES*NUM_BANDS*3 scalars (1674 here) would actually cost
// 4x that in uniform *components* -- confirmed hitting real hardware's
// GL_MAX_FRAGMENT_UNIFORM_COMPONENTS (4096) this way during validation.
// Packing the same data as RGB texels sidesteps the padding rule
// entirely and scales far beyond what plain uniform arrays allow.
uniform sampler2D u_kmTriangles;

// Pure palette colours (linear RGB + their Oklab position), for the
// nearest-single-colour safety-net fallback below -- the curved KM
// gamut has no cheap half-space membership test the way additive
// mixing's linear-RGB hull does, so this stands in for that shader's
// "naive hull clamp" baseline.
const int N_PAL = 17;   // must match export_km_uniforms.py's output for the current palette
uniform vec3 u_paletteRgb[N_PAL];
uniform vec3 u_paletteOklab[N_PAL];

const float LM_LAMBDA = 1.0;   // Levenberg-Marquardt diagonal damping (see additive_mix.frag)
const int   N_ITERS   = 10;
const float EPS_L     = 1e-6;

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
    vec3 lms_ = sign(lms) * pow(abs(lms), vec3(1.0 / 3.0));
    return u_lms2oklab * lms_;
}

vec2 simplexClipAB(vec2 ab) {
    vec3 v = vec3(ab.x, ab.y, 1.0 - ab.x - ab.y);
    if (min(v.x, min(v.y, v.z)) >= 0.0) return ab;
    vec3 u = v;
    if (u.x < u.y) { float t = u.x; u.x = u.y; u.y = t; }
    if (u.y < u.z) { float t = u.y; u.y = u.z; u.z = t; }
    if (u.x < u.y) { float t = u.x; u.x = u.y; u.y = t; }
    vec3 cs = vec3(u.x, u.x + u.y, u.x + u.y + u.z);
    int rho = 1;
    if (u.y - (cs.y - 1.0) / 2.0 > 0.0) rho = 2;
    if (u.z - (cs.z - 1.0) / 3.0 > 0.0) rho = 3;
    float theta = (rho == 1) ? (cs.x - 1.0) : (rho == 2) ? (cs.y - 1.0) / 2.0 : (cs.z - 1.0) / 3.0;
    v = max(v - theta, 0.0);
    return v.xy;
}

// --- Kubelka-Munk: K/S ratio -> linear reflectance (palettize._km_to_lin), + its derivative ---
//
// R = 1/(1 + ks + S), S = sqrt(ks^2 + 2ks + eps). Differentiating:
// dS/dks = (ks+1)/S, so dR/dks = -(1 + dS/dks)/(1+ks+S)^2
//        = -((S+ks+1)/S) / (1+ks+S)^2 = -1/(S*(1+ks+S)) = -R/S
// -- a clean closed form (verified against central-difference numerics
// to 1e-9 before use here), no need to carry S and (1+ks+S) separately.

float kmToLin(float ks) {
    ks = max(ks, 0.0);
    float s = sqrt(ks * ks + 2.0 * ks + 1e-12);
    return clamp(1.0 / (1.0 + ks + s), 0.0, 1.0);
}

// --- per-triangle forward model + analytic Jacobian ----------------------------------------
//
// Ports mix_convert_spectral's `_forward` (palettize.py:520): K/S mix ->
// Kubelka-Munk reflectance (pointwise, 31 bands) -> CIE XYZ (linear) ->
// linear RGB (linear, clamped to [0,1] same as the Python reference) ->
// Oklab. The clamp's zero gradient outside (0,1) is respected in the
// Jacobian (clampMask below) exactly as autodiff would handle it.
vec3 evalTriangle(int f, float a, float b, out vec3 rgbOut, out vec3 Xa, out vec3 Xb) {
    vec3 XYZ = vec3(0.0);
    vec3 dXYZda = vec3(0.0);
    vec3 dXYZdb = vec3(0.0);

    for (int band = 0; band < NUM_BANDS; band++) {
        vec3 tri = texelFetch(u_kmTriangles, ivec2(band, f), 0).rgb;   // (ks0, p, q)
        float ks0 = tri.r, p = tri.g, q = tri.b;
        float ksRaw = ks0 + a * p + b * q;
        float ks = max(ksRaw, 0.0);
        float s  = sqrt(ks * ks + 2.0 * ks + 1e-12);
        float Rv = clamp(1.0 / (1.0 + ks + s), 0.0, 1.0);
        float dRdks = -Rv / s;

        vec3 cmf = texelFetch(u_d65nCmf, ivec2(band, 0), 0).rgb;
        XYZ    += Rv * cmf;
        dXYZda += dRdks * p * cmf;
        dXYZdb += dRdks * q * cmf;
    }

    vec3 rgbRaw = u_xyz2rgb * XYZ;
    rgbOut = clamp(rgbRaw, 0.0, 1.0);
    vec3 clampMask = vec3(
        (rgbRaw.x > 0.0 && rgbRaw.x < 1.0) ? 1.0 : 0.0,
        (rgbRaw.y > 0.0 && rgbRaw.y < 1.0) ? 1.0 : 0.0,
        (rgbRaw.z > 0.0 && rgbRaw.z < 1.0) ? 1.0 : 0.0);
    vec3 drgbda = clampMask * (u_xyz2rgb * dXYZda);
    vec3 drgbdb = clampMask * (u_xyz2rgb * dXYZdb);

    vec3 lms = max(u_rgb2lms * rgbOut, EPS_L);
    vec3 w   = 1.0 / (3.0 * pow(lms, vec3(2.0 / 3.0)));
    vec3 X   = u_lms2oklab * pow(lms, vec3(1.0 / 3.0));

    Xa = u_lms2oklab * (w * (u_rgb2lms * drgbda));
    Xb = u_lms2oklab * (w * (u_rgb2lms * drgbdb));
    return X;
}

// --- per-triangle Gauss-Newton search (mirrors additive_mix.frag's faceNewtonClosest) ------

vec3 triangleNewtonClosest(vec3 targetOk, out float bestDistOut) {
    vec3  bestColor = vec3(0.0);
    float bestDist  = 1.0e30;

    for (int f = 0; f < NUM_FACES; f++) {
        // Centroid start: unlike additive_mix.frag, K/S-mixing isn't
        // linear in the pixel's own linear-RGB colour, so there's no
        // equally cheap closed-form "flat" seed here -- LM damping
        // (see additive_mix.frag's derivation) makes Gauss-Newton
        // robust to a generic starting point regardless.
        float a = 1.0 / 3.0, b = 1.0 / 3.0;

        for (int it = 0; it < N_ITERS; it++) {
            vec3 rgbTmp, Xa, Xb;
            vec3 X = evalTriangle(f, a, b, rgbTmp, Xa, Xb);
            vec3 r = X - targetOk;

            float E = dot(Xa, Xa), F = dot(Xa, Xb), G = dot(Xb, Xb);
            float ga = dot(Xa, r), gb = dot(Xb, r);
            float Ed = E * (1.0 + LM_LAMBDA), Gd = G * (1.0 + LM_LAMBDA);
            float det = Ed * Gd - F * F;

            a -= (Gd * ga - F * gb) / det;
            b -= (Ed * gb - F * ga) / det;
            vec2 ab = simplexClipAB(vec2(a, b));
            a = ab.x; b = ab.y;
        }

        vec3 rgbFinal, XaFinal, XbFinal;
        vec3 X = evalTriangle(f, a, b, rgbFinal, XaFinal, XbFinal);
        vec3 rr = X - targetOk;
        float d = dot(rr, rr);
        if (d < bestDist) { bestDist = d; bestColor = rgbFinal; }
    }

    bestDistOut = bestDist;
    return bestColor;
}

void main() {
    vec3 srgb = texture(u_image, v_uv).rgb;
    vec3 lin  = srgbToLinear(srgb);
    vec3 targetOk = rgbLinToOklab(lin);

    float refinedDist;
    vec3  refined = triangleNewtonClosest(targetOk, refinedDist);

    // Safety net: nearest single pure palette colour, exactly what a
    // no-mixing Oklab snap (palettize.py's `convert()`) would produce.
    // The curved KM gamut has no cheap membership test the way additive
    // mixing's linear-RGB hull does (see the module docs above), so this
    // stands in for that shader's "naive hull clamp" baseline -- the
    // triangle search should only ever improve on it.
    float baselineDist = 1.0e30;
    vec3  baselineColor = u_paletteRgb[0];
    for (int i = 0; i < N_PAL; i++) {
        vec3 d = u_paletteOklab[i] - targetOk;
        float dist = dot(d, d);
        if (dist < baselineDist) { baselineDist = dist; baselineColor = u_paletteRgb[i]; }
    }

    vec3 outLin = (refinedDist > baselineDist) ? baselineColor : refined;

    fragColor = vec4(clamp(linearToSrgb(outLin), 0.0, 1.0), 1.0);
}
