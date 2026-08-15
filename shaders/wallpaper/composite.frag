#version 330 core
// Pass 14 (per-frame): the only pass that needs to rerun when focal
// depth changes. Crops+resizes to u_targetSize using the offset from
// combine_and_argmin.frag, then composites the K adaptively-placed
// depth layers (sort_and_bounds.frag) front-to-back with premultiplied
// alpha -- ports depth_blur.py's _depth_blur loop body, generalised to
// non-uniform layer centres, with each layer's disc blur computed as a
// fixed-N_SAMPLES Vogel-disk *gather* (common.glsl) instead of an exact
// O(r^2) convolution -- see README.md's "real-time bokeh via sparse
// gather" for why this is mathematically the same operation, just
// subsampled for a bounded per-pixel cost.

out vec4 fragColor;

uniform sampler2D u_image;              // sRGB, [0,1]
uniform sampler2D u_depth;              // R32F raw disparity (higher = closer)
uniform sampler2D u_depthMinMax;        // 1x1 RG32F: (min, max)
uniform sampler2D u_layerCenters;       // K x 1, R32F, sorted ascending (sort_and_bounds.frag)
uniform sampler2D u_cropOffset;         // 1x1: (bestOffset, bestScore) (combine_and_argmin.frag)
uniform sampler2D u_figureMedianDepth;  // 1x1: (medianDepthNormalized, totalCount) (median_1d.frag)

uniform ivec2 u_imageSize;
uniform ivec2 u_targetSize;
uniform int   u_cropAxisIsX;
uniform int   u_newLength;
uniform int   u_k;

uniform float u_sigmaMax;         // max CoC radius, in *target* pixels (depth_blur.py's sigma_max)
uniform float u_focalDepth;       // normalised [0,1]; only read if u_focalDepthIsSet
uniform bool  u_focalDepthIsSet;  // false -> fall back to u_figureMedianDepth (the shader's (d))

const int MAX_K = 8;   // must be >= any u_k the host uses

// Maps a *target*-framebuffer pixel position (can be off-integer -- gather
// samples land at fractional offsets) through the crop+resize transform
// to a source-image UV. The perpendicular (uncropped) axis is a plain
// resize of its full extent; the crop axis maps the kept [offset,
// offset+newLength) window onto the target's corresponding dimension.
vec2 sourceUVAt(vec2 targetPixel) {
    float offset = texelFetch(u_cropOffset, ivec2(0, 0), 0).r;
    vec2 srcPixel;
    if (u_cropAxisIsX == 1) {
        srcPixel.x = offset + (targetPixel.x + 0.5) / float(u_targetSize.x) * float(u_newLength) - 0.5;
        srcPixel.y =          (targetPixel.y + 0.5) / float(u_targetSize.y) * float(u_imageSize.y) - 0.5;
    } else {
        srcPixel.y = offset + (targetPixel.y + 0.5) / float(u_targetSize.y) * float(u_newLength) - 0.5;
        srcPixel.x =          (targetPixel.x + 0.5) / float(u_targetSize.x) * float(u_imageSize.x) - 0.5;
    }
    return (srcPixel + 0.5) / vec2(u_imageSize);
}

float normDepthAt(vec2 uv) {
    float depth = texture(u_depth, uv).r;
    vec2  mm    = texelFetch(u_depthMinMax, ivec2(0, 0), 0).rg;
    return (mm.g - mm.r > 1e-6) ? (depth - mm.r) / (mm.g - mm.r) : 0.0;
}

// Generalisation of _depth_blur's tent basis (depth_blur.py:400) to
// non-uniform, sorted `centers` -- still an exact partition of unity
// (sum over i of tentWeight(d, i, ...) == 1 for any d within
// [centers[0], centers[k-1]], clamped flat beyond either end).
float tentWeight(float d, int i, float centers[MAX_K], int k) {
    float ci = centers[i];
    if (i == 0 && d <= ci) return 1.0;
    if (i == k - 1 && d >= ci) return 1.0;

    float lo = (i > 0)     ? centers[i - 1] : ci;
    float hi = (i < k - 1) ? centers[i + 1] : ci;
    float w = (d < ci)
        ? ((ci > lo) ? (d - lo) / (ci - lo) : 1.0)
        : ((hi > ci) ? (hi - d) / (hi - ci) : 1.0);
    return clamp(w, 0.0, 1.0);
}

void main() {
    vec2 targetPixel = gl_FragCoord.xy - 0.5;   // integer output pixel index, as a float

    // Per-pixel rotation of the (otherwise fixed) Vogel-disk sample
    // kernel -- see common.glsl's VOGEL_OFFSETS comment for why this is
    // necessary, not cosmetic: unrotated, N_SAMPLES=24's fixed sample
    // directions trace a visible polygon (hexagon-ish) at every pixel
    // identically, reading as a hard-edged bokeh shape instead of a
    // circle. Rotating the whole kernel by a different angle per pixel
    // keeps relative sample spacing (and so disc coverage) unchanged
    // while breaking that global coherence into much less objectionable
    // high-frequency noise.
    float rotAngle = hash21(gl_FragCoord.xy) * 2.0 * PI;
    float cosA = cos(rotAngle), sinA = sin(rotAngle);

    float centers[MAX_K];
    for (int i = 0; i < u_k; i++) centers[i] = texelFetch(u_layerCenters, ivec2(i, 0), 0).r;

    float dFocus = u_focalDepthIsSet
        ? u_focalDepth
        : texelFetch(u_figureMedianDepth, ivec2(0, 0), 0).r;
    float denom = max(dFocus, 1.0 - dFocus);   // telecentric CoC denominator, always >= 0.5

    vec3  colorAcc  = vec3(0.0);
    float weightAcc = 0.0;

    // Front-to-back: nearest layer (highest disparity/centre) first, so
    // it occludes farther ones -- centers[] is sorted ascending, so walk
    // it descending (mirrors depth_blur.py:392's
    // `order = np.argsort(centers)[::-1]`).
    for (int ii = 0; ii < u_k; ii++) {
        int i = u_k - 1 - ii;
        float rK = u_sigmaMax * abs(centers[i] - dFocus) / denom;

        vec3  colorK = vec3(0.0);
        float alphaK = 0.0;
        for (int s = 0; s < N_SAMPLES; s++) {
            vec2  baseOffset = VOGEL_OFFSETS[s];
            vec2  rotatedOffset = vec2(baseOffset.x * cosA - baseOffset.y * sinA,
                                        baseOffset.x * sinA + baseOffset.y * cosA);
            vec2  samplePixel = targetPixel + rotatedOffset * rK;
            vec2  uv = sourceUVAt(samplePixel);
            float d  = normDepthAt(uv);
            float m  = tentWeight(d, i, centers, u_k);
            vec3  c  = srgbToLinear(texture(u_image, uv).rgb);
            colorK += c * m;
            alphaK += m;
        }
        colorK /= float(N_SAMPLES);
        alphaK /= float(N_SAMPLES);

        // Premultiplied front-to-back composite (depth_blur.py:412-418):
        // each layer fills whatever the nearer layers left uncovered.
        float remaining = 1.0 - weightAcc;
        colorAcc  += remaining * colorK;
        weightAcc += remaining * alphaK;
        weightAcc  = clamp(weightAcc, 0.0, 1.0);
    }

    vec3 resultLin = colorAcc / max(weightAcc, 1e-6);
    fragColor = vec4(clamp(linearToSrgb(resultLin), 0.0, 1.0), 1.0);
}
