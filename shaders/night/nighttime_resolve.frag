#version 330 core
// Pass 6: final colour transform. Generalizes _nighttime (palettize.py:1088)
// to a single continuous u_amount in [-1, 1]: +1 reproduces the original
// night effect exactly (darken L -> 1-sqrt(1-L), cool b toward blue in
// inverse proportion to L, both suppressed at protected light-peak
// locations and replaced there by L + light_boost*protect); 0 is the
// source image unchanged; -1 is the algebraic mirror -- an "extreme day"
// push (brighten L -> 1-(1-L)^2, warm b toward yellow, strongest exactly
// where the night version's darkening/cooling was strongest, i.e. the
// shadows) with no light-peak protection, since there's nothing to
// protect a peak *from* on the brightening side.
//
// Both L and b transforms are one member of the same power-law family
// at every amount, not just at +-1 and 0: the L curve is
// 1-(1-L)^p(amount) with p(amount) = 2^(-amount) (p=0.5 at +1, matching
// 1-sqrt(1-L) exactly; p=1 at 0, identity; p=2 at -1, the mirror), and
// the b curve's exponent is its reciprocal q(amount) = 2^amount (q=2 at
// +1, matching bNorm*(L + bNorm*(1-L)) exactly, since that's
// L*bNorm + (1-L)*bNorm^2; q=1 at 0, identity; q=0.5 at -1). This keeps
// the whole slider continuous and reversible through the same formula,
// not a crossfade between two different effects.

in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D u_oklab;      // (L, a, b) from oklab_convert.frag
uniform sampler2D u_protect;    // [0,1] protection map from combine_max.frag
uniform sampler2D u_bMinMax;    // 1x1 RG32F: (min, max) of the b channel
uniform float u_lightBoost;     // palettize.py default 0.2
uniform float u_amount;         // -1 (extreme day) .. 0 (unchanged) .. +1 (full night)

void main() {
    vec3 lab = texture(u_oklab, v_uv).rgb;
    float L = clamp(lab.x, 0.0, 1.0);
    float a = lab.y;
    float b = lab.z;
    // Light-peak protection only applies on the night side (amount > 0)
    // -- faded out, not just switched off, so it's continuous through 0
    // along with everything else.
    float protect = texture(u_protect, v_uv).r * max(u_amount, 0.0);

    vec2 mm = texelFetch(u_bMinMax, ivec2(0, 0), 0).rg;
    float bMin = mm.x, bMax = mm.y;

    float p = pow(2.0, -u_amount);
    float q = pow(2.0, u_amount);

    float bShifted;
    if (bMax - bMin > 1e-6) {
        float bNorm = clamp((b - bMin) / (bMax - bMin), 0.0, 1.0);
        float bNormNew = L * bNorm + (1.0 - L) * pow(bNorm, q);
        bShifted = bNormNew * (bMax - bMin) + bMin;
    } else {
        bShifted = b;
    }

    float LShifted = 1.0 - pow(clamp(1.0 - L, 0.0, 1.0), p);
    float LLit  = clamp(L + u_lightBoost * protect, 0.0, 1.0);

    float LNew = LShifted * (1.0 - protect) + LLit * protect;
    float bNew = bShifted * (1.0 - protect) + b * protect;

    vec3 rgbLin = oklabToRgbLin(vec3(LNew, a, bNew));
    fragColor = vec4(clamp(linearToSrgb(rgbLin), 0.0, 1.0), 1.0);
}
