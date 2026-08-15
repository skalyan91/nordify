#version 330 core
// Pass 8a: for one candidate crop offset, compute the risk-weighted
// gradient magnitude at every (row, crop-axis-position-within-window)
// cell -- ports the per-offset operand of entropy_crop.py's
// _correlate_valid (a weighted sliding window, computed here directly
// per candidate rather than via FFT, since candidates are few and this
// only runs at setup time). Output is always oriented crop-axis=width,
// perpendicular-axis=height regardless of whether the image's own crop
// axis is x or y (entropy_crop.py's `m = mag if axis=='x' else mag.T`
// does the same canonicalisation), so reduce_pairwise.frag downstream
// never needs to know which image axis is being cropped.

out vec2 fragColor;   // (weightedMag, 0) -- RG so reduce_pairwise.frag can consume it directly

uniform sampler2D u_gradMag;    // full image gradient magnitude, R32F
uniform int u_cropAxisIsX;      // 1: crop axis is image x, 0: crop axis is image y
uniform int u_offset;           // this candidate's crop-axis offset
uniform int u_newLength;        // kept window length along the crop axis
uniform int u_other;            // perpendicular axis length
uniform float u_viewingDistanceFactor;
uniform float u_e2Degrees;

void main() {
    ivec2 dst = ivec2(gl_FragCoord.xy);   // dst.x in [0, newLength), dst.y in [0, other)
    int k = dst.x;
    int r = dst.y;

    ivec2 srcP = (u_cropAxisIsX == 1) ? ivec2(u_offset + k, r) : ivec2(r, u_offset + k);
    float mag = texelFetch(u_gradMag, srcP, 0).r;

    float risk = edgeRiskWeight(float(k), float(u_newLength), float(u_other),
                                u_viewingDistanceFactor, u_e2Degrees);
    fragColor = vec2(mag * risk, 0.0);
}
