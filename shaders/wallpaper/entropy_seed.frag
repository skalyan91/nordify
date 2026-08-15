#version 330 core
// Pass 8c (1/2): after weighted_mag_for_offset.frag's output has been
// reduce_pairwise'd (axisIsX=1) down to a 1 x other "profile" column
// (entropy_crop.py's `weighted_sums[:, o]`), seed the (value, x*log2(x))
// moment pair per row -- clamping negative float noise to 0 first,
// matching Python's `np.clip(weighted_sums[:, o], 0, None)`. The
// resulting RG buffer is then reduce_pairwise'd again (axisIsX=0) down
// to a single (total, plogp) texel, finished by entropy_1d.frag.

out vec4 fragColor;

uniform sampler2D u_profile;   // 1 x other, RG32F (R = weighted sum for this row)

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    float v = max(texelFetch(u_profile, p, 0).r, 0.0);
    fragColor = vec4(v, xlog2x(v), 0.0, 0.0);
}
