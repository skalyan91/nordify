#version 330 core
// Pass 8b (and 12's histogram use, and general-purpose elsewhere): one
// 2-to-1 step of a pairwise SUM reduction over an RG32F buffer, along
// either its width or height. Run repeatedly (halving the chosen axis
// each time, rounding up) until that axis reaches 1. Both channels
// reduce identically -- used with the second channel unused (0) for a
// plain sum (e.g. weighted_mag_for_offset.frag's crop-axis reduction),
// and with both channels live for the (value, x*log2(x)) moment pair
// entropy_seed.frag produces (see common.glsl's entropyFromMoments).

out vec4 fragColor;

uniform sampler2D u_src;    // RG32F
uniform ivec2 u_srcSize;
uniform int u_axisIsX;      // 1: halve width, 0: halve height

void main() {
    ivec2 dst = ivec2(gl_FragCoord.xy);
    ivec2 a, b;
    if (u_axisIsX == 1) {
        a = ivec2(min(2 * dst.x,     u_srcSize.x - 1), dst.y);
        b = ivec2(min(2 * dst.x + 1, u_srcSize.x - 1), dst.y);
    } else {
        a = ivec2(dst.x, min(2 * dst.y,     u_srcSize.y - 1));
        b = ivec2(dst.x, min(2 * dst.y + 1, u_srcSize.y - 1));
    }
    vec2 sum = texelFetch(u_src, a, 0).rg + texelFetch(u_src, b, 0).rg;
    fragColor = vec4(sum, 0.0, 0.0);
}
