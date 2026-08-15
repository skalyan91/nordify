#version 330 core
// Downsamples u_src by 2x via a 2x2 box filter (plain average) -- run
// once before the expensive multi-scale blur/DoG detection stack (see
// pipeline.py's/night.js's run() for how this fits into the wider
// pipeline and the accuracy tradeoff it makes: every large-radius blur
// pass that follows then operates on 1/4 the pixels with half the
// effective radius, since separable-blur cost scales as pixels x radius.

out vec4 fragColor;

uniform sampler2D u_src;
uniform ivec2 u_srcSize;

void main() {
    ivec2 dstCoord = ivec2(gl_FragCoord.xy);
    ivec2 base = dstCoord * 2;
    ivec2 maxCoord = u_srcSize - 1;
    ivec2 c00 = min(base, maxCoord);
    ivec2 c10 = min(base + ivec2(1, 0), maxCoord);
    ivec2 c01 = min(base + ivec2(0, 1), maxCoord);
    ivec2 c11 = min(base + ivec2(1, 1), maxCoord);
    vec4 sum = texelFetch(u_src, c00, 0) + texelFetch(u_src, c10, 0)
             + texelFetch(u_src, c01, 0) + texelFetch(u_src, c11, 0);
    fragColor = sum * 0.25;
}
