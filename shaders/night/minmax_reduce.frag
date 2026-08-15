#version 330 core
// Pass 1b (reduce): one 2x2->1 step of the min/max reduction chain. Run
// ceil(log2(max(W,H))) times, each halving both dimensions (rounding up),
// until the output is 1x1 -- host-driven ping-pong. Identical to
// wallpaper/minmax_reduce.frag (copied rather than shared since the two
// pipelines are otherwise independent).

out vec2 fragColor;   // (min, max)

uniform sampler2D u_src;     // RG32F, previous step's (min, max)
uniform ivec2 u_srcSize;     // previous step's pixel dimensions

void main() {
    ivec2 dst = ivec2(gl_FragCoord.xy);
    ivec2 x0y0 = ivec2(min(2 * dst.x,     u_srcSize.x - 1), min(2 * dst.y,     u_srcSize.y - 1));
    ivec2 x1y0 = ivec2(min(2 * dst.x + 1, u_srcSize.x - 1), min(2 * dst.y,     u_srcSize.y - 1));
    ivec2 x0y1 = ivec2(min(2 * dst.x,     u_srcSize.x - 1), min(2 * dst.y + 1, u_srcSize.y - 1));
    ivec2 x1y1 = ivec2(min(2 * dst.x + 1, u_srcSize.x - 1), min(2 * dst.y + 1, u_srcSize.y - 1));

    vec2 a = texelFetch(u_src, x0y0, 0).rg;
    vec2 b = texelFetch(u_src, x1y0, 0).rg;
    vec2 c = texelFetch(u_src, x0y1, 0).rg;
    vec2 d = texelFetch(u_src, x1y1, 0).rg;

    float mn = min(min(a.r, b.r), min(c.r, d.r));
    float mx = max(max(a.g, b.g), max(c.g, d.g));
    fragColor = vec2(mn, mx);
}
