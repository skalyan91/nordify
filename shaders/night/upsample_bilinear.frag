#version 330 core
// Bilinear-upsamples u_src (u_srcSize) to the current framebuffer's
// (larger) u_dstSize -- brings downsample2x.frag's reduced-resolution
// detection output (the final protection map) back to full resolution
// before nighttime_resolve.frag consumes it. Manual bilinear (4
// texelFetches + lerp) rather than relying on the texture's own sampler
// filtering, matching this codebase's texelFetch-only convention (every
// other pass here samples explicitly rather than through sampler filter
// state, which also keeps this correct regardless of how the source
// texture's min/mag filters happen to be set).

out vec4 fragColor;

uniform sampler2D u_src;
uniform ivec2 u_srcSize;
uniform ivec2 u_dstSize;

void main() {
    // Standard pixel-centre-aligned resize mapping: dst pixel centre
    // gl_FragCoord.xy (already i+0.5), rescaled into src space and
    // re-offset by -0.5 to land back in "texel index" coordinates.
    vec2 srcPos = (gl_FragCoord.xy / vec2(u_dstSize)) * vec2(u_srcSize) - 0.5;
    vec2 f = fract(srcPos);
    ivec2 base = ivec2(floor(srcPos));
    ivec2 maxCoord = u_srcSize - 1;

    ivec2 c00 = clamp(base, ivec2(0), maxCoord);
    ivec2 c10 = clamp(base + ivec2(1, 0), ivec2(0), maxCoord);
    ivec2 c01 = clamp(base + ivec2(0, 1), ivec2(0), maxCoord);
    ivec2 c11 = clamp(base + ivec2(1, 1), ivec2(0), maxCoord);

    vec4 v00 = texelFetch(u_src, c00, 0);
    vec4 v10 = texelFetch(u_src, c10, 0);
    vec4 v01 = texelFetch(u_src, c01, 0);
    vec4 v11 = texelFetch(u_src, c11, 0);

    vec4 top = mix(v00, v10, f.x);
    vec4 bot = mix(v01, v11, f.x);
    fragColor = mix(top, bot, f.y);
}
