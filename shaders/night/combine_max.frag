#version 330 core
// Pass 5: combine the 5 per-level protection maps. palettize.py's
// _light_protection_map combines overlapping bumps via *maximum*
// (palettize.py:1055) so a light detected at more than one adjacent scale
// doesn't get over-protected; this pass sums them and clamps to [0,1]
// instead -- see gaussian_blur_h.frag's header for why (a 2D Gaussian
// bump is separable, so summing is the cheap operation; max has no
// equivalently cheap separable form). The two agree exactly for
// isolated peaks (the common case) and only differ when several peaks'
// bumps overlap heavily, where sum-then-clamp saturates to the same 1.0
// max would, just slightly earlier -- a deliberate real-time trade, not
// an oversight.

out vec4 fragColor;

uniform sampler2D u_p0, u_p1, u_p2, u_p3, u_p4;

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    float sum = texelFetch(u_p0, p, 0).r + texelFetch(u_p1, p, 0).r + texelFetch(u_p2, p, 0).r
              + texelFetch(u_p3, p, 0).r + texelFetch(u_p4, p, 0).r;
    fragColor = vec4(clamp(sum, 0.0, 1.0), 0.0, 0.0, 0.0);
}
