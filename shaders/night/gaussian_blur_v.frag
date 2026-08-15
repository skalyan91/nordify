#version 330 core
// Vertical half of the separable blur -- see gaussian_blur_h.frag.

out vec4 fragColor;

uniform sampler2D u_src;
uniform ivec2 u_imageSize;
uniform float u_sigma;
uniform int u_radius;
uniform bool u_normalize;

const int MAX_RADIUS = 96;

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    float sum = 0.0, wsum = 0.0;
    for (int i = -MAX_RADIUS; i <= MAX_RADIUS; i++) {
        if (i < -u_radius || i > u_radius) continue;
        int y = clamp(p.y + i, 0, u_imageSize.y - 1);
        float w = exp(-float(i * i) / (2.0 * u_sigma * u_sigma));
        sum  += w * texelFetch(u_src, ivec2(p.x, y), 0).r;
        wsum += w;
    }
    float result = u_normalize ? sum / wsum : sum;
    fragColor = vec4(result, 0.0, 0.0, 0.0);
}
