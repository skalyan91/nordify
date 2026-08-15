#version 330 core
// Pass 5a (2/3): vertical half of the separable blur (see luminance_blur_h.frag).

out float fragColor;

uniform sampler2D u_gray;   // R32F, output of luminance_blur_h.frag
uniform ivec2 u_imageSize;

const int   RADIUS = 4;
const float WEIGHTS[9] = float[](
    0.00761442, 0.03607497, 0.10958608, 0.21344454, 0.26655997,
    0.21344454, 0.10958608, 0.03607497, 0.00761442
);

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    float sum = 0.0;
    for (int i = -RADIUS; i <= RADIUS; i++) {
        int y = clamp(p.y + i, 0, u_imageSize.y - 1);
        sum += WEIGHTS[i + RADIUS] * texelFetch(u_gray, ivec2(p.x, y), 0).r;
    }
    fragColor = sum;
}
