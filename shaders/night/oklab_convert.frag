#version 330 core
// Pass 1: sRGB -> Oklab (L, a, b). Ports palettize._bgr_to_oklab (channel
// order doesn't matter here -- the host uploads RGB either way).

in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D u_image;   // sRGB, [0, 1]

void main() {
    vec3 lin = srgbToLinear(texture(u_image, v_uv).rgb);
    fragColor = vec4(rgbLinToOklab(lin), 1.0);
}
