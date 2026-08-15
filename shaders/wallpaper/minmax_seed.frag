#version 330 core
// Pass 0a: seed the min/max reduction chain from the raw depth texture.
// depth_blur.py normalises disparity via depth_raw.min()/.max() throughout
// (_depth_blur, _detect_figure_focus) -- this is that reduction's first
// step, done on GPU instead of numpy so no CPU readback is needed even
// though the depth map can change at runtime.

in vec2 v_uv;
out vec2 fragColor;   // (min, max)

uniform sampler2D u_depth;   // R32F, raw disparity (higher = closer)

void main() {
    float d = texture(u_depth, v_uv).r;
    fragColor = vec2(d, d);
}
