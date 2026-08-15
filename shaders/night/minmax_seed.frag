#version 330 core
// Pass 1b (seed): seed the min/max reduction chain from the Oklab `b`
// channel (needed for _nighttime's b_norm step). Same pattern as
// wallpaper/minmax_seed.frag, just reading .b from the Oklab texture
// instead of a standalone depth texture.

in vec2 v_uv;
out vec2 fragColor;   // (min, max)

uniform sampler2D u_oklab;   // (L, a, b) from oklab_convert.frag

void main() {
    float b = texture(u_oklab, v_uv).b;
    fragColor = vec2(b, b);
}
