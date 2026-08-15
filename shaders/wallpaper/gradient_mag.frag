#version 330 core
// Pass 5b (3/3): Sobel gradient magnitude of the blurred luma (see
// luminance_blur_h/v.frag), BORDER_REPLICATE via clamped texelFetch.
// Ports entropy_crop.py's _gradient_magnitude exactly (ksize=3 Sobel,
// sqrt(gx^2+gy^2)).

out float fragColor;

uniform sampler2D u_grayBlurred;   // R32F
uniform ivec2 u_imageSize;

float fetchClamped(ivec2 p) {
    p = clamp(p, ivec2(0), u_imageSize - 1);
    return texelFetch(u_grayBlurred, p, 0).r;
}

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);

    float tl = fetchClamped(p + ivec2(-1, -1)), tc = fetchClamped(p + ivec2(0, -1)), tr = fetchClamped(p + ivec2(1, -1));
    float ml = fetchClamped(p + ivec2(-1,  0)),                                      mr = fetchClamped(p + ivec2(1,  0));
    float bl = fetchClamped(p + ivec2(-1,  1)), bc = fetchClamped(p + ivec2(0,  1)), br = fetchClamped(p + ivec2(1,  1));

    float gx = (tr + 2.0 * mr + br) - (tl + 2.0 * ml + bl);
    float gy = (bl + 2.0 * bc + br) - (tl + 2.0 * tc + tr);

    fragColor = sqrt(gx * gx + gy * gy);
}
