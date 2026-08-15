#version 330 core
// Pass 5a (1/3): horizontal half of a separable Gaussian blur (sigma=1.5,
// radius=4, BORDER_REPLICATE) on the crop edge-source image's luma --
// entropy_crop.py's _gradient_magnitude does this before Sobel. Uses
// OpenCV's own BGR2GRAY weights (ITU-R BT.601, applied to the raw sRGB
// bytes, not linearised light) rather than common.glsl's `luminance()`
// (linear-light BT.709), to match cv2.cvtColor exactly -- gradient
// detection doesn't care about physical linearity, only about matching
// the Python reference's actual pixel values.
//
// BORDER_REPLICATE is implemented by clamping the fetch coordinate
// itself (texelFetch, not a filtered sampler), so it holds regardless
// of the host's texture wrap-mode setup.

out float fragColor;

uniform sampler2D u_image;   // sRGB bytes-as-floats, [0,1]
uniform ivec2 u_imageSize;

const int   RADIUS = 4;
const float WEIGHTS[9] = float[](
    0.00761442, 0.03607497, 0.10958608, 0.21344454, 0.26655997,
    0.21344454, 0.10958608, 0.03607497, 0.00761442
);

float cvGray(vec3 srgb) {
    return 0.299 * srgb.r + 0.587 * srgb.g + 0.114 * srgb.b;
}

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    float sum = 0.0;
    for (int i = -RADIUS; i <= RADIUS; i++) {
        int x = clamp(p.x + i, 0, u_imageSize.x - 1);
        vec3 srgb = texelFetch(u_image, ivec2(x, p.y), 0).rgb;
        sum += WEIGHTS[i + RADIUS] * cvGray(srgb);
    }
    fragColor = sum;
}
