#version 330 core
// Pass 1 (vertex half): scatter each source pixel's foveal x depth weight
// into its segment's accumulator slot. Draw with
// glDrawArrays(GL_POINTS, 0, W*H) and no vertex buffer -- gl_VertexID
// alone determines the source pixel. Framebuffer target is
// MAX_SEGMENTS x 1, blended with (GL_ONE, GL_ONE) so slots simply sum.
//
// Ports depth_blur.py:851-853 (combined_weight = foveal_weight *
// normalised depth) per source pixel, scattered by its segment ID
// instead of being summed with numpy's `combined_weight[piece_mask].sum()`.

uniform sampler2D u_depth;         // R32F, raw disparity (higher = closer)
uniform sampler2D u_segmentation;  // R32F, integer candidate-region ID per pixel (0 = background)
uniform sampler2D u_depthMinMax;   // 1x1 RG32F: (min, max) of u_depth, from the minmax_* pass chain
uniform ivec2 u_imageSize;
uniform float u_viewingDistanceFactor;   // depth_blur.py default 1.5
uniform float u_e2Degrees;               // depth_blur.py default 2.3

const int MAX_SEGMENTS = 64;   // must match the host's accumulator texture width

out float v_weight;

void main() {
    int id = gl_VertexID;
    ivec2 p = ivec2(id % u_imageSize.x, id / u_imageSize.x);

    float depth = texelFetch(u_depth, p, 0).r;
    vec2  mm    = texelFetch(u_depthMinMax, ivec2(0, 0), 0).rg;
    float normDepth = (mm.g - mm.r > 1e-6) ? (depth - mm.r) / (mm.g - mm.r) : 0.0;

    float fov = fovealWeight(vec2(p) + 0.5, vec2(u_imageSize), u_viewingDistanceFactor, u_e2Degrees);
    v_weight = fov * normDepth;

    float segId = texelFetch(u_segmentation, p, 0).r;
    float xNdc  = (round(segId) + 0.5) / float(MAX_SEGMENTS) * 2.0 - 1.0;

    // Point rasterises to exactly one pixel at the framebuffer's default
    // point size (1.0) -- do not call glPointSize() to something else.
    gl_Position = vec4(xNdc, 0.0, 0.0, 1.0);
}
