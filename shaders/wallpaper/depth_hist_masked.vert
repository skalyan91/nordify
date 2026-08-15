#version 330 core
// Pass 3 (vertex half): scatter a 256-bin depth histogram, counting only
// pixels belonging to the winning figure segment (from argmax_1d.frag's
// output). Same GL_POINTS-scatter-via-blending technique as
// segment_score.vert. Feeds median_1d.frag, which ports the
// `np.median(depth_raw[piece_mask])` half of depth_blur.py:882/893.

uniform sampler2D u_depth;
uniform sampler2D u_segmentation;
uniform sampler2D u_depthMinMax;   // 1x1 RG32F
uniform sampler2D u_figureInfo;    // 1x1: (winningSegId, weight) from argmax_1d.frag
uniform ivec2 u_imageSize;

const int NUM_BINS = 256;

out float v_include;

void main() {
    int id = gl_VertexID;
    ivec2 p = ivec2(id % u_imageSize.x, id / u_imageSize.x);

    float segId = texelFetch(u_segmentation, p, 0).r;
    float winId = texelFetch(u_figureInfo, ivec2(0, 0), 0).r;
    v_include = (round(segId) == round(winId)) ? 1.0 : 0.0;

    float depth = texelFetch(u_depth, p, 0).r;
    vec2  mm    = texelFetch(u_depthMinMax, ivec2(0, 0), 0).rg;
    float normDepth = (mm.g - mm.r > 1e-6) ? (depth - mm.r) / (mm.g - mm.r) : 0.0;
    int bin = clamp(int(floor(normDepth * float(NUM_BINS))), 0, NUM_BINS - 1);

    float xNdc = (float(bin) + 0.5) / float(NUM_BINS) * 2.0 - 1.0;
    gl_Position = vec4(xNdc, 0.0, 0.0, 1.0);
}
