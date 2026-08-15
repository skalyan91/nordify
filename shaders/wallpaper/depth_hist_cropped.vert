#version 330 core
// Pass 10 (vertex half): scatter a foveally-weighted depth histogram,
// restricted to pixels inside the winning crop window (pass 9's
// u_cropOffset) -- feeds the GMM depth-layer clustering (an
// extension beyond depth_blur.py, per the user's request to replace its
// fixed 16 uniform tent slabs with a handful of adaptively-placed ones).
// Weighting by foveal acuity means depth mass far from the frame centre
// (background clutter) influences the layer placement less, matching
// the perceptual weighting already used elsewhere in this pipeline.

uniform sampler2D u_depth;
uniform sampler2D u_depthMinMax;   // 1x1 RG32F
uniform sampler2D u_cropOffset;    // 1x1: (bestOffset, bestScore) from combine_and_argmin.frag
uniform ivec2 u_imageSize;
uniform int   u_cropAxisIsX;
uniform int   u_newLength;
uniform float u_viewingDistanceFactor;
uniform float u_e2Degrees;

const int NUM_BINS = 256;

out float v_weight;

void main() {
    int id = gl_VertexID;
    ivec2 p = ivec2(id % u_imageSize.x, id / u_imageSize.x);

    int offset = int(texelFetch(u_cropOffset, ivec2(0, 0), 0).r + 0.5);
    int pos = (u_cropAxisIsX == 1) ? p.x : p.y;
    bool inWindow = (pos >= offset) && (pos < offset + u_newLength);

    float depth = texelFetch(u_depth, p, 0).r;
    vec2  mm    = texelFetch(u_depthMinMax, ivec2(0, 0), 0).rg;
    float normDepth = (mm.g - mm.r > 1e-6) ? (depth - mm.r) / (mm.g - mm.r) : 0.0;

    float fov = fovealWeight(vec2(p) + 0.5, vec2(u_imageSize), u_viewingDistanceFactor, u_e2Degrees);
    v_weight = inWindow ? fov : 0.0;

    int bin = clamp(int(floor(normDepth * float(NUM_BINS))), 0, NUM_BINS - 1);
    float xNdc = (float(bin) + 0.5) / float(NUM_BINS) * 2.0 - 1.0;
    gl_Position = vec4(xNdc, 0.0, 0.0, 1.0);
}
