#version 330 core
// Pass 6 (vertex half): scatter, along the crop axis only (summing the
// perpendicular axis in), how many figure-segment pixels sit at each
// crop-axis position. Feeds the O(1)-per-candidate figure-cut penalty in
// combine_and_argmin.frag (an extension beyond entropy_crop.py -- see
// README.md's "figure-sensitive" section).

uniform sampler2D u_segmentation;
uniform sampler2D u_figureInfo;   // 1x1: (winningSegId, weight)
uniform ivec2 u_imageSize;
uniform int u_cropAxisIsX;        // 1: index by column (x), 0: index by row (y)
uniform int u_cropAxisLength;     // u_imageSize.x if u_cropAxisIsX else u_imageSize.y

out float v_include;

void main() {
    int id = gl_VertexID;
    ivec2 p = ivec2(id % u_imageSize.x, id / u_imageSize.x);

    float segId = texelFetch(u_segmentation, p, 0).r;
    float winId = texelFetch(u_figureInfo, ivec2(0, 0), 0).r;
    v_include = (round(segId) == round(winId)) ? 1.0 : 0.0;

    int pos = (u_cropAxisIsX == 1) ? p.x : p.y;
    float xNdc = (float(pos) + 0.5) / float(u_cropAxisLength) * 2.0 - 1.0;
    gl_Position = vec4(xNdc, 0.0, 0.0, 1.0);
}
