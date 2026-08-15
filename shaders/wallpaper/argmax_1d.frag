#version 330 core
// Pass 2: argmax over the MAX_SEGMENTS-wide accumulator from
// segment_score.*, skipping slot 0 (reserved for background/unassigned
// pixels -- see the segmentation-map contract in README.md). One 1x1
// draw. Ports depth_blur.py:891 (`max(candidates, key=...)`).
//
// Also reused (with U_DESCENDING=false via the host's per-draw define,
// or simply by feeding it -score) for pass 9's argmin over crop
// candidates -- both are "find the best of a handful of texels."

out vec4 fragColor;   // (bestId, bestValue, 0, 0)

uniform sampler2D u_accum;   // MAX_SEGMENTS x 1, R32F
uniform int u_count;         // number of texels to scan (MAX_SEGMENTS here)
uniform int u_startIndex;    // 1 here, to skip the background slot
uniform bool u_findMax;      // true: argmax (figure scoring); false: argmin (crop search)

void main() {
    float bestVal = u_findMax ? -1.0e30 : 1.0e30;
    float bestIdx = float(u_startIndex);

    for (int i = u_startIndex; i < u_count; i++) {
        float v = texelFetch(u_accum, ivec2(i, 0), 0).r;
        bool better = u_findMax ? (v > bestVal) : (v < bestVal);
        if (better) { bestVal = v; bestIdx = float(i); }
    }

    fragColor = vec4(bestIdx, bestVal, 0.0, 0.0);
}
