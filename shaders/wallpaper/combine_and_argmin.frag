#version 330 core
// Pass 9: combine each crop candidate's entropy (from entropy_1d.frag)
// with its figure-cut penalty, and argmin over the NUM_CANDIDATES
// results. Ports entropy_crop.py:205's
// `np.lexsort((totals, entropies))[0]` (primary key: entropy, secondary:
// total edge weight cut) plus the figure-sensitivity extension described
// in README.md.
//
// The figure-cut penalty is O(1) per candidate given the prefix sum from
// prefix_sum_1d.frag: countInWindow = prefixAt(end) - prefixAt(start-1),
// cutFraction = 1 - countInWindow/total. Scaled by u_maxEntropyBits
// (= log2(other), entropy's own max possible value) so the two terms
// are commensurate regardless of image size, then by
// u_figurePenaltyWeight (default 4.0) so avoiding the figure outright
// dominates ordinary entropy differences, while candidates that both
// avoid it fully are still compared on entropy alone (penalty = 0).

out vec4 fragColor;   // (bestOffset, bestScore, 0, 0)

const int NUM_CANDIDATES = 32;   // must match the host's u_candidateScores width

uniform sampler2D u_candidateScores;     // NUM_CANDIDATES x 1, RG32F: (entropy, total)
uniform sampler2D u_figureCountPrefix;   // cropAxisLength x 1, R32F, inclusive prefix sum
uniform int   u_candidateOffsets[NUM_CANDIDATES];
uniform int   u_newLength;
uniform int   u_cropAxisLength;
uniform float u_maxEntropyBits;
uniform float u_figurePenaltyWeight;

float prefixAt(int i) {   // inclusive prefix sum at index i; i<0 reads as 0
    if (i < 0) return 0.0;
    return texelFetch(u_figureCountPrefix, ivec2(i, 0), 0).r;
}

void main() {
    float figureTotal = prefixAt(u_cropAxisLength - 1);

    float bestScore  = 1.0e30;
    float bestOffset = 0.0;
    float bestTotal  = 1.0e30;

    for (int c = 0; c < NUM_CANDIDATES; c++) {
        vec2 es = texelFetch(u_candidateScores, ivec2(c, 0), 0).rg;   // (entropy, total)
        int offset = u_candidateOffsets[c];
        int endIdx = offset + u_newLength - 1;

        float countInWindow = prefixAt(endIdx) - prefixAt(offset - 1);
        float cutFraction = (figureTotal > 0.0)
            ? clamp(1.0 - countInWindow / figureTotal, 0.0, 1.0)
            : 0.0;
        float penalty = u_figurePenaltyWeight * cutFraction * u_maxEntropyBits;

        float score = es.r + penalty;
        bool better = (score < bestScore) || (score == bestScore && es.g < bestTotal);
        if (better) { bestScore = score; bestOffset = float(offset); bestTotal = es.g; }
    }

    fragColor = vec4(bestOffset, bestScore, 0.0, 0.0);
}
