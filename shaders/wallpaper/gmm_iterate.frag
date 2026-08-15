#version 330 core
// Pass 12: one EM step for a 1D Gaussian Mixture Model over the cropped
// depth histogram, replacing Lloyd's-algorithm K-means. Host runs this
// several times, ping-ponging two K x 1 RGBA32F "component" textures
// (mean, variance, weight, unused) -- reads u_components from the
// previous step, writes the next step's (can't alias the same texture
// for both). K and NUM_BINS are both tiny, so every fragment redundantly
// re-scans all 256 bins against all K components; negligible cost at
// this size (same trade kmeans_iterate.frag made).
//
// GMM strictly generalises K-means: hard nearest-centroid assignment is
// the equal-variance, zero-temperature limit of soft, density-weighted
// responsibility assignment. Modelling each component's own variance
// separately -- rather than implicitly assuming every cluster has the
// same "size" the way plain nearest-centroid assignment does -- lets a
// tight, compact depth layer (e.g. a sharply-defined foreground figure)
// and a wide, diffuse one (e.g. a smoothly graded background) coexist
// without the wide one's own scale distorting where the tight one's
// boundary should sit, and lets gradually-blending depth (no hard planes
// at all) split into overlapping soft components instead of being forced
// across an arbitrary hard K-means boundary.

out vec4 fragColor;   // (newMean, newVariance, newWeight, 1.0)

uniform sampler2D u_hist;         // NUM_BINS x 1, R32F
uniform sampler2D u_components;   // K x 1, RGBA32F: (mean, variance, weight, _) -- previous step
uniform int u_k;

const int NUM_BINS = 256;
const int MAX_K = 8;   // must match sort_and_bounds.frag's / composite.frag's MAX_K
const float MIN_VARIANCE = 1e-4;
const float TWO_PI = 6.283185307179586;

float gaussianDensity(float x, float mean, float variance) {
    float d = x - mean;
    return exp(-(d * d) / (2.0 * variance)) / sqrt(TWO_PI * variance);
}

// Responsibility of component `myIdx` for one histogram bin, given the
// PREVIOUS iteration's (fixed) component parameters -- the standard EM
// E-step, computed on demand rather than materialised into a (256, K)
// buffer since nothing here is expensive enough to need caching.
float responsibility(float binValue, int myIdx, float means[MAX_K], float variances[MAX_K], float weights[MAX_K]) {
    float denom = 0.0;
    float numerMine = 0.0;
    for (int j = 0; j < u_k; j++) {
        float p = weights[j] * gaussianDensity(binValue, means[j], variances[j]);
        denom += p;
        if (j == myIdx) numerMine = p;
    }
    return (denom > 0.0) ? (numerMine / denom) : 0.0;
}

void main() {
    int myIdx = int(gl_FragCoord.x);

    float means[MAX_K];
    float variances[MAX_K];
    float weights[MAX_K];
    for (int j = 0; j < u_k; j++) {
        vec4 c = texelFetch(u_components, ivec2(j, 0), 0);
        means[j] = c.r; variances[j] = c.g; weights[j] = c.b;
    }

    // M-step, mean: needs this component's own total responsibility mass
    // ("N_j") and the responsibility-weighted sum of bin values, both
    // against the OLD (fixed) parameters above.
    float weightedSum = 0.0;
    float totalResp   = 0.0;
    float totalMass   = 0.0;
    for (int i = 0; i < NUM_BINS; i++) {
        float binValue  = (float(i) + 0.5) / float(NUM_BINS);
        float binWeight = texelFetch(u_hist, ivec2(i, 0), 0).r;
        totalMass += binWeight;
        if (binWeight <= 0.0) continue;

        float r = responsibility(binValue, myIdx, means, variances, weights);
        weightedSum += binWeight * r * binValue;
        totalResp   += binWeight * r;
    }

    // Empty component: keep the previous parameters rather than collapsing
    // to a degenerate (mean, ~0 variance, ~0 weight) triple -- same guard
    // kmeans_iterate.frag used for an empty cluster, extended to all three
    // GMM parameters so a dead component doesn't poison sort_and_bounds.frag's
    // ordering or get stuck at zero responsibility on every future E-step.
    if (totalResp <= 0.0) {
        fragColor = vec4(means[myIdx], variances[myIdx], weights[myIdx], 1.0);
        return;
    }

    float newMean = weightedSum / totalResp;

    // M-step, variance: re-scanned around the just-updated mean (standard
    // EM order -- update the mean, then the variance around it, within
    // the same iteration), reusing the same fixed responsibilities.
    float weightedSqSum = 0.0;
    for (int i = 0; i < NUM_BINS; i++) {
        float binValue  = (float(i) + 0.5) / float(NUM_BINS);
        float binWeight = texelFetch(u_hist, ivec2(i, 0), 0).r;
        if (binWeight <= 0.0) continue;

        float r = responsibility(binValue, myIdx, means, variances, weights);
        float d = binValue - newMean;
        weightedSqSum += binWeight * r * d * d;
    }

    float newVariance = max(MIN_VARIANCE, weightedSqSum / totalResp);
    float newWeight   = totalResp / max(totalMass, 1e-20);

    fragColor = vec4(newMean, newVariance, newWeight, 1.0);
}
