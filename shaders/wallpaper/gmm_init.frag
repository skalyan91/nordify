#version 330 core
// Pass 11: seed K initial GMM components (mean, variance, weight),
// spread evenly BY VALUE over [0, 1] -- not by mass-quantile the way the
// earlier K-means seeding placed its centroids. This isn't a simplification;
// mass-quantile seeding was tried first and measurably fails for exactly
// the depth histograms this pipeline sees most often (a large, flat,
// dominant background region -- sky, a wall -- plus a much smaller
// foreground figure and/or a smooth gradient): confirmed on a synthetic
// histogram shaped like that (a dominant near-zero spike, a small tight
// figure spike, a diffuse mid-range gradient), quantile seeding placed
// 3-4 of 5 initial means already crowded on top of the dominant spike,
// and EM's soft responsibility never recovers from that start -- those
// components keep re-subdividing the same spike every iteration (each
// gets an outsized likelihood reward for precisely fitting a tall, tight
// mode), leaving the figure and the gradient to share whatever's left,
// sometimes just one badly-fit catch-all component. Value-uniform seeding
// starts every component in a different part of the value range
// regardless of where the mass actually concentrates, so EM has to prove
// each one is worth keeping there rather than starting several already
// stacked on the easiest local optimum -- confirmed on the same synthetic
// histogram to converge to a spread closely matching (and, given each
// component's own fitted variance, arguably more informative than) plain
// K-means's mass-proportional centroid placement, with no collapsed
// duplicates.

out vec4 fragColor;   // (mean, variance, weight, 1.0)

uniform int u_k;

void main() {
    int idx = int(gl_FragCoord.x);
    float mean = (float(idx) + 0.5) / float(u_k);
    // Each component starts "owning" an equal 1/K-wide slice of [0, 1];
    // variance of a Uniform(a, b) is (b-a)^2/12, the closed form for that
    // assumption. EM immediately re-estimates this once real data comes in.
    float slice = 1.0 / float(u_k);
    float variance = max(1e-4, slice * slice / 12.0);
    fragColor = vec4(mean, variance, slice, 1.0);
}
