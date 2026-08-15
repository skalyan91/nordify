#version 330 core
// Pass 13: sort the K final GMM component means ascending, so
// composite.frag's generalised (non-uniform) tent basis can walk them
// in order. K <= MAX_K is tiny, so every fragment redundantly re-sorts
// the whole array and keeps only its own index -- trivial cost. No
// separate "boundaries" output: composite.frag derives each slab's tent
// directly from its two neighbouring sorted centres.
//
// Reads only the mean channel (.r) of u_centroids, which works
// unchanged whether that texture is GMM's RGBA32F (mean, variance,
// weight, _) components or a plain R32F texture -- a sampler2D doesn't
// care which, and this pass only ever needs the mean.

out vec4 fragColor;   // (sortedMean, 0, 0, 0)

uniform sampler2D u_centroids;   // K x 1, RGBA32F (GMM components, final, unsorted) -- only .r is read
uniform int u_k;

const int MAX_K = 8;

void main() {
    float vals[MAX_K];
    for (int j = 0; j < u_k; j++) vals[j] = texelFetch(u_centroids, ivec2(j, 0), 0).r;

    for (int i = 1; i < u_k; i++) {
        float key = vals[i];
        int j = i - 1;
        while (j >= 0 && vals[j] > key) {
            vals[j + 1] = vals[j];
            j--;
        }
        vals[j + 1] = key;
    }

    int myIdx = int(gl_FragCoord.x);
    fragColor = vec4(vals[myIdx], 0.0, 0.0, 0.0);
}
