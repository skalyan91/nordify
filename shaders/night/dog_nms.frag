#version 330 core
// Pass 3: one Difference-of-Gaussians level's peak detection, ported
// from _dog_light_peaks (palettize.py:997). DoG values are computed
// inline from the blur stack (dog_level = blur_level - blur_{level+1})
// rather than written out as their own textures -- this pass already
// needs several blur levels at once (its own neighbours, for the
// cross-scale check), so nothing is saved by precomputing DoGs
// separately.
//
// Run once per level (u_level = 0..4, host loop), each into its own
// output texture. A peak must be a *spatial* local maximum (its own 3x3
// neighbourhood, matching cv2.dilate with a 3x3 rect kernel: dog >=
// dilate(dog) === dog >= every neighbour, ties allowed) AND a *scale*
// local maximum (>= the same pixel's DoG value at the adjacent levels)
// AND clear the absolute threshold. Output is 0 where not a peak, else
// the DoG value itself (consumed as bump amplitude by the protection
// blur pass).

out vec4 fragColor;

uniform sampler2D u_blur0, u_blur1, u_blur2, u_blur3, u_blur4, u_blur5;
uniform ivec2 u_imageSize;
uniform int   u_level;        // 0..4: this pass computes dog_level = blur_level - blur_{level+1}
uniform float u_threshold;    // palettize.py default 0.12
uniform float u_strengthScale;   // palettize.py's _light_protection_map default 3.0

float fetchBlur(int idx, ivec2 p) {
    p = clamp(p, ivec2(0), u_imageSize - 1);
    if (idx == 0) return texelFetch(u_blur0, p, 0).r;
    if (idx == 1) return texelFetch(u_blur1, p, 0).r;
    if (idx == 2) return texelFetch(u_blur2, p, 0).r;
    if (idx == 3) return texelFetch(u_blur3, p, 0).r;
    if (idx == 4) return texelFetch(u_blur4, p, 0).r;
    return texelFetch(u_blur5, p, 0).r;
}

float dogAt(int level, ivec2 p) {
    return fetchBlur(level, p) - fetchBlur(level + 1, p);
}

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    float center = dogAt(u_level, p);

    bool isMax = true;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) continue;
            if (dogAt(u_level, p + ivec2(dx, dy)) > center) isMax = false;
        }
    }

    if (u_level > 0 && dogAt(u_level - 1, p) > center) isMax = false;
    if (u_level < 4 && dogAt(u_level + 1, p) > center) isMax = false;

    isMax = isMax && (center > u_threshold);

    // Output the bump *amplitude* (_light_protection_map's
    // `amp = min(1, strength*strength_scale)`) directly -- raw DoG
    // strength has no other consumer, so there's no reason to defer
    // this pointwise transform to a separate pass.
    float amp = isMax ? min(1.0, center * u_strengthScale) : 0.0;
    fragColor = vec4(amp, 0.0, 0.0, 0.0);
}
