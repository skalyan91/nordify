#version 330 core
// Reusable horizontal half of a separable Gaussian blur, parametrised by
// sigma at runtime rather than baked into per-sigma shader files -- one
// pass pair serves all 6 detection-scale blurs (normalised) *and* all 5
// protection-bump blurs (unnormalised, see combine_max.frag's header for
// why summing separable bumps stands in for _light_protection_map's
// per-peak maximum). Always reads/writes the source's .r channel, so it
// works equally on oklab_convert.frag's L channel and dog_nms.frag's
// single-channel peak-strength fields.
//
// u_radius (host-computed as ceil(3*sigma), same 3-sigma cutoff
// _dog_light_peaks/_light_protection_map's cv2.GaussianBlur and Gaussian
// bump both implicitly use) bounds the *runtime* extent; MAX_RADIUS
// bounds the compile-time loop. The branch is uniform across an entire
// draw (u_radius doesn't vary per-pixel), so this doesn't cost warp
// divergence, just some skipped-iteration overhead for small sigmas --
// acceptable for a compute-once (not per-frame) pipeline.

out vec4 fragColor;

uniform sampler2D u_src;
uniform ivec2 u_imageSize;
uniform float u_sigma;
uniform int u_radius;
uniform bool u_normalize;

const int MAX_RADIUS = 96;

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    float sum = 0.0, wsum = 0.0;
    for (int i = -MAX_RADIUS; i <= MAX_RADIUS; i++) {
        if (i < -u_radius || i > u_radius) continue;
        int x = clamp(p.x + i, 0, u_imageSize.x - 1);
        float w = exp(-float(i * i) / (2.0 * u_sigma * u_sigma));
        sum  += w * texelFetch(u_src, ivec2(x, p.y), 0).r;
        wsum += w;
    }
    float result = u_normalize ? sum / wsum : sum;
    fragColor = vec4(result, 0.0, 0.0, 0.0);
}
