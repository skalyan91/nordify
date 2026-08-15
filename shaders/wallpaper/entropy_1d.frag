#version 330 core
// Pass 8c (2/2): finish one candidate's score from its reduced (total,
// plogp) moment pair (see entropy_seed.frag / common.glsl's
// entropyFromMoments). The host draws this with the framebuffer's
// viewport set to this candidate's single texel within the shared
// NUM_CANDIDATES x 1 u_candidateScores texture, so no separate
// gather/combine pass is needed to place each candidate's result.

out vec4 fragColor;   // (entropy, total, 0, 0)

uniform sampler2D u_moments;   // 1x1 RG32F: (total, plogp)

void main() {
    vec2 m = texelFetch(u_moments, ivec2(0, 0), 0).rg;
    float ent = entropyFromMoments(m.r, m.g);
    fragColor = vec4(ent, m.r, 0.0, 0.0);
}
