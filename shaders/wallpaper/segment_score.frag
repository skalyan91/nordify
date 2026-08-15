#version 330 core
// Pass 1 (fragment half): emit this point's weight; additive blending
// (GL_ONE, GL_ONE) does the accumulation across all W*H points landing
// on the same segment-ID column.

in float v_weight;
out vec4 fragColor;

void main() {
    fragColor = vec4(v_weight, 0.0, 0.0, 0.0);
}
