#version 330 core
// Pass 7: one Hillis-Steele step of an inclusive prefix (running) sum
// over a 1D array stored as a `length x 1` texture. Host runs this
// ceil(log2(length)) times with u_step = 1, 2, 4, ..., ping-ponging src
// and dst textures each time (can't read and write the same texture).
//
// out[i] = in[i] + (i >= step ? in[i - step] : 0)
//
// After the final step, texel i holds sum(in[0..i]) inclusive -- used
// for the figure-cut-fraction lookup in combine_and_argmin.frag (a
// window's count = prefixAt(end) - prefixAt(start-1)).

out vec4 fragColor;

uniform sampler2D u_src;   // length x 1, R32F
uniform int u_step;

void main() {
    int i = int(gl_FragCoord.x);
    float v = texelFetch(u_src, ivec2(i, 0), 0).r;
    if (i >= u_step) {
        v += texelFetch(u_src, ivec2(i - u_step, 0), 0).r;
    }
    fragColor = vec4(v, 0.0, 0.0, 0.0);
}
