#version 330 core
in float v_weight;
out vec4 fragColor;

void main() {
    fragColor = vec4(v_weight, 0.0, 0.0, 0.0);
}
