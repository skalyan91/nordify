#version 330 core
in float v_include;
out vec4 fragColor;

void main() {
    fragColor = vec4(v_include, 0.0, 0.0, 0.0);
}
