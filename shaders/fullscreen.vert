#version 330 core

// Fullscreen triangle — no vertex buffer needed, draw with glDrawArrays(GL_TRIANGLES, 0, 3).
out vec2 v_uv;

void main() {
    vec2 pos = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    v_uv = pos;
    gl_Position = vec4(pos * 2.0 - 1.0, 0.0, 1.0);
}
