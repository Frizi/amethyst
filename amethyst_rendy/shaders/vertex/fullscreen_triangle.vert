#version 450

const vec2 positions[3] = vec2[](
    vec2(-1.0, -1.0),
    vec2(-1.0, 3.0),
    vec2(3.0, -1.0)
);

layout(location = 0) out VertexData {
    vec2 tex_coord;
} vertex;

void main() {
    vec2 coord = positions[gl_VertexIndex];
    vertex.tex_coord = coord * 0.5 + 0.5;
    gl_Position = vec4(coord, 0.0, 1.0);
}
