#version 450

layout(std140, set = 0, binding = 0) uniform ViewArgs {
    mat4 world_to_view;
};

layout(location = 0) in vec3 position;
layout(location = 1) in vec2 tex_coord;
layout(location = 2) in mat4 model; // instance rate

layout(location = 0) out VertexData {
    vec2 tex_coord;
} vertex;

void main() {
    vec4 vertex_position = model * vec4(position, 1.0);
    gl_Position = world_to_view * vertex_position;
    vertex.tex_coord = tex_coord;
}
