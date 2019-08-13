#version 450

layout(std140, set = 0, binding = 0) uniform ViewArgs {
    mat4 world_to_view;
};

layout(location = 0) in vec3 position;
layout(location = 1) in mat4 model; // instance rate

void main() {
    gl_Position = world_to_view * model * vec4(position, 1.0);
}
