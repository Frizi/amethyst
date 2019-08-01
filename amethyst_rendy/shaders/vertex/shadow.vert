#version 450

layout(std140, set = 0, binding = 0) uniform Projview {
    mat4 proj;
    mat4 view;
};

layout(location = 0) in vec3 position;
layout(location = 1) in mat4 model; // instance rate

void main() {
    gl_Position = proj * view * model * vec4(position, 1.0);
}
