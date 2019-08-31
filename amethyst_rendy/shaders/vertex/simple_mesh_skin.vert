#version 450

layout(std140, set = 0, binding = 0) uniform ViewArgs {
    mat4 world_to_view;
};

layout(std430, set = 2, binding = 0) readonly buffer JointTransforms {
    mat4 joints[];
};

layout(location = 0) in vec3 position;
layout(location = 1) in uvec4 joint_ids;
layout(location = 2) in vec4 joint_weights;
layout(location = 3) in mat4 model; // instance rate
layout(location = 7) in uint joints_offset; // instance rate

void main() {
    mat4 joint_transform =
        joint_weights.x * joints[int(joints_offset + joint_ids.x)] +
        joint_weights.y * joints[int(joints_offset + joint_ids.y)] +
        joint_weights.z * joints[int(joints_offset + joint_ids.z)] +
        joint_weights.w * joints[int(joints_offset + joint_ids.w)];
    gl_Position = world_to_view * model * joint_transform * vec4(position, 1.0);
}