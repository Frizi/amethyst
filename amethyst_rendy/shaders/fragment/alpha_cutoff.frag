#version 450

#include "header/math.frag"

layout(std140, set = 1, binding = 0) uniform Material {
    UvOffset uv_offset;
    float alpha_cutoff;
};

layout(set = 1, binding = 1) uniform sampler2D albedo;

layout(location = 0) in VertexData {
    vec2 tex_coord;
} vertex;

void main() {
    float alpha = texture(albedo, tex_coords(vertex.tex_coord, uv_offset)).w;
    if(alpha < alpha_cutoff) discard;
}
