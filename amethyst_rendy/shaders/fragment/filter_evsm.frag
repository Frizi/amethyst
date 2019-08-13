#version 450

#include "header/evsm.frag"

layout(set = 0, binding = 0) uniform sampler2DMS depth;

layout(location = 0) in VertexData {
    vec2 tex_coord;
} vertex;

layout(location = 0) out vec4 out_color;

const int SHADOWAA_SAMPLES = 4;
vec4 shadow_depth_to_evsm(sampler2DMS shadow_map, ivec2 coords, float z_scale) {
    float sample_weight = 1.0 / float(SHADOWAA_SAMPLES);
    vec2 exponents = get_evsm_exponents(z_scale);

    
    // Simple average (box filter) for now
    vec4 average = vec4(0.0, 0.0, 0.0, 0.0);
    for (int i = 0; i < SHADOWAA_SAMPLES; ++i) {
        // Convert to EVSM representation
        float depth = texelFetch(shadow_map, coords, i).x;
        vec2 warped_depth = warp_depth(depth, exponents);
        average += sample_weight * vec4(warped_depth, warped_depth * warped_depth);
    }
    
    return average;
}

void main() {
    out_color = shadow_depth_to_evsm(depth, ivec2(gl_FragCoord), 1.0);
}
