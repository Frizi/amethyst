#ifndef SHADOW_MAP_FRAG
#define SHADOW_MAP_FRAG

#include "atlas.frag"

#ifndef MAX_DIR_SHADOWS_CONST_ID
    #define MAX_DIR_SHADOWS_CONST_ID 0
#endif

#ifndef MAX_SHADOW_CASCADES_CONST_ID
    #define MAX_SHADOW_CASCADES_CONST_ID (MAX_DIR_SHADOWS_CONST_ID + 1)
#endif

layout (constant_id = MAX_DIR_SHADOWS_CONST_ID) const int MAX_DIR_SHADOWS = 1;
layout (constant_id = MAX_SHADOW_CASCADES_CONST_ID) const int MAX_SHADOW_CASCADES = 4;

struct Cascade
{
    float interval_begin;
    float interval_end;
    // These are given in texture coordinate [0, 1] space
    vec3 scale;
    vec3 bias;
    AtlasSlot slot;
};


struct DirShadowData {
    mat4 view_to_light;
    Cascade cascades[MAX_SHADOW_CASCADES];
};

#if defined(SHADOWS_DESC_SET) && defined(SHADOWS_DESC_BINDING_OFFSET)
    layout(std140, set = SHADOWS_DESC_SET, binding = SHADOWS_DESC_BINDING_OFFSET) uniform Shadows {
        DirShadowData shadows[MAX_DIR_SHADOWS];
    };
    layout(set = SHADOWS_DESC_SET, binding = (SHADOWS_DESC_BINDING_OFFSET + 1)) uniform sampler2DArray shadow_map_atlas;
#endif

vec3 project_into_light_tex_coord(mat4 view_to_light, vec4 global_space_position) {
    vec4 position_light = view_to_light * global_space_position;
    return (position_light.xyz / position_light.w) * vec3(0.5, 0.5, 1.0) + vec3(0.5, 0.5, 0.0);
}

#endif
