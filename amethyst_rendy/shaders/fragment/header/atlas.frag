#ifndef ATLAS_FRAG
#define ATLAS_FRAG

struct AtlasSlot {
    uint layer;
    uvec2 location; // integer location on size^2 grid
    uint size; // power of two slot/grid size
};

vec2 atlas_tex_coord(vec2 sub_tex_coord, AtlasSlot slot) {
    // TODO: actually implement atlas positioning
    return sub_tex_coord;
}

#endif