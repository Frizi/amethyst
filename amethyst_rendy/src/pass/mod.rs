//! Passes and shaders implemented by amethyst

mod base_3d;
mod debug_lines;
mod flat;
mod flat2d;
mod fullscreen_triangle;
mod pbr;
mod shaded;
mod shadow;
mod skybox;

pub use self::{
    base_3d::*, debug_lines::*, flat::*, flat2d::*, fullscreen_triangle::*, pbr::*, shaded::*,
    shadow::*, skybox::*,
};

define_shaders! {
    static ref POS_TEX_VERTEX <- "../../compiled/vertex/pos_tex.vert.spv";
    static ref POS_TEX_SKIN_VERTEX <- "../../compiled/vertex/pos_tex_skin.vert.spv";
    static ref POS_NORM_TEX_VERTEX <- "../../compiled/vertex/pos_norm_tex.vert.spv";
    static ref POS_NORM_TEX_SKIN_VERTEX <- "../../compiled/vertex/pos_norm_tex_skin.vert.spv";
    static ref POS_NORM_TANG_TEX_VERTEX <- "../../compiled/vertex/pos_norm_tang_tex.vert.spv";
    static ref POS_NORM_TANG_TEX_SKIN_VERTEX <- "../../compiled/vertex/pos_norm_tang_tex_skin.vert.spv";
    static ref FLAT_FRAGMENT <- "../../compiled/fragment/flat.frag.spv";
    static ref SHADED_FRAGMENT <- "../../compiled/fragment/shaded.frag.spv";
    static ref PBR_FRAGMENT <- "../../compiled/fragment/pbr.frag.spv";
    static ref SPRITE_VERTEX <- "../../compiled/vertex/sprite.vert.spv";
    static ref SPRITE_FRAGMENT <- "../../compiled/fragment/sprite.frag.spv";
    static ref SKYBOX_VERTEX <- "../../compiled/vertex/skybox.vert.spv";
    static ref SKYBOX_FRAGMENT <- "../../compiled/fragment/skybox.frag.spv";
    static ref DEBUG_LINES_VERTEX <- "../../compiled/vertex/debug_lines.vert.spv";
    static ref DEBUG_LINES_FRAGMENT <- "../../compiled/fragment/debug_lines.frag.spv";
    static ref SIMPLE_MESH_VERTEX <- "../../compiled/vertex/simple_mesh.vert.spv";
    static ref SHADOW_SKIN_VERTEX <- "../../compiled/vertex/simple_mesh_skin.vert.spv";
    static ref FULLSCREEN_TRIANGLE_VERTEX <- "../../compiled/vertex/fullscreen_triangle.vert.spv";
    [pub(crate)] static ref FILTER_EVSM_FRAGMENT <- "../../compiled/fragment/filter_evsm.frag.spv";
}
