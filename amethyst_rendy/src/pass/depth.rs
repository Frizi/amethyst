use super::base_3d::*;
use crate::{mtl::TexAlbedo, skinning::JointCombined};
use rendy::{
    mesh::{AsVertex, Position, TexCoord, VertexFormat},
    shader::SpirvShader,
};

/// Implementation of `Base3DPassDef` for Physically-based (PBR) rendering pass.
#[derive(Debug)]
pub struct DepthPassDef;
impl Base3DPassDef for DepthPassDef {
    const NAME: &'static str = "Pbr";
    type TextureSet = TexAlbedo;
    fn vertex_shader() -> &'static SpirvShader {
        &super::POS_TEX_VERTEX
    }
    fn vertex_skinned_shader() -> &'static SpirvShader {
        &super::POS_TEX_SKIN_VERTEX
    }
    fn fragment_shader() -> &'static SpirvShader {
        &super::ALPHA_CUTOFF_FRAGMENT
    }
    fn base_format() -> Vec<VertexFormat> {
        vec![Position::vertex(), TexCoord::vertex()]
    }
    fn skinned_format() -> Vec<VertexFormat> {
        vec![
            Position::vertex(),
            TexCoord::vertex(),
            JointCombined::vertex(),
        ]
    }

    fn num_colors() -> usize {
        0
    }
}

/// Describes a 3d opaque depth prepass
pub type DrawDepthDesc<B> = DrawBase3DDesc<B, DepthPassDef>;
/// Draws a 3d opaque depth prepass
pub type DrawDepth<B> = DrawBase3D<B, DepthPassDef>;
