use crate::{
    batch::{GroupIterator, OrderedTwoLevelBatch, TwoLevelBatch},
    mtl::{FullTextureSet, Material, StaticTextureSet},
    pipeline::{PipelineDescBuilder, PipelinesBuilder},
    pod::{SkinnedVertexArgs, VertexArgs},
    resources::Tint,
    skinning::JointTransforms,
    submodules::{DynamicVertexBuffer, EnvironmentSub, MaterialId, MaterialSub, SkinningSub},
    types::{Backend, Mesh},
    util,
    visibility::{VisOpaque, VisOpaqueCutoff, VisTransparent, VisibilityDef},
};
use amethyst_assets::{AssetStorage, Handle};
use amethyst_core::{
    ecs::{Join, Read, ReadStorage, SystemData, World},
    transform::Transform,
};
use derivative::Derivative;
use rendy::{
    command::{QueueId, RenderPassEncoder},
    factory::Factory,
    graph::{
        render::{PrepareResult, RenderGroup, RenderGroupDesc},
        DescBuilder, GraphContext, ImageAccess, ImageId, NodeBuffer, NodeImage,
    },
    hal::{self, device::Device, image, pso},
    mesh::{AsVertex, VertexFormat},
    resource::{Escape, Handle as RendyHandle, ImageView, ImageViewInfo, Sampler},
    shader::{Shader, SpirvShader},
};
use smallvec::SmallVec;
use std::marker::PhantomData;

macro_rules! profile_scope_impl {
    ($string:expr) => {
        #[cfg(feature = "profiler")]
        let _profile_scope = thread_profiler::ProfileScope::new(format!(
            "{} {}: {}",
            module_path!(),
            <T as Base3DPassDef>::NAME,
            $string
        ));
    };
}

/// Define drawing opaque 3d meshes with specified shaders and texture set
pub trait Base3DPassDef: 'static + std::fmt::Debug + Send + Sync {
    /// The human readable name of this pass
    const NAME: &'static str;

    /// The [mtl::StaticTextureSet] type implementation for this pass
    type TextureSet: for<'a> StaticTextureSet<'a>;

    /// Returns the vertex `SpirvShader` which will be used for this pass
    fn vertex_shader() -> &'static SpirvShader;

    /// Returns the vertex `SpirvShader` which will be used for this pass on skinned meshes
    fn vertex_skinned_shader() -> &'static SpirvShader;

    /// TODO: actually need separate shaders for cutout and not cutout

    /// Returns the fragment `SpirvShader` which will be used for this pass
    fn fragment_shader() -> &'static SpirvShader;

    /// Returns the `VertexFormat` of this pass
    fn base_format() -> Vec<VertexFormat>;

    /// Returns the `VertexFormat` of this pass for skinned meshes
    fn skinned_format() -> Vec<VertexFormat>;

    /// Returns the number of color attachments
    fn num_colors() -> usize {
        1
    }
}

/// Draw opaque 3d meshes with specified shaders and texture set
#[derive(Clone, Derivative)]
#[derivative(Debug(bound = ""), Default(bound = ""))]
pub struct DrawBase3DDesc<B: Backend, T: Base3DPassDef> {
    skinning: bool,
    shadow_map: Option<ImageId>,
    marker: PhantomData<(B, T)>,
}

impl<B: Backend, T: Base3DPassDef> DrawBase3DDesc<B, T> {
    /// Create pass in default configuration
    pub fn new() -> Self {
        Default::default()
    }

    /// Create pass in with vertex skinning enabled
    pub fn skinned() -> Self {
        Self {
            skinning: true,
            ..Default::default()
        }
    }

    /// Add shadows using specified image as shadow map. `None` disables shadows.
    pub fn with_shadow_map(mut self, shadow_map: Option<ImageId>) -> Self {
        self.shadow_map = shadow_map;
        self
    }

    /// Create pass in with vertex skinning enabled if true is passed
    pub fn with_skinning(mut self, skinned: bool) -> Self {
        self.skinning = skinned;
        self
    }
}

impl<B: Backend, T: Base3DPassDef> RenderGroupDesc<B, World> for DrawBase3DDesc<B, T> {
    fn images(&self) -> Vec<ImageAccess> {
        let mut vec = Vec::new();
        if self.shadow_map.is_some() {
            vec.push(ImageAccess {
                access: image::Access::SHADER_READ,
                usage: image::Usage::SAMPLED,
                layout: image::Layout::ShaderReadOnlyOptimal,
                stages: pso::PipelineStage::FRAGMENT_SHADER,
            });
        }
        vec
    }

    fn colors(&self) -> usize {
        T::num_colors()
    }

    fn builder(self) -> DescBuilder<B, World, Self>
    where
        Self: Sized,
    {
        let shadow_map = self.shadow_map.clone();

        let mut builder = DescBuilder::new(self);
        if let Some(shadow_map) = shadow_map {
            builder.add_image(shadow_map);
        };
        builder
    }

    fn build(
        self,
        ctx: &GraphContext<B>,
        factory: &mut Factory<B>,
        _queue: QueueId,
        _aux: &World,
        framebuffer_width: u32,
        framebuffer_height: u32,
        subpass: hal::pass::Subpass<'_, B>,
        _buffers: Vec<NodeBuffer>,
        mut images: Vec<NodeImage>,
    ) -> Result<Box<dyn RenderGroup<B, World>>, failure::Error> {
        profile_scope_impl!("build");

        let env = EnvironmentSub::new(
            factory,
            [
                hal::pso::ShaderStageFlags::VERTEX,
                hal::pso::ShaderStageFlags::FRAGMENT,
            ],
        )?;
        let materials = MaterialSub::new(factory)?;
        let skinning = SkinningSub::new(factory)?;

        let mut vertex_format_base = T::base_format();
        let mut vertex_format_skinned = T::skinned_format();

        let (mut pipelines, pipeline_layout) = build_pipelines::<B, T>(
            factory,
            subpass,
            framebuffer_width,
            framebuffer_height,
            &vertex_format_base,
            &vertex_format_skinned,
            self.skinning,
            false,
            vec![
                env.raw_layout(),
                materials.raw_layout(),
                skinning.raw_layout(),
            ],
        )?;

        vertex_format_base.sort();
        vertex_format_skinned.sort();

        let shadow_map = if self.shadow_map.is_some() {
            let node_image = images.pop().expect("Shadow map image not passed");

            // this can probably be made into a submodule on it's own
            let image = ctx.get_image(node_image.id).expect("Image does not exist");
            let sampler = factory.get_sampler(image::SamplerInfo::new(
                image::Filter::Linear,
                image::WrapMode::Clamp,
            ))?;
            let image_view = factory.create_image_view(
                image.clone(),
                ImageViewInfo {
                    view_kind: image::ViewKind::D2,
                    format: image.format(),
                    swizzle: hal::format::Swizzle::NO,
                    range: node_image.range.clone(),
                },
            )?;

            Some((node_image, sampler, image_view))
        } else {
            None
        };

        Ok(Box::new(DrawBase3D::<B, T> {
            pipeline_basic: pipelines.remove(0),
            pipeline_skinned: pipelines.pop(),
            pipeline_layout,
            static_batches: Default::default(),
            skinned_batches: Default::default(),
            vertex_format_base,
            vertex_format_skinned,
            env,
            materials,
            skinning,
            models: DynamicVertexBuffer::new(),
            skinned_models: DynamicVertexBuffer::new(),
            shadow_map,
            marker: PhantomData,
        }))
    }
}

/// Base implementation of a 3D render pass which can be consumed by actual 3D render passes,
/// such as [pass::pbr::DrawPbr]
#[derive(Derivative)]
#[derivative(Debug(bound = ""))]
pub struct DrawBase3D<B: Backend, T: Base3DPassDef> {
    pipeline_basic: B::GraphicsPipeline,
    pipeline_skinned: Option<B::GraphicsPipeline>,
    pipeline_layout: B::PipelineLayout,
    static_batches: TwoLevelBatch<MaterialId, u32, SmallVec<[VertexArgs; 4]>>,
    skinned_batches: TwoLevelBatch<MaterialId, u32, SmallVec<[SkinnedVertexArgs; 4]>>,
    vertex_format_base: Vec<VertexFormat>,
    vertex_format_skinned: Vec<VertexFormat>,
    env: EnvironmentSub<B>,
    materials: MaterialSub<B, T::TextureSet>,
    skinning: SkinningSub<B>,
    models: DynamicVertexBuffer<B, VertexArgs>,
    skinned_models: DynamicVertexBuffer<B, SkinnedVertexArgs>,
    shadow_map: Option<(NodeImage, RendyHandle<Sampler<B>>, Escape<ImageView<B>>)>,
    marker: PhantomData<T>,
}

impl<B: Backend, T: Base3DPassDef> RenderGroup<B, World> for DrawBase3D<B, T> {
    fn prepare(
        &mut self,
        factory: &Factory<B>,
        _queue: QueueId,
        index: usize,
        _subpass: hal::pass::Subpass<'_, B>,
        resources: &World,
    ) -> PrepareResult {
        profile_scope_impl!("prepare opaque");

        let (
            mesh_storage,
            vis_opaque,
            vis_opaque_cutoff,
            meshes,
            materials,
            transforms,
            joints,
            tints,
        ) = <(
            Read<'_, AssetStorage<Mesh>>,
            Read<'_, VisOpaque>,
            Read<'_, VisOpaqueCutoff>,
            ReadStorage<'_, Handle<Mesh>>,
            ReadStorage<'_, Handle<Material>>,
            ReadStorage<'_, Transform>,
            ReadStorage<'_, JointTransforms>,
            ReadStorage<'_, Tint>,
        )>::fetch(resources);

        // Prepare environment
        self.env
            .process(factory, index, resources, self.shadow_map.as_ref());
        self.materials.maintain();

        self.static_batches.clear_inner();
        self.skinned_batches.clear_inner();

        let materials_ref = &mut self.materials;
        let skinning_ref = &mut self.skinning;
        let statics_ref = &mut self.static_batches;
        let skinned_ref = &mut self.skinned_batches;

        let mut static_input = (&materials, &meshes, &transforms, tints.maybe(), !&joints).join();
        let mut skinned_input = (&materials, &meshes, &transforms, tints.maybe(), &joints).join();

        // We don't care about cutoff here, as this is usually rendered after depth prepass
        let all_opaques = || {
            vis_opaque
                .entities()
                .iter()
                .chain(vis_opaque_cutoff.entities())
        };

        {
            profile_scope_impl!("prepare");
            all_opaques()
                .filter_map(|e| static_input.get_unchecked(e.id()))
                .map(|(mat, mesh, tform, tint, _)| {
                    ((mat, mesh.id()), VertexArgs::from_object_data(tform, tint))
                })
                .for_each_group(|(mat, mesh_id), data| {
                    if mesh_storage.contains_id(mesh_id) {
                        if let Some((mat, _)) = materials_ref.insert(factory, resources, mat) {
                            statics_ref.insert(mat, mesh_id, data.drain(..));
                        }
                    }
                });
        }
        if self.pipeline_skinned.is_some() {
            profile_scope_impl!("prepare_skinning");
            all_opaques()
                .filter_map(|e| skinned_input.get_unchecked(e.id()))
                .map(|(mat, mesh, tform, tint, joints)| {
                    (
                        (mat, mesh.id()),
                        SkinnedVertexArgs::from_object_data(
                            tform,
                            tint,
                            skinning_ref.insert(joints),
                        ),
                    )
                })
                .for_each_group(|(mat, mesh_id), data| {
                    if mesh_storage.contains_id(mesh_id) {
                        if let Some((mat, _)) = materials_ref.insert(factory, resources, mat) {
                            skinned_ref.insert(mat, mesh_id, data.drain(..));
                        }
                    }
                });
        };

        {
            profile_scope_impl!("write");

            self.static_batches.prune();
            self.skinned_batches.prune();

            self.models.write(
                factory,
                index,
                self.static_batches.count() as u64,
                self.static_batches.data(),
            );

            self.skinned_models.write(
                factory,
                index,
                self.skinned_batches.count() as u64,
                self.skinned_batches.data(),
            );
            self.skinning.commit(factory, index);
        }
        PrepareResult::DrawRecord
    }

    fn draw_inline(
        &mut self,
        mut encoder: RenderPassEncoder<'_, B>,
        index: usize,
        _subpass: hal::pass::Subpass<'_, B>,
        resources: &World,
    ) {
        profile_scope_impl!("draw opaque");

        let mesh_storage = <Read<'_, AssetStorage<Mesh>>>::fetch(resources);
        let models_loc = self.vertex_format_base.len() as u32;
        let skin_models_loc = self.vertex_format_skinned.len() as u32;

        encoder.bind_graphics_pipeline(&self.pipeline_basic);
        self.env.bind(index, &self.pipeline_layout, 0, &mut encoder);

        if self.models.bind(index, models_loc, 0, &mut encoder) {
            let mut instances_drawn = 0;
            for (&mat_id, batches) in self.static_batches.iter() {
                if self.materials.loaded(mat_id) {
                    self.materials
                        .bind(&self.pipeline_layout, 1, mat_id, &mut encoder);
                    for (mesh_id, batch_data) in batches {
                        debug_assert!(mesh_storage.contains_id(*mesh_id));
                        if let Some(mesh) =
                            B::unwrap_mesh(unsafe { mesh_storage.get_by_id_unchecked(*mesh_id) })
                        {
                            if let Err(error) = mesh.bind_and_draw(
                                0,
                                &self.vertex_format_base,
                                instances_drawn..instances_drawn + batch_data.len() as u32,
                                &mut encoder,
                            ) {
                                log::warn!(
                                    "Trying to draw a mesh that lacks {:?} vertex attributes. Pass {} requires attributes {:?}.",
                                    error.not_found.attributes,
                                    T::NAME,
                                    T::base_format(),
                                );
                            }
                        }
                        instances_drawn += batch_data.len() as u32;
                    }
                }
            }
        }

        if let Some(pipeline_skinned) = self.pipeline_skinned.as_ref() {
            encoder.bind_graphics_pipeline(pipeline_skinned);

            if self
                .skinned_models
                .bind(index, skin_models_loc, 0, &mut encoder)
            {
                self.skinning
                    .bind(index, &self.pipeline_layout, 2, &mut encoder);

                let mut instances_drawn = 0;
                for (&mat_id, batches) in self.skinned_batches.iter() {
                    if self.materials.loaded(mat_id) {
                        self.materials
                            .bind(&self.pipeline_layout, 1, mat_id, &mut encoder);
                        for (mesh_id, batch_data) in batches {
                            debug_assert!(mesh_storage.contains_id(*mesh_id));
                            if let Some(mesh) = B::unwrap_mesh(unsafe {
                                mesh_storage.get_by_id_unchecked(*mesh_id)
                            }) {
                                if let Err(error) = mesh.bind_and_draw(
                                    0,
                                    &self.vertex_format_skinned,
                                    instances_drawn..instances_drawn + batch_data.len() as u32,
                                    &mut encoder,
                                ) {
                                    log::warn!(
                                    "Trying to draw a mesh that lacks {:?} vertex attributes. Pass {} requires attributes {:?}.",
                                    error.not_found.attributes,
                                    T::NAME,
                                    T::base_format(),
                                );
                                }
                            }
                            instances_drawn += batch_data.len() as u32;
                        }
                    }
                }
            }
        }
    }

    fn dispose(mut self: Box<Self>, factory: &mut Factory<B>, _aux: &World) {
        profile_scope_impl!("dispose");
        unsafe {
            factory
                .device()
                .destroy_graphics_pipeline(self.pipeline_basic);
            if let Some(pipeline) = self.pipeline_skinned.take() {
                factory.device().destroy_graphics_pipeline(pipeline);
            }
            factory
                .device()
                .destroy_pipeline_layout(self.pipeline_layout);
        }
    }
}

/// Draw transparent mesh with physically based lighting
#[derive(Clone, Derivative)]
#[derivative(Debug(bound = ""), Default(bound = ""))]
pub struct DrawBase3DTransparentDesc<B: Backend, T: Base3DPassDef> {
    skinning: bool,
    marker: PhantomData<(B, T)>,
}

impl<B: Backend, T: Base3DPassDef> DrawBase3DTransparentDesc<B, T> {
    /// Create pass in default configuration
    pub fn new() -> Self {
        Self {
            skinning: false,
            marker: PhantomData,
        }
    }

    /// Create pass in with vertex skinning enabled
    pub fn skinned() -> Self {
        Self {
            skinning: true,
            marker: PhantomData,
        }
    }

    /// Create pass in with vertex skinning enabled if true is passed
    pub fn with_skinning(mut self, skinned: bool) -> Self {
        self.skinning = skinned;
        self
    }
}

impl<B: Backend, T: Base3DPassDef> RenderGroupDesc<B, World> for DrawBase3DTransparentDesc<B, T> {
    fn build(
        self,
        _ctx: &GraphContext<B>,
        factory: &mut Factory<B>,
        _queue: QueueId,
        _aux: &World,
        framebuffer_width: u32,
        framebuffer_height: u32,
        subpass: hal::pass::Subpass<'_, B>,
        _buffers: Vec<NodeBuffer>,
        _images: Vec<NodeImage>,
    ) -> Result<Box<dyn RenderGroup<B, World>>, failure::Error> {
        let env = EnvironmentSub::new(
            factory,
            [
                hal::pso::ShaderStageFlags::VERTEX,
                hal::pso::ShaderStageFlags::FRAGMENT,
            ],
        )?;

        let materials = MaterialSub::new(factory)?;
        let skinning = SkinningSub::new(factory)?;

        let mut vertex_format_base = T::base_format();
        let mut vertex_format_skinned = T::skinned_format();

        let (mut pipelines, pipeline_layout) = build_pipelines::<B, T>(
            factory,
            subpass,
            framebuffer_width,
            framebuffer_height,
            &vertex_format_base,
            &vertex_format_skinned,
            self.skinning,
            true,
            vec![
                env.raw_layout(),
                materials.raw_layout(),
                skinning.raw_layout(),
            ],
        )?;

        vertex_format_base.sort();
        vertex_format_skinned.sort();

        Ok(Box::new(DrawBase3DTransparent::<B, T> {
            pipeline_basic: pipelines.remove(0),
            pipeline_skinned: pipelines.pop(),
            pipeline_layout,
            static_batches: Default::default(),
            skinned_batches: Default::default(),
            vertex_format_base,
            vertex_format_skinned,
            env,
            materials,
            skinning,
            models: DynamicVertexBuffer::new(),
            skinned_models: DynamicVertexBuffer::new(),
            change: Default::default(),
            marker: PhantomData,
        }))
    }
}

/// Draw transparent mesh with physically based lighting
#[derive(Derivative)]
#[derivative(Debug(bound = ""))]
pub struct DrawBase3DTransparent<B: Backend, T: Base3DPassDef> {
    pipeline_basic: B::GraphicsPipeline,
    pipeline_skinned: Option<B::GraphicsPipeline>,
    pipeline_layout: B::PipelineLayout,
    static_batches: OrderedTwoLevelBatch<MaterialId, u32, VertexArgs>,
    skinned_batches: OrderedTwoLevelBatch<MaterialId, u32, SkinnedVertexArgs>,
    vertex_format_base: Vec<VertexFormat>,
    vertex_format_skinned: Vec<VertexFormat>,
    env: EnvironmentSub<B>,
    materials: MaterialSub<B, FullTextureSet>,
    skinning: SkinningSub<B>,
    models: DynamicVertexBuffer<B, VertexArgs>,
    skinned_models: DynamicVertexBuffer<B, SkinnedVertexArgs>,
    change: util::ChangeDetection,
    marker: PhantomData<(T)>,
}

impl<B: Backend, T: Base3DPassDef> RenderGroup<B, World> for DrawBase3DTransparent<B, T> {
    fn prepare(
        &mut self,
        factory: &Factory<B>,
        _queue: QueueId,
        index: usize,
        _subpass: hal::pass::Subpass<'_, B>,
        resources: &World,
    ) -> PrepareResult {
        profile_scope_impl!("prepare transparent");

        let (mesh_storage, vis_transparent, meshes, materials, transforms, joints, tints) =
            <(
                Read<'_, AssetStorage<Mesh>>,
                Read<'_, VisTransparent>,
                ReadStorage<'_, Handle<Mesh>>,
                ReadStorage<'_, Handle<Material>>,
                ReadStorage<'_, Transform>,
                ReadStorage<'_, JointTransforms>,
                ReadStorage<'_, Tint>,
            )>::fetch(resources);

        // Prepare environment
        self.env.process(factory, index, resources, None);
        self.materials.maintain();

        self.static_batches.swap_clear();
        self.skinned_batches.swap_clear();

        let materials_ref = &mut self.materials;
        let skinning_ref = &mut self.skinning;
        let statics_ref = &mut self.static_batches;
        let skinned_ref = &mut self.skinned_batches;
        let mut changed = false;

        let mut joined = ((&materials, &meshes, &transforms, tints.maybe()), !&joints).join();
        vis_transparent
            .entities()
            .iter()
            .filter_map(|e| joined.get_unchecked(e.id()))
            .map(|((mat, mesh, tform, tint), _)| {
                ((mat, mesh.id()), VertexArgs::from_object_data(tform, tint))
            })
            .for_each_group(|(mat, mesh_id), data| {
                if mesh_storage.contains_id(mesh_id) {
                    if let Some((mat, this_changed)) = materials_ref.insert(factory, resources, mat)
                    {
                        changed = changed || this_changed;
                        statics_ref.insert(mat, mesh_id, data.drain(..));
                    }
                }
            });

        if self.pipeline_skinned.is_some() {
            let mut joined = (&materials, &meshes, &transforms, tints.maybe(), &joints).join();

            vis_transparent
                .entities()
                .iter()
                .filter_map(|e| joined.get_unchecked(e.id()))
                .map(|(mat, mesh, tform, tint, joints)| {
                    (
                        (mat, mesh.id()),
                        SkinnedVertexArgs::from_object_data(
                            tform,
                            tint,
                            skinning_ref.insert(joints),
                        ),
                    )
                })
                .for_each_group(|(mat, mesh_id), data| {
                    if mesh_storage.contains_id(mesh_id) {
                        if let Some((mat, this_changed)) =
                            materials_ref.insert(factory, resources, mat)
                        {
                            changed = changed || this_changed;
                            skinned_ref.insert(mat, mesh_id, data.drain(..));
                        }
                    }
                });
        }

        self.models.write(
            factory,
            index,
            self.static_batches.count() as u64,
            Some(self.static_batches.data()),
        );

        self.skinned_models.write(
            factory,
            index,
            self.skinned_batches.count() as u64,
            Some(self.skinned_batches.data()),
        );

        self.skinning.commit(factory, index);

        changed = changed || self.static_batches.changed();
        changed = changed || self.skinned_batches.changed();

        self.change.prepare_result(index, changed)
    }

    fn draw_inline(
        &mut self,
        mut encoder: RenderPassEncoder<'_, B>,
        index: usize,
        _subpass: hal::pass::Subpass<'_, B>,
        resources: &World,
    ) {
        profile_scope_impl!("draw transparent");

        let mesh_storage = <Read<'_, AssetStorage<Mesh>>>::fetch(resources);
        let layout = &self.pipeline_layout;
        let encoder = &mut encoder;

        let models_loc = self.vertex_format_base.len() as u32;
        let skin_models_loc = self.vertex_format_skinned.len() as u32;

        encoder.bind_graphics_pipeline(&self.pipeline_basic);
        self.env.bind(index, layout, 0, encoder);

        if self.models.bind(index, models_loc, 0, encoder) {
            for (&mat, batches) in self.static_batches.iter() {
                if self.materials.loaded(mat) {
                    self.materials.bind(layout, 1, mat, encoder);
                    for (mesh, range) in batches {
                        debug_assert!(mesh_storage.contains_id(*mesh));
                        if let Some(mesh) =
                            B::unwrap_mesh(unsafe { mesh_storage.get_by_id_unchecked(*mesh) })
                        {
                            if let Err(error) = mesh.bind_and_draw(
                                0,
                                &self.vertex_format_base,
                                range.clone(),
                                encoder,
                            ) {
                                log::warn!(
                                    "Trying to draw a mesh that lacks {:?} vertex attributes. Pass {} requires attributes {:?}.",
                                    error.not_found.attributes,
                                    T::NAME,
                                    T::base_format(),
                                );
                            }
                        }
                    }
                }
            }
        }

        if let Some(pipeline_skinned) = self.pipeline_skinned.as_ref() {
            encoder.bind_graphics_pipeline(pipeline_skinned);

            if self.skinned_models.bind(index, skin_models_loc, 0, encoder) {
                self.skinning.bind(index, layout, 2, encoder);
                for (&mat, batches) in self.skinned_batches.iter() {
                    if self.materials.loaded(mat) {
                        self.materials.bind(layout, 1, mat, encoder);
                        for (mesh, range) in batches {
                            debug_assert!(mesh_storage.contains_id(*mesh));
                            if let Some(mesh) =
                                B::unwrap_mesh(unsafe { mesh_storage.get_by_id_unchecked(*mesh) })
                            {
                                if let Err(error) = mesh.bind_and_draw(
                                    0,
                                    &self.vertex_format_skinned,
                                    range.clone(),
                                    encoder,
                                ) {
                                    log::warn!(
                                        "Trying to draw a skinned mesh that lacks {:?} vertex attributes. Pass {} requires attributes {:?}.",
                                        error.not_found.attributes,
                                        T::NAME,
                                        T::skinned_format(),
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    fn dispose(mut self: Box<Self>, factory: &mut Factory<B>, _aux: &World) {
        unsafe {
            factory
                .device()
                .destroy_graphics_pipeline(self.pipeline_basic);
            if let Some(pipeline) = self.pipeline_skinned.take() {
                factory.device().destroy_graphics_pipeline(pipeline);
            }
            factory
                .device()
                .destroy_pipeline_layout(self.pipeline_layout);
        }
    }
}

fn build_pipelines<B: Backend, T: Base3DPassDef>(
    factory: &Factory<B>,
    subpass: hal::pass::Subpass<'_, B>,
    framebuffer_width: u32,
    framebuffer_height: u32,
    vertex_format_base: &[VertexFormat],
    vertex_format_skinned: &[VertexFormat],
    skinning: bool,
    transparent: bool,
    layouts: Vec<&B::DescriptorSetLayout>,
) -> Result<(Vec<B::GraphicsPipeline>, B::PipelineLayout), failure::Error> {
    let pipeline_layout = unsafe {
        factory
            .device()
            .create_pipeline_layout(layouts, None as Option<(_, _)>)
    }?;

    let vertex_desc = vertex_format_base
        .iter()
        .map(|f| (f.clone(), pso::VertexInputRate::Vertex))
        .chain(Some((
            VertexArgs::vertex(),
            pso::VertexInputRate::Instance(1),
        )))
        .collect::<Vec<_>>();

    let shader_vertex_basic = unsafe { T::vertex_shader().module(factory).unwrap() };
    let shader_fragment = unsafe { T::fragment_shader().module(factory).unwrap() };
    let pipe_desc = PipelineDescBuilder::new()
        .with_vertex_desc(&vertex_desc)
        .with_shaders(util::simple_shader_set(
            &shader_vertex_basic,
            Some(&shader_fragment),
        ))
        .with_layout(&pipeline_layout)
        .with_subpass(subpass)
        .with_framebuffer_size(framebuffer_width, framebuffer_height)
        .with_face_culling(pso::Face::BACK)
        .with_depth_test(Some(pso::DepthTest {
            fun: pso::Comparison::LessEqual,
            write: !transparent,
        }))
        .with_blend_targets(
            std::iter::repeat(pso::ColorBlendDesc {
                mask: pso::ColorMask::ALL,
                blend: if transparent {
                    Some(pso::BlendState::PREMULTIPLIED_ALPHA)
                } else {
                    None
                },
            })
            .take(T::num_colors())
            .collect(),
        );

    let pipelines = if skinning {
        let shader_vertex_skinned = unsafe { T::vertex_skinned_shader().module(factory).unwrap() };

        let vertex_desc = vertex_format_skinned
            .iter()
            .map(|f| (f.clone(), pso::VertexInputRate::Vertex))
            .chain(Some((
                SkinnedVertexArgs::vertex(),
                pso::VertexInputRate::Instance(1),
            )))
            .collect::<Vec<_>>();

        let pipe = PipelinesBuilder::new()
            .with_pipeline(pipe_desc.clone())
            .with_child_pipeline(
                0,
                pipe_desc
                    .with_vertex_desc(&vertex_desc)
                    .with_shaders(util::simple_shader_set(
                        &shader_vertex_skinned,
                        Some(&shader_fragment),
                    )),
            )
            .build(factory, None);

        unsafe {
            factory.destroy_shader_module(shader_vertex_skinned);
        }

        pipe
    } else {
        PipelinesBuilder::new()
            .with_pipeline(pipe_desc)
            .build(factory, None)
    };

    unsafe {
        factory.destroy_shader_module(shader_vertex_basic);
        factory.destroy_shader_module(shader_fragment);
    }

    match pipelines {
        Err(e) => {
            unsafe {
                factory.device().destroy_pipeline_layout(pipeline_layout);
            }
            Err(e)
        }
        Ok(pipelines) => Ok((pipelines, pipeline_layout)),
    }
}
