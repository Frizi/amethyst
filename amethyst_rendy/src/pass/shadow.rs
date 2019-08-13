use crate::{
    batch::{GroupIterator, OneLevelBatch},
    camera::Orthographic,
    light::Light,
    pipeline::{PipelineDescBuilder, PipelinesBuilder},
    pod::ViewArgs,
    skinning::{JointCombined, JointTransforms},
    submodules::{DynamicUniform, DynamicVertexBuffer, SkinningSub},
    transparent::Transparent,
    types::{Backend, Mesh},
    util,
};
use amethyst_assets::{AssetStorage, Handle};
use amethyst_core::{
    ecs::{Join, Read, ReadStorage, SystemData, World},
    math::{convert, Matrix4, Point3, Vector3},
    transform::Transform,
    Hidden, HiddenPropagate,
};
use derivative::Derivative;
use glsl_layout::*;
use rendy::{
    command::{QueueId, RenderPassEncoder},
    factory::Factory,
    graph::{
        render::{PrepareResult, RenderGroup, RenderGroupDesc},
        GraphContext, NodeBuffer, NodeImage,
    },
    hal::{self, device::Device, pso},
    mesh::{AsVertex, Model, Position, VertexFormat},
    shader::Shader,
};
use std::marker::PhantomData;

#[cfg(feature = "profiler")]
use thread_profiler::profile_scope;

/// Draw opaque 3d meshes with specified shaders and texture set
#[derive(Clone, Derivative)]
#[derivative(Debug(bound = ""), Default(bound = ""))]
pub struct DrawShadows3DDesc<B: Backend> {
    skinning: bool,
    marker: PhantomData<B>,
}

impl<B: Backend> DrawShadows3DDesc<B> {
    /// Create pass in default configuration
    pub fn new() -> Self {
        Default::default()
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

impl<B: Backend> RenderGroupDesc<B, World> for DrawShadows3DDesc<B> {
    fn colors(&self) -> usize {
        0
    }

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
        #[cfg(feature = "profiler")]
        profile_scope!("build DrawShadows3D");

        let env = DynamicUniform::new(factory, pso::ShaderStageFlags::VERTEX)?;
        let skinning = SkinningSub::new(factory)?;

        let (mut pipelines, pipeline_layout) = build_pipelines::<B>(
            factory,
            subpass,
            framebuffer_width,
            framebuffer_height,
            &[Position::vertex()],
            &[Position::vertex(), JointCombined::vertex()],
            // self.skinning,
            vec![env.raw_layout(), skinning.raw_layout()],
        )?;

        Ok(Box::new(DrawShadows3D::<B> {
            pipeline_basic: pipelines.remove(0),
            pipeline_skinned: pipelines.pop(),
            pipeline_layout,
            static_batches: Default::default(),
            // skinned_batches: Default::default(),
            skinning,
            models: DynamicVertexBuffer::new(),
            env,
            // skinned_models: DynamicVertexBuffer::new(),
        }))
    }
}

/// Base implementation of a 3D render pass which can be consumed by actual 3D render passes,
/// such as [pass::pbr::DrawPbr]
#[derive(Derivative)]
#[derivative(Debug(bound = ""))]
pub struct DrawShadows3D<B: Backend> {
    pipeline_basic: B::GraphicsPipeline,
    pipeline_skinned: Option<B::GraphicsPipeline>,
    pipeline_layout: B::PipelineLayout,
    static_batches: OneLevelBatch<u32, Model>,
    // skinned_batches: OneLevelBatch<u32, SkinnedVertexArgs>,
    skinning: SkinningSub<B>,
    models: DynamicVertexBuffer<B, Model>,
    env: DynamicUniform<B, ViewArgs>,
    // skinned_models: DynamicVertexBuffer<B, SkinnedVertexArgs>,
}

impl<B: Backend> RenderGroup<B, World> for DrawShadows3D<B> {
    fn prepare(
        &mut self,
        factory: &Factory<B>,
        _queue: QueueId,
        index: usize,
        _subpass: hal::pass::Subpass<'_, B>,
        world: &World,
    ) -> PrepareResult {
        #[cfg(feature = "profiler")]
        profile_scope!("prepare");

        let (mesh_storage, lights, transparent, hiddens, hiddens_prop, meshes, transforms, joints) =
            <(
                Read<'_, AssetStorage<Mesh>>,
                ReadStorage<'_, Light>,
                ReadStorage<'_, Transparent>,
                ReadStorage<'_, Hidden>,
                ReadStorage<'_, HiddenPropagate>,
                ReadStorage<'_, Handle<Mesh>>,
                ReadStorage<'_, Transform>,
                ReadStorage<'_, JointTransforms>,
            )>::fetch(world);

        self.static_batches.clear_inner();
        // self.skinned_batches.clear_inner();

        let statics_ref = &mut self.static_batches;
        // let skinned_ref = &mut self.skinned_batches;
        // let skinning_ref = &mut self.skinning;

        // Prepare environment
        if let Some(light) = lights
            .join()
            .filter_map(|light| match light {
                Light::Directional(light) if light.cast_shadow => Some(light),
                _ => None,
            })
            .next()
        {
            let eye = Point3::new(0.0, 0.0, 0.0);
            let target = eye + light.direction;
            let light_view = Matrix4::<f32>::look_at_rh(&eye, &target, &Vector3::y());
            let light_proj = Orthographic::new(-20.0, 20.0, -20.0, 20.0, -200.0, 200.0)
                .as_matrix()
                .clone();
            let viewargs = ViewArgs::from_separate_matrices(light_proj, light_view);
            self.env.write(factory, index, viewargs.std140());
        } else {
            return PrepareResult::DrawRecord;
        };

        {
            #[cfg(feature = "profiler")]
            profile_scope!("prepare_rigid");
            (&meshes, &transforms, !&joints)
                .join()
                .map(|(mesh, transform, _)| {
                    let model: [[f32; 4]; 4] =
                        convert::<_, Matrix4<f32>>(*transform.global_matrix()).into();
                    ((mesh.id()), Model(model))
                })
                .for_each_group(|mesh_id, data| {
                    if mesh_storage.contains_id(mesh_id) {
                        statics_ref.insert(mesh_id, data.drain(..));
                    }
                });
        }
        // if self.pipeline_skinned.is_some() {
        //     #[cfg(feature = "profiler")]
        //     profile_scope!("prepare_skinning");

        //     let skinned_input = || (&meshes, &transforms, &joints);
        //     (skinned_input(), &visibility.visible_unordered)
        //         .join()
        //         .map(|((mat, mesh, tform, tint, joints), _)| {
        //             (
        //                 (mat, mesh.id()),
        //                 SkinnedVertexArgs::from_object_data(
        //                     tform,
        //                     tint,
        //                     skinning_ref.insert(joints),
        //                 ),
        //             )
        //         })
        //         .for_each_group(|(mat, mesh_id), data| {
        //             if mesh_storage.contains_id(mesh_id) {
        //                 skinned_ref.insert(mesh_id, data.drain(..));
        //             }
        //         });
        // };

        {
            #[cfg(feature = "profiler")]
            profile_scope!("write");

            self.static_batches.prune();
            // self.skinned_batches.prune();

            self.models.write(
                factory,
                index,
                self.static_batches.count() as u64,
                self.static_batches.data(),
            );

            // self.skinned_models.write(
            //     factory,
            //     index,
            //     self.skinned_batches.count() as u64,
            //     self.skinned_batches.data(),
            // );
            // self.skinning.commit(factory, index);
        }
        PrepareResult::DrawRecord
    }

    fn draw_inline(
        &mut self,
        mut encoder: RenderPassEncoder<'_, B>,
        index: usize,
        _subpass: hal::pass::Subpass<'_, B>,
        world: &World,
    ) {
        #[cfg(feature = "profiler")]
        profile_scope!("draw shadows");

        if self.static_batches.count() == 0 {
            return;
        }

        let vertex_format = &[Position::vertex()];

        let mesh_storage = <Read<'_, AssetStorage<Mesh>>>::fetch(world);

        encoder.bind_graphics_pipeline(&self.pipeline_basic);
        self.env.bind(index, &self.pipeline_layout, 0, &mut encoder);

        if self.models.bind(index, 1, 0, &mut encoder) {
            let mut instances_drawn = 0;
            for (&mesh_id, batch_data) in self.static_batches.iter() {
                debug_assert!(mesh_storage.contains_id(mesh_id));
                if let Some(mesh) =
                    B::unwrap_mesh(unsafe { mesh_storage.get_by_id_unchecked(mesh_id) })
                {
                    if let Err(error) = mesh.bind_and_draw(
                        0,
                        vertex_format,
                        instances_drawn..instances_drawn + batch_data.len() as u32,
                        &mut encoder,
                    ) {
                        log::warn!("Trying to draw a shadow of a mesh that lacks `Position` vertex attributes.");
                    }
                }
                instances_drawn += batch_data.len() as u32;
            }
        }

        // if let Some(pipeline_skinned) = self.pipeline_skinned.as_ref() {
        //     encoder.bind_graphics_pipeline(pipeline_skinned);

        //     if self
        //         .skinned_models
        //         .bind(index, skin_models_loc, 0, &mut encoder)
        //     {
        //         self.skinning
        //             .bind(index, &self.pipeline_layout, 2, &mut encoder);

        //         let mut instances_drawn = 0;
        //         for (&mat_id, batches) in self.skinned_batches.iter() {
        //             if self.materials.loaded(mat_id) {
        //                 self.materials
        //                     .bind(&self.pipeline_layout, 1, mat_id, &mut encoder);
        //                 for (mesh_id, batch_data) in batches {
        //                     debug_assert!(mesh_storage.contains_id(*mesh_id));
        //                     if let Some(mesh) = B::unwrap_mesh(unsafe {
        //                         mesh_storage.get_by_id_unchecked(*mesh_id)
        //                     }) {
        //                         mesh.bind_and_draw(
        //                             0,
        //                             &self.vertex_format_skinned,
        //                             instances_drawn..instances_drawn + batch_data.len() as u32,
        //                             &mut encoder,
        //                         )
        //                         .unwrap();
        //                     }
        //                     instances_drawn += batch_data.len() as u32;
        //                 }
        //             }
        //         }
        //     }
        // }
    }

    fn dispose(mut self: Box<Self>, factory: &mut Factory<B>, _aux: &World) {
        #[cfg(feature = "profiler")]
        profile_scope!("dispose");
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

fn build_pipelines<B: Backend>(
    factory: &Factory<B>,
    subpass: hal::pass::Subpass<'_, B>,
    framebuffer_width: u32,
    framebuffer_height: u32,
    vertex_format_base: &[VertexFormat],
    vertex_format_skinned: &[VertexFormat],
    // skinning: bool,
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
        .chain(Some((Model::vertex(), pso::VertexInputRate::Instance(1))))
        .collect::<Vec<_>>();

    let shader_vertex = unsafe { super::SIMPLE_MESH_VERTEX.module(factory).unwrap() };
    let pipe_desc = PipelineDescBuilder::new()
        .with_vertex_desc(&vertex_desc)
        .with_shaders(util::simple_shader_set(&shader_vertex, None))
        .with_layout(&pipeline_layout)
        .with_subpass(subpass)
        .with_framebuffer_size(framebuffer_width, framebuffer_height)
        .with_face_culling(pso::Face::BACK)
        .with_multisampling(Some(pso::Multisampling {
            rasterization_samples: 4,
            sample_shading: None,
            sample_mask: !0,
            alpha_coverage: false,
            alpha_to_one: false,
        }))
        .with_depth_test(Some(pso::DepthTest {
            fun: pso::Comparison::Less,
            write: true,
        }));

    // let pipelines = if skinning {
    //     let shader_vertex_skinned = unsafe { T::vertex_skinned_shader().module(factory).unwrap() };

    //     let vertex_desc = vertex_format_skinned
    //         .iter()
    //         .map(|f| (f.clone(), pso::VertexInputRate::Vertex))
    //         .chain(Some((
    //             SkinnedVertexArgs::vertex(),
    //             pso::VertexInputRate::Instance(1),
    //         )))
    //         .collect::<Vec<_>>();

    //     let pipe = PipelinesBuilder::new()
    //         .with_pipeline(pipe_desc.clone())
    //         .with_child_pipeline(
    //             0,
    //             pipe_desc
    //                 .with_vertex_desc(&vertex_desc)
    //                 .with_shaders(util::simple_shader_set(
    //                     &shader_vertex_skinned,
    //                     Some(&shader_fragment),
    //                 )),
    //         )
    //         .build(factory, None);

    //     unsafe {
    //         factory.destroy_shader_module(shader_vertex_skinned);
    //     }

    //     pipe
    // } else {
    // };

    let pipelines = PipelinesBuilder::new()
        .with_pipeline(pipe_desc)
        .build(factory, None);

    unsafe {
        factory.destroy_shader_module(shader_vertex);
        // factory.destroy_shader_module(shader_fragment);
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
