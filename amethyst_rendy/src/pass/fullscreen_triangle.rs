use crate::{
    pipeline::{PipelineDescBuilder, PipelinesBuilder},
    types::Backend,
    util,
};
use amethyst_core::ecs::World;
use rendy::{
    command::{QueueId, RenderPassEncoder},
    factory::Factory,
    graph::{
        render::{PrepareResult, RenderGroup, RenderGroupDesc},
        DescBuilder, GraphContext, ImageAccess, ImageId, NodeBuffer, NodeImage,
    },
    hal::{self, device::Device, image, pso},
    resource::{
        DescriptorSet, DescriptorSetLayout, Escape, Handle as RendyHandle, ImageView,
        ImageViewInfo, Sampler,
    },
    shader::{Shader, SpirvShader},
};

#[cfg(feature = "profiler")]
use thread_profiler::profile_scope;

/// Draw single fullscreen triangle with specified shader.
#[derive(Clone, Debug, PartialEq)]
pub struct DrawFullscreenTriangleDesc {
    image: ImageId,
    shader: SpirvShader,
}

impl DrawFullscreenTriangleDesc {
    /// Create instance of `DrawFullscreenTriangle` render group
    pub fn new(image: ImageId, shader: SpirvShader) -> Self {
        Self { image, shader }
    }
}

impl<B: Backend> RenderGroupDesc<B, World> for DrawFullscreenTriangleDesc {
    fn colors(&self) -> usize {
        1
    }

    fn depth(&self) -> bool {
        false
    }

    fn images(&self) -> Vec<ImageAccess> {
        vec![ImageAccess {
            access: image::Access::SHADER_READ,
            usage: image::Usage::SAMPLED,
            layout: image::Layout::ShaderReadOnlyOptimal,
            stages: pso::PipelineStage::FRAGMENT_SHADER | pso::PipelineStage::VERTEX_SHADER,
        }]
    }

    fn builder(self) -> DescBuilder<B, World, Self>
    where
        Self: Sized,
    {
        let image = self.image;
        DescBuilder::new(self).with_image(image)
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
        images: Vec<NodeImage>,
    ) -> Result<Box<dyn RenderGroup<B, World>>, failure::Error> {
        #[cfg(feature = "profiler")]
        profile_scope!("build");

        assert_eq!(images.len(), 1);

        let node_image = &images[0];
        let image = ctx.get_image(node_image.id).expect("Image does not exist");
        let sampler = factory.get_sampler(image::SamplerInfo::new(
            image::Filter::Nearest,
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

        let layout: RendyHandle<DescriptorSetLayout<B>> =
            set_layout! {factory, [1] CombinedImageSampler pso::ShaderStageFlags::FRAGMENT};

        let set = factory.create_descriptor_set(layout.clone()).unwrap();
        let desc = pso::Descriptor::CombinedImageSampler(
            image_view.raw(),
            node_image.layout,
            sampler.raw(),
        );

        unsafe {
            factory.write_descriptor_sets(vec![util::desc_write(set.raw(), 0, desc)]);
        }

        let (pipeline, pipeline_layout) = build_pipeline(
            factory,
            subpass,
            framebuffer_width,
            framebuffer_height,
            vec![layout.raw()],
            &self.shader,
        )?;

        Ok(Box::new(DrawFullscreenTriangle::<B> {
            pipeline,
            pipeline_layout,
            set,
            sampler,
            image_view,
        }))
    }
}

/// Draws debug lines
#[derive(Debug)]
pub struct DrawFullscreenTriangle<B: Backend> {
    pipeline: B::GraphicsPipeline,
    pipeline_layout: B::PipelineLayout,
    set: Escape<DescriptorSet<B>>,
    sampler: RendyHandle<Sampler<B>>,
    image_view: Escape<ImageView<B>>,
}

impl<B: Backend> RenderGroup<B, World> for DrawFullscreenTriangle<B> {
    fn prepare(
        &mut self,
        _factory: &Factory<B>,
        _queue: QueueId,
        _index: usize,
        _subpass: hal::pass::Subpass<'_, B>,
        _aux: &World,
    ) -> PrepareResult {
        PrepareResult::DrawReuse
    }

    fn draw_inline(
        &mut self,
        mut encoder: RenderPassEncoder<'_, B>,
        index: usize,
        _subpass: hal::pass::Subpass<'_, B>,
        _aux: &World,
    ) {
        #[cfg(feature = "profiler")]
        profile_scope!("draw");

        encoder.bind_graphics_pipeline(&self.pipeline);
        unsafe {
            encoder.bind_graphics_descriptor_sets(
                &self.pipeline_layout,
                0,
                Some(self.set.raw()),
                None,
            );
            encoder.draw(0..3, 0..1 as u32);
        }
    }

    fn dispose(self: Box<Self>, factory: &mut Factory<B>, _aux: &World) {
        unsafe {
            factory.device().destroy_graphics_pipeline(self.pipeline);
            factory
                .device()
                .destroy_pipeline_layout(self.pipeline_layout);
        }
    }
}

fn build_pipeline<B: Backend>(
    factory: &Factory<B>,
    subpass: hal::pass::Subpass<'_, B>,
    framebuffer_width: u32,
    framebuffer_height: u32,
    layouts: Vec<&B::DescriptorSetLayout>,
    fragment_shader: &SpirvShader,
) -> Result<(B::GraphicsPipeline, B::PipelineLayout), failure::Error> {
    let pipeline_layout = unsafe {
        factory
            .device()
            .create_pipeline_layout(layouts, None as Option<(_, _)>)
    }?;

    let shader_vertex = unsafe { super::FULLSCREEN_TRIANGLE_VERTEX.module(factory).unwrap() };
    let shader_fragment = unsafe { fragment_shader.module(factory).unwrap() };

    let pipes = PipelinesBuilder::new()
        .with_pipeline(
            PipelineDescBuilder::new()
                .with_input_assembler(pso::InputAssemblerDesc::new(hal::Primitive::TriangleList))
                .with_shaders(util::simple_shader_set(
                    &shader_vertex,
                    Some(&shader_fragment),
                ))
                .with_layout(&pipeline_layout)
                .with_subpass(subpass)
                .with_framebuffer_size(framebuffer_width, framebuffer_height)
                .with_blend_targets(vec![pso::ColorBlendDesc {
                    mask: pso::ColorMask::ALL,
                    blend: None,
                }]),
        )
        .build(factory, None);

    unsafe {
        factory.destroy_shader_module(shader_vertex);
        factory.destroy_shader_module(shader_fragment);
    }

    match pipes {
        Err(e) => {
            unsafe {
                factory.device().destroy_pipeline_layout(pipeline_layout);
            }
            Err(e)
        }
        Ok(mut pipes) => Ok((pipes.remove(0), pipeline_layout)),
    }
}
