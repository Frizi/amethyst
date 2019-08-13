use rendy::{
    command::{
        CommandBuffer, CommandPool, ExecutableState, Family, FamilyId, Fence, MultiShot,
        PendingState, Queue, QueueId, SimultaneousUse, Submission, Submit, Supports, Transfer,
    },
    factory::{blit_image, BlitImageState, BlitRegion, Factory, ImageState},
    frame::Frames,
    graph::{
        gfx_acquire_barriers, gfx_release_barriers, BufferAccess, BufferId, DynNode, GraphContext,
        ImageAccess, ImageId, NodeBuffer, NodeBuilder, NodeId, NodeImage,
    },
    hal,
};

#[derive(Debug)]
pub struct GenerateMips<B: hal::Backend> {
    pool: CommandPool<B, hal::QueueType>,
    submit: Submit<B, SimultaneousUse>,
    buffer:
        CommandBuffer<B, hal::QueueType, PendingState<ExecutableState<MultiShot<SimultaneousUse>>>>,
}

impl<B: hal::Backend> GenerateMips<B> {
    pub fn builder(target_image: ImageId, source_image: Option<ImageId>) -> GenerateMipsBuilder {
        GenerateMipsBuilder {
            target_image,
            source_image,
            dependencies: vec![],
        }
    }
}

#[derive(Debug)]
pub struct GenerateMipsBuilder {
    target_image: ImageId,
    source_image: Option<ImageId>,
    dependencies: Vec<NodeId>,
}

impl GenerateMipsBuilder {
    /// Add dependency.
    /// Node will be placed after its dependencies.
    pub fn add_dependency(&mut self, dependency: NodeId) -> &mut Self {
        self.dependencies.push(dependency);
        self
    }

    /// Add dependency.
    /// Node will be placed after its dependencies.
    pub fn with_dependency(mut self, dependency: NodeId) -> Self {
        self.add_dependency(dependency);
        self
    }
}

impl<B, T> NodeBuilder<B, T> for GenerateMipsBuilder
where
    B: hal::Backend,
    T: ?Sized,
{
    fn family(&self, _factory: &mut Factory<B>, families: &[Family<B>]) -> Option<FamilyId> {
        families
            .iter()
            .find(|family| Supports::<Transfer>::supports(&family.capability()).is_some())
            .map(|family| family.id())
    }

    fn buffers(&self) -> Vec<(BufferId, BufferAccess)> {
        Vec::new()
    }

    fn images(&self) -> Vec<(ImageId, ImageAccess)> {
        let mut images = vec![(
            self.target_image,
            ImageAccess {
                access: hal::image::Access::TRANSFER_READ | hal::image::Access::TRANSFER_WRITE,
                layout: hal::image::Layout::TransferSrcOptimal,
                usage: hal::image::Usage::TRANSFER_SRC | hal::image::Usage::TRANSFER_DST,
                stages: hal::pso::PipelineStage::TRANSFER,
            },
        )];

        if let Some(source_image) = self.source_image {
            images.push((
                source_image,
                ImageAccess {
                    access: hal::image::Access::TRANSFER_READ,
                    layout: hal::image::Layout::TransferSrcOptimal,
                    usage: hal::image::Usage::TRANSFER_SRC,
                    stages: hal::pso::PipelineStage::TRANSFER,
                },
            ));
        }
        images
    }

    fn dependencies(&self) -> Vec<NodeId> {
        self.dependencies.clone()
    }

    fn build<'a>(
        self: Box<Self>,
        ctx: &GraphContext<B>,
        factory: &mut Factory<B>,
        family: &mut Family<B>,
        queue: usize,
        _aux: &T,
        buffers: Vec<NodeBuffer>,
        images: Vec<NodeImage>,
    ) -> Result<Box<dyn DynNode<B, T>>, failure::Error> {
        assert_eq!(buffers.len(), 0);
        let node_image = images
            .iter()
            .find(|image| image.id == self.target_image)
            .unwrap();
        let node_src_image = self
            .source_image
            .and_then(|source_id| images.iter().find(|image| image.id == source_id));

        let mut pool = factory.create_command_pool(family)?;

        let buf_initial = pool.allocate_buffers(1).pop().unwrap();
        let mut buf_recording = buf_initial.begin(MultiShot(SimultaneousUse), ());
        let mut encoder = buf_recording.encoder();

        {
            let (stages, barriers) = gfx_acquire_barriers(ctx, None, Some(node_image));
            log::trace!("Acquire {:?} : {:#?}", stages, barriers);
            if !barriers.is_empty() {
                unsafe {
                    encoder.pipeline_barrier(stages, hal::memory::Dependencies::empty(), barriers);
                }
            }
        }

        if let Some(node_src_image) = node_src_image {
            let (stages, barriers) = gfx_acquire_barriers(ctx, None, Some(node_src_image));
            log::trace!("Acquire {:?} : {:#?}", stages, barriers);
            if !barriers.is_empty() {
                unsafe {
                    encoder.pipeline_barrier(stages, hal::memory::Dependencies::empty(), barriers);
                }
            }
        }

        let boundary_state = ImageState {
            queue: QueueId {
                family: family.id(),
                index: queue,
            },
            stage: hal::pso::PipelineStage::TRANSFER,
            access: hal::image::Access::TRANSFER_WRITE,
            layout: node_image.layout,
        };

        let image = ctx.get_image(node_image.id).expect("Image does not exist");

        if let Some(node_src_image) = node_src_image {
            let src_image = ctx
                .get_image(node_src_image.id)
                .expect("Image does not exist");
            let src_aspects = src_image.format().surface_desc().aspects;
            let dst_aspects = image.format().surface_desc().aspects;

            // TODO: This should probably be a copy when sizes are equal but I'm lazy.
            let blit = BlitRegion {
                src: BlitImageState {
                    subresource: hal::image::SubresourceLayers {
                        aspects: src_aspects,
                        level: 0,
                        layers: 0..image.layers(),
                    },
                    bounds: hal::image::Offset::ZERO.into_bounds(&src_image.kind().extent()),
                    last_stage: boundary_state.stage,
                    last_access: boundary_state.access,
                    last_layout: boundary_state.layout,
                    next_stage: boundary_state.stage,
                    next_access: boundary_state.access,
                    next_layout: boundary_state.layout,
                },
                dst: BlitImageState {
                    subresource: hal::image::SubresourceLayers {
                        aspects: dst_aspects,
                        level: 0,
                        layers: 0..image.layers(),
                    },
                    bounds: hal::image::Offset::ZERO.into_bounds(&image.kind().extent()),
                    last_stage: boundary_state.stage,
                    last_access: hal::image::Access::empty(),
                    last_layout: hal::image::Layout::Undefined,
                    next_stage: boundary_state.stage,
                    next_access: boundary_state.access,
                    next_layout: boundary_state.layout,
                },
            };
            unsafe {
                blit_image(
                    &mut encoder,
                    src_image,
                    image,
                    hal::image::Filter::Linear,
                    Some(blit),
                )?;
            }
        }

        let (_queue, blits) = BlitRegion::mip_blits_for_image(
            image,
            std::iter::repeat(boundary_state),
            std::iter::repeat(boundary_state),
        );

        for blit in blits {
            unsafe {
                blit_image(
                    &mut encoder,
                    image,
                    image,
                    hal::image::Filter::Linear,
                    Some(blit),
                )?;
            }
        }

        {
            let (stages, barriers) = gfx_release_barriers(ctx, None, images.iter());
            log::trace!("Release {:?} : {:#?}", stages, barriers);
            if !barriers.is_empty() {
                unsafe {
                    encoder.pipeline_barrier(stages, hal::memory::Dependencies::empty(), barriers);
                }
            }
        }

        let (submit, buffer) = buf_recording.finish().submit();

        Ok(Box::new(GenerateMips {
            pool,
            submit,
            buffer,
        }))
    }
}

impl<B, T> DynNode<B, T> for GenerateMips<B>
where
    B: hal::Backend,
    T: ?Sized,
{
    unsafe fn run<'a>(
        &mut self,
        _ctx: &GraphContext<B>,
        _factory: &Factory<B>,
        queue: &mut Queue<B>,
        _aux: &T,
        _frames: &Frames<B>,
        waits: &[(&'a B::Semaphore, hal::pso::PipelineStage)],
        signals: &[&'a B::Semaphore],
        fence: Option<&mut Fence<B>>,
    ) {
        queue.submit(
            Some(
                Submission::new()
                    .submits(Some(&self.submit))
                    .wait(waits.iter().cloned())
                    .signal(signals.iter()),
            ),
            fence,
        );
    }

    unsafe fn dispose(mut self: Box<Self>, factory: &mut Factory<B>, _aux: &T) {
        drop(self.submit);
        self.pool.free_buffers(Some(self.buffer.mark_complete()));
        factory.destroy_command_pool(self.pool);
    }
}
