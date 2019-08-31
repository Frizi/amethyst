//! A home of [RenderingBundle] with it's rendering plugins system and all types directly related to it.

use crate::{
    mtl::Material,
    rendy::{
        factory::Factory,
        graph::{
            render::{RenderGroupBuilder, RenderPassNodeBuilder, SubpassBuilder},
            GraphBuilder, ImageId, NodeId,
        },
        hal,
        wsi::Surface,
    },
    system::{GraphCreator, MeshProcessor, RenderingSystem, TextureProcessor},
    types::Backend,
    SpriteSheet,
};
use amethyst_assets::Processor;
use amethyst_core::{
    ecs::{DispatcherBuilder, World},
    SystemBundle,
};
use amethyst_error::{format_err, Error};
use std::collections::HashMap;

/// A bundle of systems used for rendering using `Rendy` render graph.
///
/// Provides a mechanism for registering rendering plugins.
/// By itself doesn't render anything, you must use `with_plugin` method
/// to define a set of functionalities you want to use.
#[derive(Debug)]
pub struct RenderingBundle<B: Backend> {
    plugins: Vec<Box<dyn RenderPlugin<B>>>,
}

impl<B: Backend> RenderingBundle<B> {
    /// Create empty `RenderingBundle`. You must register a plugin using
    /// [`with_plugin`] in order to actually display anything.
    pub fn new() -> Self {
        Self {
            plugins: Vec::new(),
        }
    }

    /// Register a [`RenderPlugin`].
    ///
    /// If you want the non-consuming version of this method, see [`add_plugin`].
    pub fn with_plugin(mut self, plugin: impl RenderPlugin<B> + 'static) -> Self {
        self.add_plugin(plugin);
        self
    }

    /// Register a [`RenderPlugin`].
    pub fn add_plugin(&mut self, plugin: impl RenderPlugin<B> + 'static) {
        self.plugins.push(Box::new(plugin));
    }

    fn into_graph_creator(self) -> RenderingBundleGraphCreator<B> {
        RenderingBundleGraphCreator {
            plugins: self.plugins,
        }
    }
}

impl<'a, 'b, B: Backend> SystemBundle<'a, 'b> for RenderingBundle<B> {
    fn build(
        mut self,
        world: &mut World,
        builder: &mut DispatcherBuilder<'a, 'b>,
    ) -> Result<(), Error> {
        builder.add(MeshProcessor::<B>::default(), "mesh_processor", &[]);
        builder.add(TextureProcessor::<B>::default(), "texture_processor", &[]);
        builder.add(Processor::<Material>::new(), "material_processor", &[]);
        builder.add(
            Processor::<SpriteSheet>::new(),
            "sprite_sheet_processor",
            &[],
        );

        // make sure that all renderer-specific systems run after game code
        builder.add_barrier();

        for plugin in &mut self.plugins {
            plugin.on_build(world, builder)?;
        }

        builder.add_thread_local(RenderingSystem::<B, _>::new(self.into_graph_creator()));
        Ok(())
    }
}

struct RenderingBundleGraphCreator<B: Backend> {
    plugins: Vec<Box<dyn RenderPlugin<B>>>,
}

impl<B: Backend> GraphCreator<B> for RenderingBundleGraphCreator<B> {
    fn rebuild(&mut self, world: &World) -> bool {
        let mut rebuild = false;
        for plugin in self.plugins.iter_mut() {
            rebuild = plugin.should_rebuild(world) || rebuild;
        }
        rebuild
    }

    fn builder(&mut self, factory: &mut Factory<B>, world: &World) -> GraphBuilder<B, World> {
        if self.plugins.is_empty() {
            log::warn!("RenderingBundle is configured to display nothing. Use `with_plugin` to add functionality.");
        }

        let mut plan = RenderPlan::new();
        for plugin in self.plugins.iter_mut() {
            plugin.on_plan(&mut plan, factory, world).unwrap();
        }
        plan.build(factory).unwrap()
    }
}

/// Basic building block of rendering in [RenderingBundle].
///
/// Can be used to register rendering-related systems to the dispatcher,
/// building render graph by registering render targets, adding [RenderableAction]s to them
/// and signalling when the graph has to be rebuild.
pub trait RenderPlugin<B: Backend>: std::fmt::Debug {
    /// Hook for adding systems and bundles to the dispatcher.
    fn on_build<'a, 'b>(
        &mut self,
        _world: &mut World,
        _builder: &mut DispatcherBuilder<'a, 'b>,
    ) -> Result<(), Error> {
        Ok(())
    }

    /// Hook for providing triggers to rebuild the render graph.
    fn should_rebuild(&mut self, _world: &World) -> bool {
        false
    }

    /// Hook for extending the rendering plan.
    fn on_plan(
        &mut self,
        plan: &mut RenderPlan<B>,
        factory: &mut Factory<B>,
        world: &World,
    ) -> Result<(), Error>;
}

/// Builder of a rendering plan for specified target.
#[derive(Debug)]
pub struct RenderPlan<B: Backend> {
    targets: HashMap<Target, TargetPlan<B>>,
    roots: Vec<Target>,
}

impl<B: Backend> RenderPlan<B> {
    fn new() -> Self {
        Self {
            targets: Default::default(),
            roots: Vec::new(),
        }
    }

    /// Mark render target as root. Root render targets are always
    /// evaluated, even if nothing depends on them.
    pub fn add_root(&mut self, target: Target) {
        if !self.roots.contains(&target) {
            self.roots.push(target);
        }
    }

    /// Define a render target with predefined set of outputs.
    pub fn define_pass(
        &mut self,
        target: Target,
        outputs: TargetPlanOutputs<B>,
    ) -> Result<(), Error> {
        for foreign in outputs.pre_foreigns() {
            self.add_lazy_dep(foreign.target(), target);
        }

        let target_plan = self
            .targets
            .entry(target)
            .or_insert_with(|| TargetPlan::new(target));

        target_plan.set_outputs(outputs)?;

        Ok(())
    }
    fn add_lazy_dep(&mut self, target: Target, dep: Target) {
        let target_plan = self
            .targets
            .entry(target)
            .or_insert_with(|| TargetPlan::new(target));

        target_plan.add_lazy_dep(dep);
    }

    /// Extend the rendering plan of a render target. Target can be defined in other plugins.
    /// The closure is evaluated only if the target contributes to the rendering result, e.g.
    /// is rendered to a window or is a dependency of other evaluated target.
    pub fn extend_target(
        &mut self,
        target: Target,
        closure: impl FnOnce(&mut TargetPlanContext<'_, B>) -> Result<(), Error> + 'static,
    ) {
        let target_plan = self
            .targets
            .entry(target)
            .or_insert_with(|| TargetPlan::new(target));
        target_plan.add_extension(Box::new(closure));
    }

    fn build(self, factory: &Factory<B>) -> Result<GraphBuilder<B, World>, Error> {
        let mut ctx = PlanContext {
            target_metadata: self
                .targets
                .iter()
                .filter_map(|(k, t)| {
                    unsafe { t.metadata(factory.physical(), &self.targets) }.map(|m| (*k, m))
                })
                .collect(),
            targets: self.targets,
            evaluations: Default::default(),
            outputs: Default::default(),
            graph_builder: GraphBuilder::new(),
        };

        for target in self.roots {
            ctx.evaluate_target(target)?;
        }

        Ok(ctx.graph_builder)
    }
}

#[derive(Debug)]
enum EvaluationState {
    Evaluating,
    EvaluatedOutputs,
    Built(NodeId),
}

impl EvaluationState {
    fn node(&self) -> Option<NodeId> {
        match self {
            EvaluationState::Built(node) => Some(*node),
            _ => None,
        }
    }
    fn has_outputs(&self) -> bool {
        match self {
            EvaluationState::Built(_) | EvaluationState::EvaluatedOutputs => true,
            _ => false,
        }
    }

    fn is_built(&self) -> bool {
        self.node().is_some()
    }
}

/// Metadata for a planned render target.
/// Defines effective size and layer count that target's renderpass will operate on.
#[derive(Debug, Clone, Copy)]
pub struct TargetMetadata {
    width: u32,
    height: u32,
    layers: u16,
}

impl TargetMetadata {
    fn shrink_to(&mut self, other: TargetMetadata) {
        use std::cmp::min;
        self.width = min(self.width, other.width);
        self.height = min(self.height, other.height);
        self.layers = min(self.layers, other.layers);
    }
}

#[derive(Debug)]
pub struct PlanContext<B: Backend> {
    targets: HashMap<Target, TargetPlan<B>>,
    target_metadata: HashMap<Target, TargetMetadata>,
    evaluations: HashMap<Target, EvaluationState>,
    outputs: HashMap<TargetImage, ImageId>,
    graph_builder: GraphBuilder<B, World>,
}

impl<B: Backend> PlanContext<B> {
    pub fn mark_evaluating(&mut self, target: Target, outputs: bool) -> Result<(), Error> {
        match self.evaluations.get(&target) {
            // this case is not a soft runtime error, as this should never be allowed by the API.
            Some(EvaluationState::Built(_)) => panic!("Trying to reevaluate a render plan for {:?}.", target),
            Some(EvaluationState::EvaluatedOutputs) if outputs => panic!("Trying to reevaluate a render plan outputs for {:?}.", target),
            Some(EvaluationState::Evaluating) => return Err(format_err!("Trying to evaluate {:?} render plan that is already evaluating. Circular dependency detected.", target)),
            None if !outputs => panic!("Trying to evaluate a render plan without evaluating outputs for {:?}.", target),
            None | Some(EvaluationState::EvaluatedOutputs) => {},
        };
        self.evaluations.insert(target, EvaluationState::Evaluating);
        Ok(())
    }

    fn submit_outputs(&mut self, target: Target) {
        if let Some(EvaluationState::EvaluatedOutputs) | Some(EvaluationState::Built(_)) =
            self.evaluations.get(&target)
        {
            panic!(
                "Trying to resubmit a render pass outputs for {:?}. This is a RenderingBundle bug.",
                target
            );
        }
        self.evaluations
            .insert(target, EvaluationState::EvaluatedOutputs);
    }

    fn evaluate_outputs(&mut self, target: Target) -> Result<(), Error> {
        if !self
            .evaluations
            .get(&target)
            .map_or(false, |t| t.has_outputs())
        {
            if let Some(mut pass) = self.targets.remove(&target) {
                pass.evaluate_outputs(self)?;
                self.targets.insert(target, pass);
            }
        }
        Ok(())
    }

    fn evaluate_target(&mut self, target: Target) -> Result<(), Error> {
        self.evaluate_outputs(target)?;
        // prevent evaluation of roots that were accessed recursively or undefined
        if let Some(pass) = self.targets.remove(&target) {
            pass.evaluate(self)?;
        }
        Ok(())
    }

    fn submit_target(&mut self, target: Target, node: NodeId) {
        if let Some(EvaluationState::Built(_)) = self.evaluations.get(&target) {
            panic!(
                "Trying to resubmit a render pass for {:?}. This is a RenderingBundle bug.",
                target
            );
        }

        self.evaluations
            .insert(target, EvaluationState::Built(node));
    }
    fn get_target_node_raw(&self, target: Target) -> Option<NodeId> {
        self.evaluations.get(&target).and_then(|p| p.node())
    }

    pub fn get_node(&mut self, target: Target) -> Result<NodeId, Error> {
        match self.get_target_node_raw(target) {
            Some(node) => Ok(node),
            None => {
                self.evaluate_target(target)?;
                Ok(self
                    .evaluations
                    .get(&target)
                    .and_then(|p| p.node())
                    .expect("Just built"))
            }
        }
    }

    pub fn target_metadata(&self, target: Target) -> Option<TargetMetadata> {
        self.target_metadata.get(&target).copied()
    }

    pub fn get_image(&mut self, image_ref: TargetImage) -> Result<ImageId, Error> {
        self.try_get_image(image_ref)?.ok_or_else(|| {
            format_err!(
                "Output image {:?} is not registered by the target.",
                image_ref
            )
        })
    }

    pub fn try_get_image(&mut self, image_ref: TargetImage) -> Result<Option<ImageId>, Error> {
        self.evaluate_outputs(image_ref.target())?;
        Ok(self.outputs.get(&image_ref).cloned())
    }

    pub fn register_output(&mut self, output: TargetImage, image: ImageId) -> Result<(), Error> {
        if self.outputs.contains_key(&output) {
            return Err(format_err!(
                "Trying to register already registered output image {:?}",
                output
            ));
        }
        self.outputs.insert(output, image);
        Ok(())
    }

    pub fn graph(&mut self) -> &mut GraphBuilder<B, World> {
        &mut self.graph_builder
    }

    pub fn create_image(&mut self, options: ImageOptions) -> ImageId {
        self.graph_builder
            .create_image(options.kind, options.levels, options.format, options.clear)
    }
}

/// A planning context focused on specific render target.
#[derive(derivative::Derivative)]
#[derivative(Debug(bound = ""))]
pub struct TargetPlanContext<'a, B: Backend> {
    plan_context: &'a mut PlanContext<B>,
    key: Target,
    colors: usize,
    depth: bool,
    actions: Vec<(i32, RenderableAction<B>)>,
    #[derivative(Debug = "ignore")]
    post_mods: Vec<(
        i32,
        Box<dyn FnOnce(&mut PlanContext<B>, NodeId) -> Result<NodeId, Error>>,
    )>,
    deps: Vec<NodeId>,
}

impl<'a, B: Backend> TargetPlanContext<'a, B> {
    /// Add new action to render target in defined order.
    pub fn add(&mut self, order: impl Into<i32>, action: impl IntoAction<B>) -> Result<(), Error> {
        let action = action.into();

        if self.colors != action.colors() {
            return Err(format_err!(
                "Trying to add render action with {} colors to target {:?} that expects {} colors.",
                action.colors(),
                self.key,
                self.colors,
            ));
        }
        if self.depth != action.depth() {
            return Err(format_err!(
                "Trying to add render action with depth '{}' to target {:?} that expects depth '{}'.",
                action.depth(),
                self.key,
                self.depth,
            ));
        }

        self.actions.push((order.into(), action));
        Ok(())
    }

    /// Modify a pass after it has been constructed as a node, possibly adding another node in the chain.
    pub fn add_post(
        &mut self,
        order: impl Into<i32>,
        post_mod: impl FnOnce(&mut PlanContext<B>, NodeId) -> Result<NodeId, Error> + 'static,
    ) {
        self.post_mods.push((order.into(), Box::new(post_mod)));
    }

    /// Get number of color outputs of current render target.
    pub fn colors(&self) -> usize {
        self.colors
    }

    /// Check if current render target has a depth output.
    pub fn depth(&self) -> bool {
        self.depth
    }

    /// Retrieve an image produced by other render target.
    ///
    /// Results in an error if such image doesn't exist or
    /// retreiving it would result in a dependency cycle.
    pub fn get_image(&mut self, target_image: TargetImage) -> Result<ImageId, Error> {
        let image = self.plan_context.get_image(target_image)?;
        let node = self.plan_context.get_node(target_image.target())?;
        self.add_dep(node);
        Ok(image)
    }

    /// Create new local image
    pub fn create_image(&mut self, options: ImageOptions) -> ImageId {
        self.plan_context.create_image(options)
    }

    /// Retrieve an image produced by other render target.
    /// Returns `None` when such image isn't registered.
    ///
    /// Results in an error if retreiving it would result in a dependency cycle.
    pub fn try_get_image(&mut self, target_image: TargetImage) -> Result<Option<ImageId>, Error> {
        let image = self.plan_context.try_get_image(target_image)?;
        if image.is_some() {
            let node = self.plan_context.get_node(target_image.target())?;
            self.add_dep(node);
        }
        Ok(image)
    }

    /// Add explicit dependency on another node.
    ///
    /// This is done automatically when you use `get_image`.
    pub fn add_dep(&mut self, node: NodeId) {
        if !self.deps.contains(&node) {
            self.deps.push(node);
        }
    }

    /// Access underlying rendy's GraphBuilder directly.
    /// This is useful for adding custom rendering nodes
    /// that are not just standard graphics render passes,
    /// e.g. for compute dispatch.
    pub fn graph(&mut self) -> &mut GraphBuilder<B, World> {
        self.plan_context.graph()
    }

    /// Retrieve render target metadata, e.g. size.
    pub fn target_metadata(&self, target: Target) -> Option<TargetMetadata> {
        self.plan_context.target_metadata(target)
    }

    /// Access computed NodeId of render target.
    pub fn get_node(&mut self, target: Target) -> Result<NodeId, Error> {
        self.plan_context.get_node(target)
    }
}

/// An identifier for output image of specific render target.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum TargetImage {
    /// Select target color output with given index.
    Color(Target, usize),
    /// Select target depth output.
    Depth(Target),
}

impl TargetImage {
    /// Retrieve target identifier for this image
    pub fn target(&self) -> Target {
        match self {
            TargetImage::Color(target, _) => *target,
            TargetImage::Depth(target) => *target,
        }
    }
}

/// Set of options required to create an image node in render graph.
#[derive(Debug, Clone, PartialEq)]
pub struct ImageOptions {
    /// Image kind and size
    pub kind: hal::image::Kind,
    /// Number of mipmap levels
    pub levels: hal::image::Level,
    /// Image format
    pub format: hal::format::Format,
    /// Clear operation performed once per frame.
    pub clear: Option<hal::command::ClearValue>,
}

/// Definition of render target color output image.
#[derive(Debug)]
pub enum OutputColor<B: Backend> {
    /// Render to image with specified options
    Image(ImageOptions),
    /// Render directly to a window surface.
    Surface(Surface<B>, Option<hal::command::ClearValue>),
    /// Render to other pass' image before that pass is rendered.
    PreForeign(TargetImage),
}

#[derive(Debug)]
pub enum OutputDepth {
    /// Render to image with specified options
    Image(ImageOptions),
    /// Render to other pass' image before that pass is rendered.
    PreForeign(TargetImage),
}

/// Definition for set of outputs for a given render target.
#[derive(Debug)]
pub struct TargetPlanOutputs<B: Backend> {
    /// List of target color outputs with options
    pub colors: Vec<OutputColor<B>>,
    /// Settings for optional depth output
    pub depth: Option<OutputDepth>,
}

impl<B: Backend> TargetPlanOutputs<B> {
    fn pre_foreigns(&self) -> impl Iterator<Item = &TargetImage> {
        self.colors
            .iter()
            .filter_map(|c| match c {
                OutputColor::PreForeign(f) => Some(f),
                _ => None,
            })
            .chain(self.depth.as_ref().and_then(|d| match d {
                OutputDepth::PreForeign(f) => Some(f),
                _ => None,
            }))
    }
}

#[derive(derivative::Derivative)]
#[derivative(Debug(bound = ""))]
struct TargetPlan<B: Backend> {
    key: Target,
    #[derivative(Debug = "ignore")]
    extensions: Vec<Box<dyn FnOnce(&mut TargetPlanContext<'_, B>) -> Result<(), Error> + 'static>>,
    outputs: Option<TargetPlanOutputs<B>>,
    lazy_deps: Vec<Target>,
}

impl<B: Backend> TargetPlan<B> {
    fn new(key: Target) -> Self {
        Self {
            key,
            extensions: Vec::new(),
            outputs: None,
            lazy_deps: Vec::new(),
        }
    }

    fn add_lazy_dep(&mut self, target: Target) {
        if !self.lazy_deps.contains(&target) {
            self.lazy_deps.push(target);
        }
    }

    // safety:
    // * `physical_device` must be created from same `Instance` as the `Surface` present in output
    unsafe fn metadata(
        &self,
        physical_device: &B::PhysicalDevice,
        foreign_targets: &HashMap<Target, TargetPlan<B>>,
    ) -> Option<TargetMetadata> {
        fn surface_metadata<B: Backend>(
            surface: &Surface<B>,
            device: &B::PhysicalDevice,
        ) -> TargetMetadata {
            if let Some(extent) = unsafe { surface.extent(device) } {
                TargetMetadata {
                    width: extent.width,
                    height: extent.height,
                    layers: 1,
                }
            } else {
                // Window was just closed, using size of 1 is the least bad option
                // to default to. The output won't be used, things won't crash and
                // graph is either going to be destroyed or rebuilt next frame.
                TargetMetadata {
                    width: 1,
                    height: 1,
                    layers: 1,
                }
            }
        }

        fn image_metadata(options: &ImageOptions) -> TargetMetadata {
            let extent = options.kind.extent();
            TargetMetadata {
                width: extent.width,
                height: extent.height,
                layers: options.kind.num_layers(),
            }
        }

        fn foreign_metadata<B: Backend>(
            target_image: TargetImage,
            device: &B::PhysicalDevice,
            foreign_targets: &HashMap<Target, TargetPlan<B>>,
        ) -> Option<TargetMetadata> {
            if let Some(outputs) = foreign_targets
                .get(&target_image.target())
                .and_then(|plan| plan.outputs.as_ref())
            {
                match target_image {
                    TargetImage::Color(_, idx) => outputs
                        .colors
                        .get(idx)
                        .and_then(|color| output_color_metadata(color, device, foreign_targets)),
                    TargetImage::Depth(_) => outputs
                        .depth
                        .as_ref()
                        .and_then(|depth| output_depth_metadata(depth, device, foreign_targets)),
                }
            } else {
                None
            }
        }

        fn output_color_metadata<B: Backend>(
            color: &OutputColor<B>,
            device: &B::PhysicalDevice,
            foreign_targets: &HashMap<Target, TargetPlan<B>>,
        ) -> Option<TargetMetadata> {
            match color {
                OutputColor::Surface(surface, _) => Some(surface_metadata(surface, device)),
                OutputColor::Image(options) => Some(image_metadata(options)),
                OutputColor::PreForeign(foregin) => {
                    foreign_metadata(*foregin, device, foreign_targets)
                }
            }
        }

        fn output_depth_metadata<B: Backend>(
            depth: &OutputDepth,
            device: &B::PhysicalDevice,
            foreign_targets: &HashMap<Target, TargetPlan<B>>,
        ) -> Option<TargetMetadata> {
            match depth {
                OutputDepth::Image(options) => Some(image_metadata(options)),
                OutputDepth::PreForeign(foreign) => {
                    foreign_metadata(*foreign, device, foreign_targets)
                }
            }
        }

        self.outputs
            .as_ref()
            .map(|TargetPlanOutputs { colors, depth }| {
                let mut metadata = TargetMetadata {
                    width: u32::max_value(),
                    height: u32::max_value(),
                    layers: u16::max_value(),
                };

                for color in colors {
                    if let Some(meta) =
                        output_color_metadata(&color, physical_device, foreign_targets)
                    {
                        metadata.shrink_to(meta);
                    }
                }

                if let Some(meta) = depth.as_ref().and_then(|depth| {
                    output_depth_metadata(depth, physical_device, foreign_targets)
                }) {
                    metadata.shrink_to(meta);
                }

                metadata
            })
    }

    fn set_outputs(&mut self, outputs: TargetPlanOutputs<B>) -> Result<(), Error> {
        if self.outputs.is_some() {
            return Err(format_err!("Target {:?} already defined.", self.key));
        }
        self.outputs.replace(outputs);
        Ok(())
    }

    fn add_extension(
        &mut self,
        extension: Box<dyn FnOnce(&mut TargetPlanContext<'_, B>) -> Result<(), Error> + 'static>,
    ) {
        self.extensions.push(extension);
    }

    fn get_outputs(&self) -> Result<&TargetPlanOutputs<B>, Error> {
        match &self.outputs {
            None => Err(format_err!(
                "Trying to evaluate not fully defined pass {:?}. Missing `define_pass` call.",
                self.key
            )),
            Some(outs) => Ok(outs),
        }
    }

    fn take_outputs(&mut self) -> Result<TargetPlanOutputs<B>, Error> {
        match self.outputs.take() {
            None => Err(format_err!(
                "Trying to evaluate not fully defined pass {:?}. Missing `define_pass` call.",
                self.key
            )),
            Some(outs) => Ok(outs),
        }
    }

    fn evaluate_outputs(&mut self, ctx: &mut PlanContext<B>) -> Result<(), Error> {
        let outputs = self.get_outputs()?;

        ctx.mark_evaluating(self.key, true)?;

        for (i, color) in outputs.colors.iter().enumerate() {
            match color {
                OutputColor::Surface(surface, clear) => {
                    // Surfaces are not registered
                }
                OutputColor::Image(opts) => {
                    let image = ctx.create_image(opts.clone());
                    ctx.register_output(TargetImage::Color(self.key, i), image)?;
                }
                OutputColor::PreForeign(target_image) => {
                    let image = ctx.get_image(*target_image)?;
                    ctx.register_output(TargetImage::Color(self.key, i), image)?;
                }
            }
        }

        match &outputs.depth {
            None => {}
            Some(OutputDepth::Image(opts)) => {
                let image = ctx.create_image(opts.clone());
                ctx.register_output(TargetImage::Depth(self.key), image)?;
            }
            Some(OutputDepth::PreForeign(target_image)) => {
                let image = ctx.get_image(*target_image)?;
                ctx.register_output(TargetImage::Depth(self.key), image)?;
            }
        }

        ctx.submit_outputs(self.key);
        Ok(())
    }

    fn evaluate(mut self, ctx: &mut PlanContext<B>) -> Result<(), Error> {
        ctx.mark_evaluating(self.key, false)?;

        let mut outputs = self.take_outputs()?;

        let mut target_ctx = TargetPlanContext {
            plan_context: ctx,
            key: self.key,
            actions: Vec::new(),
            colors: outputs.colors.len(),
            depth: outputs.depth.is_some(),
            post_mods: Vec::new(),
            deps: Vec::new(),
        };

        for extension in self.extensions {
            extension(&mut target_ctx)?;
        }

        let TargetPlanContext {
            mut actions,
            mut post_mods,
            deps,
            ..
        } = target_ctx;

        actions.sort_by_key(|a| a.0);
        post_mods.sort_by_key(|a| a.0);

        let mut subpass = SubpassBuilder::new();
        let mut pass = RenderPassNodeBuilder::new();

        for action in actions.drain(..).map(|a| a.1) {
            match action {
                RenderableAction::RenderGroup(group) => {
                    subpass.add_dyn_group(group);
                }
            }
        }

        for (i, color) in outputs.colors.drain(..).enumerate() {
            match color {
                OutputColor::Surface(surface, clear) => {
                    subpass.add_color_surface();
                    pass.add_surface(surface, clear);
                }
                OutputColor::Image(_) | OutputColor::PreForeign(_) => {
                    let image = ctx.get_image(TargetImage::Color(self.key, i))?;
                    subpass.add_color(image);
                }
            }
        }

        if outputs.depth.is_some() {
            let image = ctx.get_image(TargetImage::Depth(self.key))?;
            subpass.set_depth_stencil(image);
        }

        for target in self.lazy_deps {
            let node = ctx.get_node(target)?;
            subpass.add_dependency(node);
        }

        for node in deps {
            subpass.add_dependency(node);
        }

        pass.add_subpass(subpass);

        let node = ctx.graph().add_node(pass);
        let node = post_mods
            .into_iter()
            .fold(Ok(node), |node, (_, post_mod)| post_mod(ctx, node?))?;

        ctx.submit_target(self.key, node);
        Ok(())
    }
}

/// An action that represents a single transformation to the
/// render graph, e.g. addition of single render group.
///
/// TODO: more actions needed for e.g. splitting pass into subpasses.
#[derive(Debug)]
pub enum RenderableAction<B: Backend> {
    /// Register single render group for evaluation during target rendering
    RenderGroup(Box<dyn RenderGroupBuilder<B, World>>),
}

impl<B: Backend> RenderableAction<B> {
    fn colors(&self) -> usize {
        match self {
            RenderableAction::RenderGroup(g) => g.colors(),
        }
    }

    fn depth(&self) -> bool {
        match self {
            RenderableAction::RenderGroup(g) => g.depth(),
        }
    }
}

/// Trait for easy conversion of various types into `RenderableAction` shell.
pub trait IntoAction<B: Backend> {
    /// Convert to `RenderableAction`.
    fn into(self) -> RenderableAction<B>;
}

impl<B: Backend, G: RenderGroupBuilder<B, World> + 'static> IntoAction<B> for G {
    fn into(self) -> RenderableAction<B> {
        RenderableAction::RenderGroup(Box::new(self))
    }
}

/// Collection of predefined constants for action ordering in the builtin targets.
/// Two actions with the same order will be applied in their insertion order.
/// The list is provided mostly as a comparison point. If you can't find the exact
/// ordering you need, provide custom `i32` that fits into the right place.
///
/// Modules that provide custom render plugins using their own orders should export
/// similar enum with ordering they have added.
#[derive(Debug)]
#[repr(i32)]
pub enum RenderOrder {
    /// register before all opaques
    BeforeOpaque = 90,
    /// register for rendering opaque objects
    Opaque = 100,
    /// register after rendering opaque objects
    AfterOpaque = 110,
    /// register before rendering transparent objects
    BeforeTransparent = 190,
    /// register for rendering transparent objects
    Transparent = 200,
    /// register after rendering transparent objects
    AfterTransparent = 210,
    /// register as post effect in linear color space
    LinearPostEffects = 300,
    /// register as tonemapping step
    ToneMap = 400,
    /// register as post effect in display color space
    DisplayPostEffects = 500,
    /// register as overlay on final render
    Overlay = 600,
}

impl Into<i32> for RenderOrder {
    fn into(self) -> i32 {
        self as i32
    }
}

/// An identifier for render target used in render plugins.
/// Predefined targets are part of default rendering flow
/// used by builtin amethyst render plugins, but the list
/// can be arbitrarily extended for custom usage in user
/// plugins using custom str identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Target {
    /// A depth-only pass that feeds initial depth into main pass
    DepthPrepass,
    /// Default render target for most operations.
    /// Usually the one that gets presented to the window.
    Main,
    /// Render target for depth used for shadow mapping.
    ShadowMapDepth,
    /// Render target for filtered EVSM data for shadow mapping.
    ShadowMapEvsm,
    /// Custom render target identifier.
    Custom(&'static str),
}

impl Default for Target {
    fn default() -> Target {
        Target::Main
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        rendy::{
            command::QueueId,
            graph::{
                render::{RenderGroup, RenderGroupDesc},
                GraphContext, NodeBuffer, NodeImage,
            },
        },
        types::{Backend, DefaultBackend},
    };
    use hal::{
        command::{ClearDepthStencil, ClearValue},
        format::Format,
    };

    #[derive(Debug)]
    struct TestGroup1;
    #[derive(Debug)]
    struct TestGroup2;

    impl<B: Backend, T> RenderGroupDesc<B, T> for TestGroup1 {
        fn build(
            self,
            ctx: &GraphContext<B>,
            factory: &mut Factory<B>,
            queue: QueueId,
            aux: &T,
            framebuffer_width: u32,
            framebuffer_height: u32,
            subpass: hal::pass::Subpass<'_, B>,
            buffers: Vec<NodeBuffer>,
            images: Vec<NodeImage>,
        ) -> Result<Box<dyn RenderGroup<B, T>>, failure::Error> {
            unimplemented!()
        }
    }
    impl<B: Backend, T> RenderGroupDesc<B, T> for TestGroup2 {
        fn build(
            self,
            ctx: &GraphContext<B>,
            factory: &mut Factory<B>,
            queue: QueueId,
            aux: &T,
            framebuffer_width: u32,
            framebuffer_height: u32,
            subpass: hal::pass::Subpass<'_, B>,
            buffers: Vec<NodeBuffer>,
            images: Vec<NodeImage>,
        ) -> Result<Box<dyn RenderGroup<B, T>>, failure::Error> {
            unimplemented!()
        }
    }

    #[test]
    #[ignore] // CI can't run tests requiring actual backend
    fn main_pass_color_image_plan() {
        let config: rendy::factory::Config = Default::default();
        let (factory, families): (Factory<DefaultBackend>, _) =
            rendy::factory::init(config).unwrap();
        let mut plan = RenderPlan::<DefaultBackend>::new();

        plan.extend_target(Target::Main, |ctx| {
            ctx.add(RenderOrder::Transparent, TestGroup1.builder())?;
            ctx.add(RenderOrder::Opaque, TestGroup2.builder())?;
            Ok(())
        });

        let kind = crate::Kind::D2(1920, 1080, 1, 1);
        plan.add_root(Target::Main);
        plan.define_pass(
            Target::Main,
            TargetPlanOutputs {
                colors: vec![OutputColor::Image(ImageOptions {
                    kind,
                    levels: 1,
                    format: Format::Rgb8Unorm,
                    clear: None,
                })],
                depth: Some(ImageOptions {
                    kind,
                    levels: 1,
                    format: Format::D32Sfloat,
                    clear: Some(ClearValue::DepthStencil(ClearDepthStencil(1.0, 0))),
                }),
            },
        )
        .unwrap();

        let planned_graph = plan.build(&factory).unwrap();

        let mut manual_graph = GraphBuilder::<DefaultBackend, World>::new();
        let color = manual_graph.create_image(kind, 1, Format::Rgb8Unorm, None);
        let depth = manual_graph.create_image(
            kind,
            1,
            Format::D32Sfloat,
            Some(ClearValue::DepthStencil(ClearDepthStencil(1.0, 0))),
        );
        manual_graph.add_node(
            RenderPassNodeBuilder::new().with_subpass(
                SubpassBuilder::new()
                    .with_group(TestGroup2.builder())
                    .with_group(TestGroup1.builder())
                    .with_color(color)
                    .with_depth_stencil(depth),
            ),
        );

        assert_eq!(
            format!("{:?}", planned_graph),
            format!("{:?}", manual_graph)
        );
    }

    #[test]
    #[ignore] // CI can't run tests requiring actual backend
    fn main_pass_surface_plan() {
        use winit::{EventsLoop, WindowBuilder};

        let ev_loop = EventsLoop::new();
        let mut window_builder = WindowBuilder::new();
        window_builder.window.visible = false;
        let window = window_builder.build(&ev_loop).unwrap();

        let size = window
            .get_inner_size()
            .unwrap()
            .to_physical(window.get_hidpi_factor());
        let window_kind = crate::Kind::D2(size.width as u32, size.height as u32, 1, 1);

        let config: rendy::factory::Config = Default::default();
        let (mut factory, families): (Factory<DefaultBackend>, _) =
            rendy::factory::init(config).unwrap();
        let mut plan = RenderPlan::<DefaultBackend>::new();

        let surface1 = factory.create_surface(&window);
        let surface2 = factory.create_surface(&window);

        plan.extend_target(Target::Main, |ctx| {
            ctx.add(RenderOrder::Opaque, TestGroup2.builder())?;
            Ok(())
        });

        plan.add_root(Target::Main);
        plan.define_pass(
            Target::Main,
            TargetPlanOutputs {
                colors: vec![OutputColor::Surface(surface1, None)],
                depth: Some(ImageOptions {
                    kind: window_kind,
                    levels: 1,
                    format: Format::D32Sfloat,
                    clear: Some(ClearValue::DepthStencil(ClearDepthStencil(1.0, 0))),
                }),
            },
        )
        .unwrap();

        plan.extend_target(Target::Main, |ctx| {
            ctx.add(RenderOrder::Transparent, TestGroup1.builder())?;
            Ok(())
        });

        let planned_graph = plan.build(&factory).unwrap();

        let mut manual_graph = GraphBuilder::<DefaultBackend, World>::new();
        let depth = manual_graph.create_image(
            window_kind,
            1,
            Format::D32Sfloat,
            Some(ClearValue::DepthStencil(ClearDepthStencil(1.0, 0))),
        );
        manual_graph.add_node(
            RenderPassNodeBuilder::new()
                .with_subpass(
                    SubpassBuilder::new()
                        .with_group(TestGroup2.builder())
                        .with_group(TestGroup1.builder())
                        .with_color_surface()
                        .with_depth_stencil(depth),
                )
                .with_surface(surface2, None),
        );

        assert_eq!(
            format!("{:?}", planned_graph),
            format!("{:?}", manual_graph)
        );
    }

    #[test]
    #[ignore] // CI can't run tests requiring actual backend
    fn transitive_dependency() {
        let config: rendy::factory::Config = Default::default();
        let (factory, families): (Factory<DefaultBackend>, _) =
            rendy::factory::init(config).unwrap();
        let mut plan = RenderPlan::<DefaultBackend>::new();

        let kind = crate::Kind::D2(1920, 1080, 1, 1);
        plan.add_root(Target::Main);
        plan.define_pass(
            Target::Main,
            TargetPlanOutputs {
                colors: vec![OutputColor::Image(ImageOptions {
                    kind,
                    levels: 1,
                    format: Format::Rgb8Unorm,
                    clear: None,
                })],
                depth: Some(ImageOptions {
                    kind,
                    levels: 1,
                    format: Format::D32Sfloat,
                    clear: Some(ClearValue::DepthStencil(ClearDepthStencil(1.0, 0))),
                }),
            },
        )
        .unwrap();

        plan.define_pass(
            Target::Custom("target2"),
            TargetPlanOutputs {
                colors: vec![OutputColor::Image(ImageOptions {
                    kind,
                    levels: 1,
                    format: Format::Rgb8Unorm,
                    clear: None,
                })],
                depth: Some(ImageOptions {
                    kind,
                    levels: 1,
                    format: Format::D32Sfloat,
                    clear: Some(ClearValue::DepthStencil(ClearDepthStencil(1.0, 0))),
                }),
            },
        )
        .unwrap();

        plan.define_pass(
            Target::Custom("target3"),
            TargetPlanOutputs {
                colors: vec![OutputColor::Image(ImageOptions {
                    kind,
                    levels: 1,
                    format: Format::Rgb8Unorm,
                    clear: None,
                })],
                depth: Some(ImageOptions {
                    kind,
                    levels: 1,
                    format: Format::D32Sfloat,
                    clear: Some(ClearValue::DepthStencil(ClearDepthStencil(1.0, 0))),
                }),
            },
        )
        .unwrap();

        plan.extend_target(Target::Main, |ctx| {
            let _img = ctx.get_image(TargetImage::Color(Target::Custom("target2"), 0));
            ctx.add(RenderOrder::Transparent, TestGroup1.builder())?;
            ctx.add(RenderOrder::Opaque, TestGroup2.builder())?;
            Ok(())
        });

        plan.extend_target(Target::Custom("target2"), |ctx| {
            let _img = ctx.get_image(TargetImage::Color(Target::Custom("target3"), 0));
            ctx.add(RenderOrder::Transparent, TestGroup1.builder())?;
            ctx.add(RenderOrder::Opaque, TestGroup2.builder())?;
            Ok(())
        });

        plan.extend_target(Target::Custom("target3"), |ctx| {
            ctx.add(RenderOrder::Transparent, TestGroup1.builder())?;
            ctx.add(RenderOrder::Opaque, TestGroup2.builder())?;
            Ok(())
        });
        let planned_graph = plan.build(&factory).unwrap();

        let mut manual_graph = GraphBuilder::<DefaultBackend, Resources>::new();
        let color0 = manual_graph.create_image(kind, 1, Format::Rgb8Unorm, None);
        let depth0 = manual_graph.create_image(
            kind,
            1,
            Format::D32Sfloat,
            Some(ClearValue::DepthStencil(ClearDepthStencil(1.0, 0))),
        );
        let color1 = manual_graph.create_image(kind, 1, Format::Rgb8Unorm, None);
        let depth1 = manual_graph.create_image(
            kind,
            1,
            Format::D32Sfloat,
            Some(ClearValue::DepthStencil(ClearDepthStencil(1.0, 0))),
        );
        let color2 = manual_graph.create_image(kind, 1, Format::Rgb8Unorm, None);
        let depth2 = manual_graph.create_image(
            kind,
            1,
            Format::D32Sfloat,
            Some(ClearValue::DepthStencil(ClearDepthStencil(1.0, 0))),
        );
        let target3 = manual_graph.add_node(
            RenderPassNodeBuilder::new().with_subpass(
                SubpassBuilder::new()
                    .with_group(TestGroup2.builder())
                    .with_group(TestGroup1.builder())
                    .with_color(color0)
                    .with_depth_stencil(depth0),
            ),
        );

        let target2 = manual_graph.add_node(
            RenderPassNodeBuilder::new().with_subpass(
                SubpassBuilder::new()
                    .with_group(TestGroup2.builder())
                    .with_group(TestGroup1.builder())
                    .with_color(color1)
                    .with_depth_stencil(depth1)
                    .with_dependency(target3),
            ),
        );

        let main_node = manual_graph.add_node(
            RenderPassNodeBuilder::new().with_subpass(
                SubpassBuilder::new()
                    .with_group(TestGroup2.builder())
                    .with_group(TestGroup1.builder())
                    .with_color(color2)
                    .with_depth_stencil(depth2)
                    .with_dependency(target2),
            ),
        );

        assert_eq!(
            format!("{:?}", planned_graph),
            format!("{:?}", manual_graph)
        );
    }
}
