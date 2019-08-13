//! Environment submodule for shared environmental descriptor set data.
//! Fetches and sets projection and lighting descriptor set information.
use crate::{
    camera::Orthographic,
    light::Light,
    pod::{self, IntoPod},
    rendy::{
        command::RenderPassEncoder,
        factory::Factory,
        graph::NodeImage,
        hal::{self, adapter::PhysicalDevice, device::Device, pso::Descriptor},
        memory::Write as _,
        resource::{
            Buffer, DescriptorSet, DescriptorSetLayout, Escape, Handle as RendyHandle, ImageView,
            Sampler,
        },
    },
    submodules::gather::{AmbientGatherer, CameraGatherer},
    types::Backend,
    util::{self, TapCountIter},
};
use amethyst_core::{
    ecs::{Join, ReadStorage, SystemData, World},
    math::{convert, Matrix4, Point3, Vector3},
    transform::Transform,
};
use glsl_layout::*;

#[cfg(feature = "profiler")]
use thread_profiler::profile_scope;

const MAX_POINT_LIGHTS: usize = 128;
const MAX_DIR_LIGHTS: usize = 16;
const MAX_SPOT_LIGHTS: usize = 128;

/// Submodule for loading and binding descriptor sets for a 3D, lit environment.
/// This also abstracts away the need for handling multiple images in flight, as it provides
/// per-image submissions.
#[derive(Debug)]
pub struct EnvironmentSub<B: Backend> {
    layout: RendyHandle<DescriptorSetLayout<B>>,
    per_image: Vec<PerImageEnvironmentSub<B>>,
}

/// Submodule for loading and binding descriptor sets for a 3D, lit environment.
/// This is the actual implementation for a given environment, but multiple instances may exist
/// for each image in flight.
#[derive(Debug)]
struct PerImageEnvironmentSub<B: Backend> {
    buffer: Option<Escape<Buffer<B>>>,
    set: Escape<DescriptorSet<B>>,
}

impl<B: Backend> EnvironmentSub<B> {
    /// Create and allocate a new `EnvironmentSub` with the provided rendy `Factory`
    /// Allocate to the supplied shader.
    pub fn new(
        factory: &Factory<B>,
        flags: [hal::pso::ShaderStageFlags; 2],
    ) -> Result<Self, failure::Error> {
        Ok(Self {
            layout: set_layout! {
                factory,
                [1] UniformBuffer flags[0],
                [5] UniformBuffer flags[1],
                [1] CombinedImageSampler hal::pso::ShaderStageFlags::FRAGMENT
            },
            per_image: Vec::new(),
        })
    }

    /// Returns the raw `DescriptorSetLayout` for this environment
    pub fn raw_layout(&self) -> &B::DescriptorSetLayout {
        self.layout.raw()
    }

    /// Performs any re-allocation and GPU memory writing required for this environment set.
    pub fn process(
        &mut self,
        factory: &Factory<B>,
        index: usize,
        world: &World,
        shadow_map: Option<&(NodeImage, RendyHandle<Sampler<B>>, Escape<ImageView<B>>)>,
    ) -> bool {
        #[cfg(feature = "profiler")]
        profile_scope!("process");

        let this_image = {
            while self.per_image.len() <= index {
                self.per_image
                    .push(PerImageEnvironmentSub::new(factory, &self.layout));
            }
            &mut self.per_image[index]
        };
        this_image.process(factory, world, shadow_map)
    }

    /// Binds this environment set for all images.
    #[inline]
    pub fn bind(
        &self,
        index: usize,
        pipeline_layout: &B::PipelineLayout,
        set_id: u32,
        encoder: &mut RenderPassEncoder<'_, B>,
    ) {
        self.per_image[index].bind(pipeline_layout, set_id, encoder);
    }
}

impl<B: Backend> PerImageEnvironmentSub<B> {
    fn new(factory: &Factory<B>, layout: &RendyHandle<DescriptorSetLayout<B>>) -> Self {
        Self {
            buffer: None,
            set: factory.create_descriptor_set(layout.clone()).unwrap(),
        }
    }

    #[inline]
    fn bind(
        &self,
        pipeline_layout: &B::PipelineLayout,
        set_id: u32,
        encoder: &mut RenderPassEncoder<'_, B>,
    ) {
        unsafe {
            encoder.bind_graphics_descriptor_sets(
                pipeline_layout,
                set_id,
                Some(self.set.raw()),
                std::iter::empty(),
            );
        }
    }

    fn process(
        &mut self,
        factory: &Factory<B>,
        world: &World,
        shadow_map: Option<&(NodeImage, RendyHandle<Sampler<B>>, Escape<ImageView<B>>)>,
    ) -> bool {
        let align = factory
            .physical()
            .limits()
            .min_uniform_buffer_offset_alignment;

        let viewargs_size = util::align_size::<pod::ViewArgs>(align, 1);
        let env_buf_size = util::align_size::<pod::Environment>(align, 1);
        let plight_buf_size = util::align_size::<pod::PointLight>(align, MAX_POINT_LIGHTS);
        let dlight_buf_size = util::align_size::<pod::DirectionalLight>(align, MAX_DIR_LIGHTS);
        let slight_buf_size = util::align_size::<pod::SpotLight>(align, MAX_SPOT_LIGHTS);
        let shadow_buf_size = util::align_size::<pod::ShadowData>(align, 1);

        let viewargs_range = 0..viewargs_size;
        let env_range = util::next_range(&viewargs_range, env_buf_size);
        let plight_range = util::next_range(&env_range, plight_buf_size);
        let dlight_range = util::next_range(&plight_range, dlight_buf_size);
        let slight_range = util::next_range(&dlight_range, slight_buf_size);
        let shadow_range = util::next_range(&slight_range, shadow_buf_size);

        let whole_range = 0..shadow_range.end;

        let new_buffer = util::ensure_buffer(
            &factory,
            &mut self.buffer,
            hal::buffer::Usage::UNIFORM,
            rendy::memory::Dynamic,
            whole_range.end,
        )
        .unwrap();
        if let Some(buffer) = self.buffer.as_mut() {
            if new_buffer {
                use util::{desc_write, opt_range};
                let buffer = buffer.raw();
                let env_set = self.set.raw();

                let desc_viewargs = Descriptor::Buffer(buffer, opt_range(viewargs_range.clone()));
                let desc_env = Descriptor::Buffer(buffer, opt_range(env_range.clone()));
                let desc_plight = Descriptor::Buffer(buffer, opt_range(plight_range.clone()));
                let desc_dlight = Descriptor::Buffer(buffer, opt_range(dlight_range.clone()));
                let desc_slight = Descriptor::Buffer(buffer, opt_range(slight_range.clone()));
                let desc_shadow = Descriptor::Buffer(buffer, opt_range(shadow_range.clone()));

                let mut sets = vec![
                    desc_write(env_set, 0, desc_viewargs),
                    desc_write(env_set, 1, desc_env),
                    desc_write(env_set, 2, desc_plight),
                    desc_write(env_set, 3, desc_dlight),
                    desc_write(env_set, 4, desc_slight),
                    desc_write(env_set, 5, desc_shadow),
                ];

                if let Some(shadow_map) = shadow_map {
                    let desc_shadow_map = Descriptor::CombinedImageSampler(
                        shadow_map.2.raw(),
                        shadow_map.0.layout,
                        shadow_map.1.raw(),
                    );
                    sets.push(desc_write(env_set, 6, desc_shadow_map))
                }

                unsafe {
                    factory.write_descriptor_sets(sets);
                }
            }

            let CameraGatherer {
                camera_position,
                viewargs,
                camera_view,
                ..
            } = CameraGatherer::gather(world);

            let camera_view_inv = camera_view.try_inverse().unwrap();

            let mut mapped = buffer.map(factory, whole_range.clone()).unwrap();
            let mut writer = unsafe { mapped.write::<u8>(factory, whole_range.clone()).unwrap() };
            let dst_slice = unsafe { writer.slice() };

            let mut env = pod::Environment {
                ambient_color: AmbientGatherer::gather(world),
                camera_position,
                point_light_count: 0,
                directional_light_count: 0,
                shadow_count: 0,
                spot_light_count: 0,
            }
            .std140();

            let shadow_count_ref = &mut env.shadow_count;

            let (lights, transforms) =
                <(ReadStorage<'_, Light>, ReadStorage<'_, Transform>)>::fetch(world);

            let point_lights = (&lights, &transforms)
                .join()
                .filter_map(|(light, transform)| match light {
                    Light::Point(light) => Some(
                        pod::PointLight {
                            position: convert::<_, Vector3<f32>>(
                                transform.global_matrix().column(3).xyz(),
                            )
                            .into_pod(),
                            color: light.color.into_pod(),
                            intensity: light.intensity,
                        }
                        .std140(),
                    ),
                    _ => None,
                })
                .take(MAX_POINT_LIGHTS);

            let dir_lights = lights
                .join()
                .filter_map(|light| match light {
                    Light::Directional(ref light) => Some(
                        pod::DirectionalLight {
                            color: light.color.into_pod(),
                            intensity: light.intensity,
                            direction: light.direction.into_pod(),
                        }
                        .std140(),
                    ),
                    _ => None,
                })
                .take(MAX_DIR_LIGHTS);

            let spot_lights = (&lights, &transforms)
                .join()
                .filter_map(|(light, transform)| {
                    if let Light::Spot(ref light) = *light {
                        Some(
                            pod::SpotLight {
                                position: convert::<_, Vector3<f32>>(
                                    transform.global_matrix().column(3).xyz(),
                                )
                                .into_pod(),
                                color: light.color.into_pod(),
                                direction: light.direction.into_pod(),
                                angle: light.angle.cos(),
                                intensity: light.intensity,
                                range: light.range,
                                smoothness: light.smoothness,
                            }
                            .std140(),
                        )
                    } else {
                        None
                    }
                })
                .take(MAX_SPOT_LIGHTS);

            let dir_shadows = lights.join().filter_map(|light| match light {
                Light::Directional(ref light) if light.cast_shadow => {
                    let eye = Point3::new(0.0, 0.0, 0.0);
                    let target = eye + light.direction;
                    let light_view = Matrix4::<f32>::look_at_rh(&eye, &target, &Vector3::y());
                    let light_proj = Orthographic::new(-20.0, 20.0, -20.0, 20.0, -200.0, 200.0)
                        .as_matrix()
                        .clone();
                    let view_to_light = light_proj * light_view;
                    let view_to_light: [[f32; 4]; 4] = view_to_light.into();
                    Some(pod::ShadowData {
                        view_to_light: view_to_light.into(),
                        cascades: Default::default(), // TODO: actually prepare cascades
                    })
                }
                _ => None,
            });

            use util::{usize_range, write_into_slice};
            write_into_slice(
                &mut dst_slice[usize_range(plight_range)],
                point_lights.tap_count(&mut env.point_light_count),
            );
            write_into_slice(
                &mut dst_slice[usize_range(dlight_range)],
                dir_lights.tap_count(&mut env.directional_light_count),
            );
            write_into_slice(
                &mut dst_slice[usize_range(slight_range)],
                spot_lights.tap_count(&mut env.spot_light_count),
            );
            write_into_slice(
                &mut dst_slice[usize_range(shadow_range)],
                dir_shadows.tap_count(&mut env.shadow_count),
            );
            write_into_slice(&mut dst_slice[usize_range(viewargs_range)], Some(viewargs));
            write_into_slice(&mut dst_slice[usize_range(env_range)], Some(env));
        }

        new_buffer
    }
}
