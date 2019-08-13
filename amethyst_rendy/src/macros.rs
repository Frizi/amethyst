macro_rules! set_layout {
    ($factory:expr, $([$times:expr] $ty:ident $flags:expr),*) => {
        $factory.create_descriptor_set_layout(
            crate::util::set_layout_bindings(
                std::iter::empty()
                    $(.chain(std::iter::once((
                        $times as u32,
                        rendy::hal::pso::DescriptorType::$ty,
                        $flags
                    ))))*
            )
        )?.into()
    }
}

#[macro_export]
macro_rules! include_shader {
    ($path:literal) => {{
        let path = $path;
        let flags = if path.ends_with(".vert.spv") {
            $crate::rendy::hal::pso::ShaderStageFlags::VERTEX
        } else if path.ends_with(".frag.spv") {
            $crate::rendy::hal::pso::ShaderStageFlags::FRAGMENT
        } else if path.ends_with(".geom.spv") {
            $crate::rendy::hal::pso::ShaderStageFlags::GEOMETRY
        } else {
            panic!("Cannot identify shader type based on path {}", path);
        };

        let bytes = include_bytes!($path);
        assert!(bytes.len() % 4 == 0);
        let data: &[u32] =
            unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const u32, bytes.len() / 4) };
        $crate::rendy::shader::SpirvShader::new(Vec::from(data), flags, "main")
    }};
}

#[macro_export]
macro_rules! define_shaders {
    {$($([$($vis:tt)*])* static ref $name:ident <- $path:literal;)*} => {
        $crate::lazy_static::lazy_static! {
            $(
            $($($vis)*)* static ref $name: $crate::rendy::shader::SpirvShader = $crate::include_shader!($path);
            )*
        }
    }
}
