use crate::storage::ProcessingState;
use std::sync::Arc;

use amethyst_core::ecs::storage::UnprotectedStorage;
use amethyst_error::{Error, ResultExt};

#[cfg(feature = "profiler")]
use thread_profiler::profile_scope;

use crate::{Handle, Reload, SingleFile, Source};

/// One of the three core traits of this crate.
///
/// You want to implement this for every type of asset like
///
/// * `Mesh`
/// * `Texture`
/// * `Terrain`
///
/// and so on. Now, an asset may be available in different formats.
/// That's why we have the `Data` associated type here. You can specify
/// an intermediate format here, like the vertex data for a mesh or the samples
/// for audio data.
///
/// This data is then generated by the `Format` trait.
pub trait Asset: Send + Sync + 'static {
    /// An identifier for this asset used for debugging.
    const NAME: &'static str;

    /// The `Data` type the asset can be created from.
    type Data: Send + Sync + 'static;

    /// The ECS storage type to be used. You'll want to use `DenseVecStorage` in most cases.
    type HandleStorage: UnprotectedStorage<Handle<Self>> + Send + Sync;
}

/// Defines a way to process asset's data into the asset. This allows
/// using default `Processor` system to process assets that implement that type.
pub trait ProcessableAsset: Asset + Sized {
    /// Processes asset data into asset during loading.
    fn process(data: Self::Data) -> Result<ProcessingState<Self>, Error>;
}

impl<T: Asset<Data = T>> ProcessableAsset for T {
    fn process(data: Self::Data) -> Result<ProcessingState<Self>, Error> {
        Ok(ProcessingState::Loaded(data))
    }
}

/// A format, providing a conversion from bytes to asset data, which is then
/// in turn accepted by `Asset::from_data`. Examples for formats are
/// `Png`, `Obj` and `Wave`.
pub trait Format<A: Asset>: Send + 'static {
    /// A unique identifier for this format.
    const NAME: &'static str;
    /// Options specific to the format, which are passed to `import`.
    /// E.g. for textures this would be stuff like mipmap levels and
    /// sampler info.
    type Options: Send + 'static;

    /// Reads the given bytes and produces asset data.
    ///
    /// ## Reload
    ///
    /// The reload structure has metadata which allows the asset management
    /// to reload assets if necessary (for hot reloading).
    /// You should only create this if `create_reload` is `true`.
    /// Also, the parameter is just a request, which means you can also return `None`.
    fn import(
        &self,
        name: String,
        source: Arc<dyn Source>,
        options: Self::Options,
        create_reload: bool,
    ) -> Result<FormatValue<A>, Error>;
}

/// The `Ok` return value of `Format::import` for a given asset type `A`.
pub struct FormatValue<A: Asset> {
    /// The format data.
    pub data: A::Data,
    /// An optional reload structure
    pub reload: Option<Box<dyn Reload<A>>>,
}

impl<A: Asset> FormatValue<A> {
    /// Creates a `FormatValue` from only the data (setting `reload` to `None`).
    pub fn data(data: A::Data) -> Self {
        FormatValue { data, reload: None }
    }
}

/// This is a simplified version of `Format`, which doesn't give you as much freedom,
/// but in return is simpler to implement.
/// All `SimpleFormat` types automatically implement `Format`.
/// This format assumes that the asset name is the full path and the asset is only
/// contained in one file.
pub trait SimpleFormat<A: Asset> {
    /// A unique identifier for this format.
    const NAME: &'static str;
    /// Options specific to the format, which are passed to `import`.
    /// E.g. for textures this would be stuff like mipmap levels and
    /// sampler info.
    type Options: Clone + Send + Sync + 'static;

    /// Produces asset data from given bytes.
    fn import(&self, bytes: Vec<u8>, options: Self::Options) -> Result<A::Data, Error>;
}

impl<A, T> Format<A> for T
where
    A: Asset,
    T: SimpleFormat<A> + Clone + Send + Sync + 'static,
{
    const NAME: &'static str = T::NAME;
    type Options = T::Options;

    fn import(
        &self,
        name: String,
        source: Arc<dyn Source>,
        options: Self::Options,
        create_reload: bool,
    ) -> Result<FormatValue<A>, Error> {
        #[cfg(feature = "profiler")]
        profile_scope!("import_asset");
        if create_reload {
            let (b, m) = source
                .load_with_metadata(&name)
                .with_context(|_| crate::error::Error::Source)?;
            let data = T::import(&self, b, options.clone())?;
            let reload = SingleFile::new(self.clone(), m, options, name, source);
            let reload = Some(Box::new(reload) as Box<dyn Reload<A>>);
            Ok(FormatValue { data, reload })
        } else {
            let b = source
                .load(&name)
                .with_context(|_| crate::error::Error::Source)?;
            let data = T::import(&self, b, options)?;

            Ok(FormatValue::data(data))
        }
    }
}
