//! API module that re-exports the generated metadata
//! This allows us to use our own metadata instead of crabtensor's built-in metadata

// Include the generated metadata from our build script
include!(concat!(env!("OUT_DIR"), "/metadata.rs"));

// Re-export the runtime types that crabtensor expects
pub use self::runtime_types::*;
