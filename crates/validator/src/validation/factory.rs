//! Hardware Validator Factory
//!
//! Factory for creating hardware validators with different configurations.

use super::{types::*, validator::HardwareValidator};
use crate::bittensor_core::WeightSetter;
use crate::persistence::SimplePersistence;
use std::path::PathBuf;
use std::sync::Arc;

/// Factory for creating hardware validators
pub struct HardwareValidatorFactory;

impl HardwareValidatorFactory {
    /// Create hardware validator with default configuration
    pub async fn create_default(
        persistence: Arc<SimplePersistence>,
    ) -> ValidationResult<HardwareValidator> {
        HardwareValidator::new(ValidationConfig::default(), persistence).await
    }

    /// Create hardware validator with custom configuration
    pub async fn create_with_config(
        config: ValidationConfig,
        persistence: Arc<SimplePersistence>,
    ) -> ValidationResult<HardwareValidator> {
        HardwareValidator::new(config, persistence).await
    }

    /// Create hardware validator with custom gpu-attestor binary path
    pub async fn create_with_binary_path(
        binary_path: PathBuf,
        persistence: Arc<SimplePersistence>,
    ) -> ValidationResult<HardwareValidator> {
        let config = ValidationConfig {
            gpu_attestor_binary_path: binary_path,
            ..Default::default()
        };
        HardwareValidator::new(config, persistence).await
    }

    /// Create hardware validator with weight setter for scoring integration
    pub async fn create_with_weight_setter(
        config: ValidationConfig,
        persistence: Arc<SimplePersistence>,
        weight_setter: Arc<WeightSetter>,
    ) -> ValidationResult<HardwareValidator> {
        HardwareValidator::with_weight_setter(config, persistence, Some(weight_setter)).await
    }
}
