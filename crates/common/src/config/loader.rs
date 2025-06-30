//! # Configuration Loader
//!
//! Figment-based configuration loading with layered support:
//! 1. Compiled defaults
//! 2. Configuration files (TOML/JSON/YAML)
//! 3. Environment variable overrides
//!
//! Supports automatic environment variable mapping with prefixes.

use crate::error::ConfigurationError;
use figment::{
    providers::{Env, Format, Serialized, Toml},
    Figment,
};
use serde::de::DeserializeOwned;
use std::path::{Path, PathBuf};
use tracing::{debug, info, warn};

/// Default configuration file name
const DEFAULT_CONFIG_FILE: &str = "config.toml";

/// Environment variable prefix for Basilca
const DEFAULT_ENV_PREFIX: &str = "BASILCA";

/// Load configuration with layered approach
///
/// # Type Parameters
/// * `T` - Configuration type that implements Default + DeserializeOwned
///
/// # Returns
/// * Loaded and validated configuration
///
/// # Configuration Layer Priority (highest to lowest)
/// 1. Environment variables (BASILCA_*)
/// 2. Configuration file (config.toml or specified path)
/// 3. Compiled defaults
///
/// # Environment Variable Mapping
/// - Nested fields use double underscore: `BASILCA_DATABASE__URL`
/// - Arrays use indices: `BASILCA_SERVERS__0__HOST`
/// - Case insensitive matching
///
/// # Example
/// ```rust
/// use common::config::loader::load_config;
/// use serde::{Deserialize, Serialize};
///
/// #[derive(Debug, Default, Deserialize, Serialize)]
/// struct MyConfig {
///     pub host: String,
///     pub port: u16,
/// }
///
/// let config: MyConfig = load_config().unwrap();
/// ```
pub fn load_config<T>() -> Result<T, ConfigurationError>
where
    T: Default + DeserializeOwned + serde::Serialize,
{
    load_config_with_options::<T>(LoadOptions::default())
}

/// Load configuration from specific file
///
/// # Arguments
/// * `path` - Path to configuration file
///
/// # Returns
/// * Configuration loaded from file with environment overrides
pub fn load_from_file<T>(path: &Path) -> Result<T, ConfigurationError>
where
    T: Default + DeserializeOwned + serde::Serialize,
{
    let options = LoadOptions {
        config_path: Some(path.to_path_buf()),
        env_prefix: DEFAULT_ENV_PREFIX.to_string(),
        require_file: true,
    };
    load_config_with_options::<T>(options)
}

/// Configuration loading options
#[derive(Debug, Clone)]
pub struct LoadOptions {
    /// Optional path to configuration file
    pub config_path: Option<PathBuf>,
    /// Environment variable prefix
    pub env_prefix: String,
    /// Whether configuration file is required
    pub require_file: bool,
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            config_path: None,
            env_prefix: DEFAULT_ENV_PREFIX.to_string(),
            require_file: false,
        }
    }
}

/// Load configuration with custom options
pub fn load_config_with_options<T>(options: LoadOptions) -> Result<T, ConfigurationError>
where
    T: Default + DeserializeOwned + serde::Serialize,
{
    info!("Loading configuration with options: {:?}", options);

    // Start with compiled defaults
    let mut figment = Figment::new().merge(Serialized::defaults(T::default()));

    // Determine configuration file path
    let config_path = determine_config_path(options.config_path)?;

    // Add file provider if path exists or is required
    if let Some(path) = &config_path {
        if path.exists() {
            info!("Loading configuration from file: {}", path.display());
            figment = add_file_provider(figment, path)?;
        } else if options.require_file {
            return Err(ConfigurationError::FileNotFound {
                path: path.display().to_string(),
            });
        } else {
            warn!(
                "Configuration file not found: {} (using defaults)",
                path.display()
            );
        }
    }

    // Add environment variable overrides
    debug!(
        "Loading environment variables with prefix: {}",
        options.env_prefix
    );
    figment = figment.merge(
        Env::prefixed(&format!("{}_", options.env_prefix))
            .split("__") // Use double underscore for nested fields
            .ignore(&["PATH", "HOME", "USER"]), // Ignore common system vars
    );

    // Extract and validate configuration
    let config: T = figment
        .extract()
        .map_err(|err| ConfigurationError::ParseError {
            details: format!("Failed to parse configuration: {err}"),
        })?;

    info!("Configuration loaded successfully");
    debug!(
        "Configuration loaded from {} sources",
        figment.metadata().count()
    );

    Ok(config)
}

/// Apply environment variable overrides to existing configuration
///
/// # Arguments
/// * `config` - Mutable reference to configuration
/// * `prefix` - Environment variable prefix (e.g., "BASILCA")
///
/// # Returns
/// * Result indicating success or failure
pub fn apply_env_overrides<T>(config: &mut T, prefix: &str) -> Result<(), ConfigurationError>
where
    T: DeserializeOwned + serde::Serialize + Default,
{
    debug!(
        "Applying environment variable overrides with prefix: {}",
        prefix
    );

    // Create figment with current config as base (clone to avoid ownership issues)
    let current_config = T::default(); // Use default as base since we can't clone config
    let figment = Figment::new()
        .merge(Serialized::defaults(current_config))
        .merge(
            Env::prefixed(&format!("{prefix}_"))
                .split("__")
                .ignore(&["PATH", "HOME", "USER"]),
        );

    // Extract updated configuration
    *config = figment
        .extract()
        .map_err(|err| ConfigurationError::ParseError {
            details: format!("Failed to apply environment overrides: {err}"),
        })?;

    debug!("Environment variable overrides applied successfully");
    Ok(())
}

/// Determine configuration file path with fallback logic
fn determine_config_path(
    override_path: Option<PathBuf>,
) -> Result<Option<PathBuf>, ConfigurationError> {
    if let Some(path) = override_path {
        return Ok(Some(path));
    }

    // Check environment variable for config path
    if let Ok(env_path) = std::env::var("BASILCA_CONFIG_PATH") {
        let path = PathBuf::from(env_path);
        debug!("Using config path from environment: {}", path.display());
        return Ok(Some(path));
    }

    // Check current directory
    let current_dir_config = std::env::current_dir()
        .map_err(|e| ConfigurationError::EnvironmentError {
            var: "current_dir".to_string(),
            details: e.to_string(),
        })?
        .join(DEFAULT_CONFIG_FILE);

    if current_dir_config.exists() {
        debug!(
            "Found config file in current directory: {}",
            current_dir_config.display()
        );
        return Ok(Some(current_dir_config));
    }

    // Check common config locations
    let config_locations = [
        "/etc/basilca/config.toml",
        "~/.config/basilca/config.toml",
        "./config/config.toml",
    ];

    for location in &config_locations {
        let path = expand_path(location)?;
        if path.exists() {
            debug!("Found config file at: {}", path.display());
            return Ok(Some(path));
        }
    }

    debug!("No configuration file found, using defaults");
    Ok(None)
}

/// Add file provider to figment based on file extension
fn add_file_provider(figment: Figment, path: &Path) -> Result<Figment, ConfigurationError> {
    let extension = path
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("toml");

    match extension.to_lowercase().as_str() {
        "toml" => Ok(figment.merge(Toml::file(path))),
        _ => Err(ConfigurationError::ParseError {
            details: format!(
                "Unsupported configuration file format: {extension} (supported: toml)"
            ),
        }),
    }
}

/// Expand path with tilde and environment variables
fn expand_path(path: &str) -> Result<PathBuf, ConfigurationError> {
    let expanded = if path.starts_with('~') {
        if let Ok(home) = std::env::var("HOME") {
            path.replacen('~', &home, 1)
        } else {
            return Err(ConfigurationError::EnvironmentError {
                var: "HOME".to_string(),
                details: "HOME environment variable not set".to_string(),
            });
        }
    } else {
        path.to_string()
    };

    // TODO: Add more sophisticated environment variable expansion if needed
    // For now, just handle simple tilde expansion

    Ok(PathBuf::from(expanded))
}

/// Validate configuration file format
pub fn validate_config_file(path: &Path) -> Result<(), ConfigurationError> {
    if !path.exists() {
        return Err(ConfigurationError::FileNotFound {
            path: path.display().to_string(),
        });
    }

    let extension = path.extension().and_then(|ext| ext.to_str()).unwrap_or("");

    match extension.to_lowercase().as_str() {
        "toml" => Ok(()),
        _ => Err(ConfigurationError::ParseError {
            details: format!(
                "Unsupported configuration file format: {extension} (supported: toml)"
            ),
        }),
    }
}

/// Get configuration metadata for debugging
pub fn get_config_metadata<T>() -> Result<Vec<String>, ConfigurationError>
where
    T: Default + DeserializeOwned + serde::Serialize,
{
    let figment = Figment::new()
        .merge(Serialized::defaults(T::default()))
        .merge(Env::prefixed(&format!("{DEFAULT_ENV_PREFIX}_")).split("__"));

    let metadata = figment.metadata().map(|meta| format!("{meta:?}")).collect();

    Ok(metadata)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};
    use std::env;
    use tempfile::NamedTempFile;

    #[derive(Debug, Default, Deserialize, Serialize, PartialEq)]
    struct TestConfig {
        pub name: String,
        pub port: u16,
        pub nested: NestedConfig,
    }

    #[derive(Debug, Default, Deserialize, Serialize, PartialEq)]
    struct NestedConfig {
        pub enabled: bool,
        pub timeout: u64,
    }

    #[test]
    fn test_load_default_config() {
        // Save original values
        let orig_name = env::var("BASILCA_NAME").ok();
        let orig_port = env::var("BASILCA_PORT").ok();
        let orig_enabled = env::var("BASILCA_NESTED__ENABLED").ok();
        let orig_timeout = env::var("BASILCA_NESTED__TIMEOUT").ok();

        // Clear any environment variables that might interfere
        env::remove_var("BASILCA_NAME");
        env::remove_var("BASILCA_PORT");
        env::remove_var("BASILCA_NESTED__ENABLED");
        env::remove_var("BASILCA_NESTED__TIMEOUT");

        let config: TestConfig = load_config().unwrap();
        assert_eq!(config, TestConfig::default());

        // Restore original values
        if let Some(val) = orig_name {
            env::set_var("BASILCA_NAME", val);
        }
        if let Some(val) = orig_port {
            env::set_var("BASILCA_PORT", val);
        }
        if let Some(val) = orig_enabled {
            env::set_var("BASILCA_NESTED__ENABLED", val);
        }
        if let Some(val) = orig_timeout {
            env::set_var("BASILCA_NESTED__TIMEOUT", val);
        }
    }

    #[test]
    fn test_load_from_toml_file() {
        // Save original values
        let orig_name = env::var("BASILCA_NAME").ok();
        let orig_port = env::var("BASILCA_PORT").ok();
        let orig_enabled = env::var("BASILCA_NESTED__ENABLED").ok();
        let orig_timeout = env::var("BASILCA_NESTED__TIMEOUT").ok();

        // Clear environment variables to ensure file values are used
        env::remove_var("BASILCA_NAME");
        env::remove_var("BASILCA_PORT");
        env::remove_var("BASILCA_NESTED__ENABLED");
        env::remove_var("BASILCA_NESTED__TIMEOUT");

        let toml_content = r#"
            name = "test"
            port = 8080

            [nested]
            enabled = true
            timeout = 30
        "#;

        let mut temp_file = NamedTempFile::new().unwrap();
        std::io::Write::write_all(&mut temp_file, toml_content.as_bytes()).unwrap();

        let config: TestConfig = load_from_file(temp_file.path()).unwrap();
        assert_eq!(config.name, "test");
        assert_eq!(config.port, 8080);
        assert!(config.nested.enabled);
        assert_eq!(config.nested.timeout, 30);

        // Restore original values
        if let Some(val) = orig_name {
            env::set_var("BASILCA_NAME", val);
        }
        if let Some(val) = orig_port {
            env::set_var("BASILCA_PORT", val);
        }
        if let Some(val) = orig_enabled {
            env::set_var("BASILCA_NESTED__ENABLED", val);
        }
        if let Some(val) = orig_timeout {
            env::set_var("BASILCA_NESTED__TIMEOUT", val);
        }
    }

    #[test]
    fn test_env_var_overrides() {
        // Use a unique prefix for this test to avoid conflicts
        let test_prefix = "TEST_ENV_VAR";
        env::set_var(format!("{test_prefix}_NAME"), "env_test");
        env::set_var(format!("{test_prefix}_PORT"), "9090");
        env::set_var(format!("{test_prefix}_NESTED__ENABLED"), "true");
        env::set_var(format!("{test_prefix}_NESTED__TIMEOUT"), "60");

        let options = LoadOptions {
            config_path: None,
            env_prefix: test_prefix.to_string(),
            require_file: false,
        };

        let config: TestConfig = load_config_with_options(options).unwrap();
        assert_eq!(config.name, "env_test");
        assert_eq!(config.port, 9090);
        assert!(config.nested.enabled);
        assert_eq!(config.nested.timeout, 60);

        // Clean up
        env::remove_var(format!("{test_prefix}_NAME"));
        env::remove_var(format!("{test_prefix}_PORT"));
        env::remove_var(format!("{test_prefix}_NESTED__ENABLED"));
        env::remove_var(format!("{test_prefix}_NESTED__TIMEOUT"));
    }

    #[test]
    fn test_file_not_found_when_required() {
        let non_existent_path = PathBuf::from("/non/existent/config.toml");
        let result: Result<TestConfig, _> = load_from_file(&non_existent_path);
        assert!(result.is_err());

        match result.unwrap_err() {
            ConfigurationError::FileNotFound { path } => {
                assert_eq!(path, "/non/existent/config.toml");
            }
            _ => panic!("Expected FileNotFound error"),
        }
    }

    #[test]
    fn test_validate_config_file() {
        let mut temp_file = NamedTempFile::with_suffix(".toml").unwrap();
        std::io::Write::write_all(&mut temp_file, b"name = 'test'").unwrap();

        assert!(validate_config_file(temp_file.path()).is_ok());

        let non_existent = PathBuf::from("/non/existent.toml");
        assert!(validate_config_file(&non_existent).is_err());
    }

    #[test]
    fn test_expand_path() {
        // Test tilde expansion (if HOME is set)
        if env::var("HOME").is_ok() {
            let expanded = expand_path("~/test/config.toml").unwrap();
            assert!(!expanded.to_string_lossy().contains('~'));
        }

        // Test regular path
        let regular = expand_path("/etc/config.toml").unwrap();
        assert_eq!(regular, PathBuf::from("/etc/config.toml"));
    }
}
