use std::fmt;

use serde::{Deserialize, Serialize};

use crate::compute::Resources;

// ============================================================================
// Rental Name
// ============================================================================

const DEFAULT_RENTAL_NAME_PREFIX: &str = "rental-";
pub const MAX_RENTAL_NAME_LEN: usize = 64;
const DEFAULT_RENTAL_ID_CHARS: usize = MAX_RENTAL_NAME_LEN - DEFAULT_RENTAL_NAME_PREFIX.len();

/// A validated rental name.
///
/// Invariants enforced at construction:
/// - 1..=64 characters
/// - Starts with a lowercase ASCII letter or digit
/// - Contains only lowercase ASCII letters, digits, dash, or underscore
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct RentalName(String);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InvalidRentalName(pub String);

impl fmt::Display for InvalidRentalName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for InvalidRentalName {}

impl RentalName {
    pub fn new(name: impl Into<String>) -> Result<Self, InvalidRentalName> {
        let name = name.into();
        let mut chars = name.chars();

        let Some(first) = chars.next() else {
            return Err(InvalidRentalName("Rental name cannot be empty".to_string()));
        };

        if name.len() > MAX_RENTAL_NAME_LEN {
            return Err(InvalidRentalName(
                "Rental name cannot exceed 64 characters".to_string(),
            ));
        }

        if !first.is_ascii_lowercase() && !first.is_ascii_digit() {
            return Err(InvalidRentalName(
                "Rental name must start with a lowercase letter or digit".to_string(),
            ));
        }

        if !chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-' || c == '_') {
            return Err(InvalidRentalName(
                "Rental name may only contain lowercase letters, digits, dash, or underscore"
                    .to_string(),
            ));
        }

        Ok(Self(name))
    }

    /// Generate a default name from a rental ID. Always valid by construction.
    pub fn default_from_id(rental_id: &str) -> Self {
        let suffix: String = rental_id.chars().take(DEFAULT_RENTAL_ID_CHARS).collect();
        Self(format!("{DEFAULT_RENTAL_NAME_PREFIX}{suffix}"))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn into_inner(self) -> String {
        self.0
    }
}

impl TryFrom<String> for RentalName {
    type Error = InvalidRentalName;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl From<RentalName> for String {
    fn from(name: RentalName) -> Self {
        name.0
    }
}

impl fmt::Display for RentalName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Generate a default rental name string from a rental ID.
///
/// Prefer `RentalName::default_from_id` when you need a validated type.
/// This function is kept for contexts that work with raw strings (e.g. DB fallbacks).
pub fn generate_default_rental_name(rental_id: &str) -> String {
    let suffix: String = rental_id.chars().take(DEFAULT_RENTAL_ID_CHARS).collect();
    format!("{DEFAULT_RENTAL_NAME_PREFIX}{suffix}")
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum AccessType {
    #[serde(rename = "ssh")]
    Ssh,
    #[serde(rename = "jupyter")]
    Jupyter,
    #[serde(rename = "vscode")]
    Vscode,
    #[serde(rename = "custom")]
    Custom,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RentalDuration {
    pub hours: u32,
    #[serde(default)]
    pub auto_extend: bool,
    #[serde(default)]
    pub max_extensions: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct VolumeMount {
    pub host_path: Option<String>, // hostPath generally disallowed; prefer PVC
    pub container_path: String,
    #[serde(default)]
    pub read_only: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RentalContainer {
    pub image: String,
    #[serde(default)]
    pub env: Vec<(String, String)>,
    #[serde(default)]
    pub command: Vec<String>,
    #[serde(default)]
    pub ports: Vec<RentalPort>,
    #[serde(default)]
    pub volumes: Vec<VolumeMount>,
    pub resources: Resources,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RentalPort {
    pub container_port: u16,
    #[serde(default = "default_protocol")]
    pub protocol: String, // TCP | UDP
}

fn default_protocol() -> String {
    "TCP".into()
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct IngressRule {
    pub port: u16,
    pub exposure: String, // NodePort | LoadBalancer | ClusterIP
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RentalNetwork {
    #[serde(default)]
    pub ingress: Vec<IngressRule>,
    #[serde(default)]
    pub egress_policy: String, // restricted | egress-only | open
    #[serde(default)]
    pub allowed_egress: Vec<String>,
    #[serde(default)]
    pub public_ip_required: bool,
    #[serde(default)]
    pub bandwidth_mbps: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RentalSsh {
    pub enabled: bool,
    pub public_key: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RentalJupyter {
    pub password: Option<String>,
    pub token: Option<String>,
    pub base_image: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RentalEnvironment {
    pub base_image: Option<String>,
    pub pre_install_script: Option<String>,
    #[serde(default)]
    pub environment_variables: Vec<(String, String)>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RentalStorage {
    pub persistent_volume_gb: u32,
    pub storage_class: Option<String>,
    pub mount_path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MinerSelector {
    pub id: Option<String>,
    pub region: Option<String>,
    pub tier: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RentalBilling {
    pub max_hourly_rate: f64,
    pub payment_method: String,
    pub account_id: Option<String>,
    pub deposit_amount: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RentalSpec {
    pub container: RentalContainer,
    pub duration: RentalDuration,
    pub access_type: AccessType,
    pub network: RentalNetwork,
    pub storage: Option<RentalStorage>,
    pub ssh: Option<RentalSsh>,
    pub jupyter: Option<RentalJupyter>,
    pub environment: Option<RentalEnvironment>,
    pub miner_selector: Option<MinerSelector>,
    pub billing: Option<RentalBilling>,
    #[serde(default)]
    pub ttl_seconds: u32,
    // Tenancy is carried in the API layer (namespace); include optional reference here if needed.
    pub tenancy: Option<(String, String)>, // (user_id, project_id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::{GpuSpec, Resources};

    #[test]
    fn serde_round_trip_rental_spec() {
        let spec = RentalSpec {
            container: RentalContainer {
                image: "image".into(),
                env: vec![("A".into(), "1".into())],
                command: vec!["bash".into()],
                ports: vec![RentalPort {
                    container_port: 8080,
                    protocol: "TCP".into(),
                }],
                volumes: vec![VolumeMount {
                    host_path: None,
                    container_path: "/data".into(),
                    read_only: false,
                }],
                resources: Resources {
                    cpu: "4".into(),
                    memory: "16Gi".into(),
                    gpus: GpuSpec {
                        count: 1,
                        model: vec!["A100".into()],
                    },
                },
            },
            duration: RentalDuration {
                hours: 24,
                auto_extend: false,
                max_extensions: 0,
            },
            access_type: AccessType::Ssh,
            network: RentalNetwork {
                ingress: vec![IngressRule {
                    port: 8080,
                    exposure: "NodePort".into(),
                }],
                egress_policy: "restricted".into(),
                allowed_egress: vec!["https://s3.amazonaws.com".into()],
                public_ip_required: false,
                bandwidth_mbps: Some(100),
            },
            storage: Some(RentalStorage {
                persistent_volume_gb: 200,
                storage_class: Some("fast-ssd".into()),
                mount_path: "/data".into(),
            }),
            ssh: Some(RentalSsh {
                enabled: true,
                public_key: "ssh-ed25519 AAAA...".into(),
            }),
            jupyter: None,
            environment: Some(RentalEnvironment {
                base_image: Some("nvidia/cuda:12.0-base".into()),
                pre_install_script: None,
                environment_variables: vec![],
            }),
            miner_selector: Some(MinerSelector {
                id: None,
                region: Some("us-east-1".into()),
                tier: Some("premium".into()),
            }),
            billing: Some(RentalBilling {
                max_hourly_rate: 3.5,
                payment_method: "postpaid".into(),
                account_id: Some("acct_1".into()),
                deposit_amount: None,
            }),
            ttl_seconds: 0,
            tenancy: Some(("user-1".into(), "proj-1".into())),
        };

        let json = serde_json::to_string(&spec).unwrap();
        let de: RentalSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(de, spec);
    }
}
