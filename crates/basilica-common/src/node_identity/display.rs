//! Display formatting for node identities
//!
//! This module implements the IdentityDisplay trait to provide
//! consistent formatting for node identities across CLI commands.

use anyhow::Result;
use serde_json::json;

use crate::node_identity::interfaces::{IdentityDisplay, NodeIdentity};

/// Implementation of IdentityDisplay for any type implementing NodeIdentity
pub struct NodeIdentityDisplay<'a> {
    identity: &'a dyn NodeIdentity,
}

impl<'a> NodeIdentityDisplay<'a> {
    /// Creates a new display wrapper for a node identity
    pub fn new(identity: &'a dyn NodeIdentity) -> Self {
        Self { identity }
    }
}

impl<'a> IdentityDisplay for NodeIdentityDisplay<'a> {
    /// Returns short UUID for human readability
    ///
    /// This is the default display format for CLI commands
    fn format_compact(&self) -> String {
        self.identity.short_uuid()
    }

    /// Returns full UUID display
    ///
    /// Format: "UUID: 550e8400-e29b-41d4-a716-446655440000"
    fn format_verbose(&self) -> String {
        format!("UUID: {}", self.identity.uuid())
    }

    /// Returns JSON with UUID and creation time
    ///
    /// Format: {"uuid": "...", "created_at": "..."}
    fn format_json(&self) -> Result<String> {
        let json_obj = json!({
            "uuid": self.identity.uuid().to_string(),
            "created_at": self.identity.created_at()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
        });

        serde_json::to_string_pretty(&json_obj)
            .map_err(|e| anyhow::anyhow!("Failed to serialize to JSON: {}", e))
    }
}

/// Extension trait to add display methods directly to NodeIdentity types
pub trait NodeIdentityDisplayExt {
    /// Creates a display formatter for this identity
    fn display(&self) -> NodeIdentityDisplay<'_>;
}

impl<T: NodeIdentity> NodeIdentityDisplayExt for T {
    fn display(&self) -> NodeIdentityDisplay<'_> {
        NodeIdentityDisplay::new(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::SystemTime;
    use uuid::Uuid;

    struct MockIdentity {
        uuid: Uuid,
        created_at: SystemTime,
    }

    impl NodeIdentity for MockIdentity {
        fn uuid(&self) -> &Uuid {
            &self.uuid
        }

        fn created_at(&self) -> SystemTime {
            self.created_at
        }

        fn matches(&self, query: &str) -> bool {
            self.uuid.to_string().starts_with(query)
        }

        fn full_display(&self) -> String {
            self.uuid.to_string()
        }

        fn short_uuid(&self) -> String {
            self.uuid.to_string()[..8].to_string()
        }
    }

    #[test]
    fn test_format_compact() {
        let uuid = Uuid::new_v4();
        let identity = MockIdentity {
            uuid,
            created_at: SystemTime::now(),
        };

        let display = NodeIdentityDisplay::new(&identity);
        assert_eq!(display.format_compact(), uuid.to_string()[..8].to_string());
    }

    #[test]
    fn test_format_verbose() {
        let uuid = Uuid::new_v4();
        let identity = MockIdentity {
            uuid,
            created_at: SystemTime::now(),
        };

        let display = NodeIdentityDisplay::new(&identity);
        let verbose = display.format_verbose();

        assert!(verbose.contains(&format!("UUID: {uuid}")));
    }

    #[test]
    fn test_format_json() {
        let uuid = Uuid::new_v4();
        let identity = MockIdentity {
            uuid,
            created_at: SystemTime::now(),
        };

        let display = NodeIdentityDisplay::new(&identity);
        let json = display.format_json().expect("Should format JSON");

        assert!(json.contains(&format!("\"uuid\": \"{uuid}\"")));
        assert!(json.contains("\"created_at\""));

        // Verify it's valid JSON
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("Should be valid JSON");
        assert_eq!(parsed["uuid"], uuid.to_string());
    }

    #[test]
    fn test_display_extension_trait() {
        let uuid = Uuid::new_v4();
        let identity = MockIdentity {
            uuid,
            created_at: SystemTime::now(),
        };

        // Test using the extension trait
        let compact = identity.display().format_compact();
        assert_eq!(compact, uuid.to_string()[..8].to_string());

        let verbose = identity.display().format_verbose();
        assert!(verbose.contains(&uuid.to_string()));
    }
}
