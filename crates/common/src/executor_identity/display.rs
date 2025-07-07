//! Display formatting for executor identities
//!
//! This module implements the IdentityDisplay trait to provide
//! consistent formatting for executor identities across CLI commands.

use anyhow::Result;
use serde_json::json;

use crate::executor_identity::{ExecutorIdentity, IdentityDisplay};

/// Implementation of IdentityDisplay for any type implementing ExecutorIdentity
pub struct ExecutorIdentityDisplay<'a> {
    identity: &'a dyn ExecutorIdentity,
}

impl<'a> ExecutorIdentityDisplay<'a> {
    /// Creates a new display wrapper for an executor identity
    pub fn new(identity: &'a dyn ExecutorIdentity) -> Self {
        Self { identity }
    }
}

impl<'a> IdentityDisplay for ExecutorIdentityDisplay<'a> {
    /// Returns HUID only for human readability
    ///
    /// This is the default display format for CLI commands
    fn format_compact(&self) -> String {
        self.identity.huid().to_string()
    }

    /// Returns full UUID + HUID display
    ///
    /// Format: "HUID: swift-falcon-a3f2\nUUID: 550e8400-e29b-41d4-a716-446655440000"
    fn format_verbose(&self) -> String {
        format!(
            "HUID: {}\nUUID: {}",
            self.identity.huid(),
            self.identity.uuid()
        )
    }

    /// Returns JSON with both identifiers
    ///
    /// Format: {"uuid": "...", "huid": "...", "created_at": "..."}
    fn format_json(&self) -> Result<String> {
        let json_obj = json!({
            "uuid": self.identity.uuid().to_string(),
            "huid": self.identity.huid(),
            "created_at": self.identity.created_at()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
        });

        serde_json::to_string_pretty(&json_obj)
            .map_err(|e| anyhow::anyhow!("Failed to serialize to JSON: {}", e))
    }
}

/// Extension trait to add display methods directly to ExecutorIdentity types
pub trait ExecutorIdentityDisplayExt {
    /// Creates a display formatter for this identity
    fn display(&self) -> ExecutorIdentityDisplay;
}

impl<T: ExecutorIdentity> ExecutorIdentityDisplayExt for T {
    fn display(&self) -> ExecutorIdentityDisplay {
        ExecutorIdentityDisplay::new(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::SystemTime;
    use uuid::Uuid;

    struct MockIdentity {
        uuid: Uuid,
        huid: String,
        created_at: SystemTime,
    }

    impl ExecutorIdentity for MockIdentity {
        fn uuid(&self) -> &Uuid {
            &self.uuid
        }

        fn huid(&self) -> &str {
            &self.huid
        }

        fn created_at(&self) -> SystemTime {
            self.created_at
        }

        fn matches(&self, query: &str) -> bool {
            self.huid.starts_with(query) || self.uuid.to_string().starts_with(query)
        }

        fn full_display(&self) -> String {
            format!("{} ({})", self.huid, self.uuid)
        }

        fn short_uuid(&self) -> String {
            self.uuid.to_string()[..8].to_string()
        }
    }

    #[test]
    fn test_format_compact() {
        let identity = MockIdentity {
            uuid: Uuid::new_v4(),
            huid: "swift-falcon-a3f2".to_string(),
            created_at: SystemTime::now(),
        };

        let display = ExecutorIdentityDisplay::new(&identity);
        assert_eq!(display.format_compact(), "swift-falcon-a3f2");
    }

    #[test]
    fn test_format_verbose() {
        let uuid = Uuid::new_v4();
        let identity = MockIdentity {
            uuid,
            huid: "swift-falcon-a3f2".to_string(),
            created_at: SystemTime::now(),
        };

        let display = ExecutorIdentityDisplay::new(&identity);
        let verbose = display.format_verbose();

        assert!(verbose.contains("HUID: swift-falcon-a3f2"));
        assert!(verbose.contains(&format!("UUID: {uuid}")));
        assert!(verbose.contains('\n'));
    }

    #[test]
    fn test_format_json() {
        let uuid = Uuid::new_v4();
        let identity = MockIdentity {
            uuid,
            huid: "swift-falcon-a3f2".to_string(),
            created_at: SystemTime::now(),
        };

        let display = ExecutorIdentityDisplay::new(&identity);
        let json = display.format_json().expect("Should format JSON");

        assert!(json.contains(&format!("\"uuid\": \"{uuid}\"")));
        assert!(json.contains("\"huid\": \"swift-falcon-a3f2\""));
        assert!(json.contains("\"created_at\""));

        // Verify it's valid JSON
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("Should be valid JSON");
        assert_eq!(parsed["huid"], "swift-falcon-a3f2");
        assert_eq!(parsed["uuid"], uuid.to_string());
    }

    #[test]
    fn test_display_extension_trait() {
        let identity = MockIdentity {
            uuid: Uuid::new_v4(),
            huid: "brave-tiger-5678".to_string(),
            created_at: SystemTime::now(),
        };

        // Test using the extension trait
        let compact = identity.display().format_compact();
        assert_eq!(compact, "brave-tiger-5678");

        let verbose = identity.display().format_verbose();
        assert!(verbose.contains("brave-tiger-5678"));
    }
}
