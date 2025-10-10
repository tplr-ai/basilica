use anyhow::Result;
use async_trait::async_trait;
use std::time::SystemTime;
use uuid::Uuid;

pub trait NodeIdentity: Send + Sync {
    fn uuid(&self) -> &Uuid;
    fn created_at(&self) -> SystemTime;
    fn matches(&self, query: &str) -> bool;
    fn full_display(&self) -> String;
    fn short_uuid(&self) -> String;
}

#[async_trait]
pub trait IdentityPersistence: Send + Sync {
    async fn get_or_create(&self) -> Result<Box<dyn NodeIdentity>>;
    async fn find_by_identifier(&self, id: &str) -> Result<Option<Box<dyn NodeIdentity>>>;
    async fn save(&self, id: &dyn NodeIdentity) -> Result<()>;
}

pub trait IdentityDisplay: Send + Sync {
    fn format_compact(&self) -> String;
    fn format_verbose(&self) -> String;
    fn format_json(&self) -> Result<String>;
}

pub trait WordProvider: Send + Sync {
    fn get_adjective(&self, index: usize) -> Option<&str>;
    fn get_noun(&self, index: usize) -> Option<&str>;
    fn adjective_count(&self) -> usize;
    fn noun_count(&self) -> usize;
    fn validate_word_lists(&self) -> Result<()>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    // Mock implementation of NodeIdentity for testing
    struct MockNodeIdentity {
        uuid: Uuid,
        created_at: SystemTime,
    }

    impl NodeIdentity for MockNodeIdentity {
        fn uuid(&self) -> &Uuid {
            &self.uuid
        }

        fn created_at(&self) -> SystemTime {
            self.created_at
        }

        fn matches(&self, query: &str) -> bool {
            if query.len() < 3 {
                return false;
            }

            self.uuid.to_string().starts_with(query)
        }

        fn full_display(&self) -> String {
            self.uuid.to_string()
        }

        fn short_uuid(&self) -> String {
            self.uuid.to_string()[..8].to_string()
        }
    }

    // Mock implementation of IdentityPersistence for testing
    struct MockIdentityPersistence {
        stored: std::sync::Mutex<Option<Box<dyn NodeIdentity>>>,
    }

    impl MockIdentityPersistence {
        fn new() -> Self {
            Self {
                stored: std::sync::Mutex::new(None),
            }
        }
    }

    #[async_trait]
    impl IdentityPersistence for MockIdentityPersistence {
        async fn get_or_create(&self) -> Result<Box<dyn NodeIdentity>> {
            let mut storage = self.stored.lock().unwrap();
            if storage.is_none() {
                let mock = MockNodeIdentity {
                    uuid: Uuid::new_v4(),
                    created_at: SystemTime::now(),
                };
                *storage = Some(Box::new(mock) as Box<dyn NodeIdentity>);
            }

            // Clone the stored identity
            let stored = storage.as_ref().unwrap();
            let cloned = MockNodeIdentity {
                uuid: *stored.uuid(),
                created_at: stored.created_at(),
            };
            Ok(Box::new(cloned))
        }

        async fn find_by_identifier(&self, id: &str) -> Result<Option<Box<dyn NodeIdentity>>> {
            let storage = self.stored.lock().unwrap();
            if let Some(stored) = storage.as_ref() {
                if stored.matches(id) {
                    let cloned = MockNodeIdentity {
                        uuid: *stored.uuid(),
                        created_at: stored.created_at(),
                    };
                    return Ok(Some(Box::new(cloned)));
                }
            }
            Ok(None)
        }

        async fn save(&self, id: &dyn NodeIdentity) -> Result<()> {
            let mut storage = self.stored.lock().unwrap();
            let saved = MockNodeIdentity {
                uuid: *id.uuid(),
                created_at: id.created_at(),
            };
            *storage = Some(Box::new(saved));
            Ok(())
        }
    }

    // Mock implementation of IdentityDisplay for testing
    struct MockIdentityDisplay {
        uuid: Uuid,
    }

    impl IdentityDisplay for MockIdentityDisplay {
        fn format_compact(&self) -> String {
            self.uuid.to_string()[..8].to_string()
        }

        fn format_verbose(&self) -> String {
            format!("UUID: {}", self.uuid)
        }

        fn format_json(&self) -> Result<String> {
            Ok(format!(r#"{{"uuid":"{}"}}"#, self.uuid))
        }
    }

    // Mock implementation of WordProvider for testing
    struct MockWordProvider {
        adjectives: Vec<String>,
        nouns: Vec<String>,
    }

    impl MockWordProvider {
        fn new() -> Self {
            Self {
                adjectives: vec![
                    "swift".to_string(),
                    "brave".to_string(),
                    "clever".to_string(),
                ],
                nouns: vec![
                    "falcon".to_string(),
                    "tiger".to_string(),
                    "eagle".to_string(),
                ],
            }
        }
    }

    impl WordProvider for MockWordProvider {
        fn get_adjective(&self, index: usize) -> Option<&str> {
            self.adjectives.get(index).map(|s| s.as_str())
        }

        fn get_noun(&self, index: usize) -> Option<&str> {
            self.nouns.get(index).map(|s| s.as_str())
        }

        fn adjective_count(&self) -> usize {
            self.adjectives.len()
        }

        fn noun_count(&self) -> usize {
            self.nouns.len()
        }

        fn validate_word_lists(&self) -> Result<()> {
            if self.adjective_count() < 2 {
                anyhow::bail!("Not enough adjectives");
            }
            if self.noun_count() < 2 {
                anyhow::bail!("Not enough nouns");
            }
            Ok(())
        }
    }

    #[test]
    fn test_node_identity_trait() {
        let created = SystemTime::now();
        let id = MockNodeIdentity {
            uuid: Uuid::new_v4(),
            created_at: created,
        };

        // Test basic methods
        assert_eq!(id.created_at(), created);
        assert_eq!(id.full_display(), id.uuid().to_string());
        assert_eq!(id.short_uuid().len(), 8);

        // Test UUID prefix matching
        let uuid_str = id.uuid().to_string();
        assert!(id.matches(&uuid_str[..8]));
        assert!(!id.matches("sw")); // Too short
    }

    #[tokio::test]
    async fn test_identity_persistence_trait() {
        let persistence = MockIdentityPersistence::new();

        // Test get_or_create
        let id1 = persistence.get_or_create().await.unwrap();
        let id2 = persistence.get_or_create().await.unwrap();
        assert_eq!(id1.uuid(), id2.uuid());

        // Test find_by_identifier
        let uuid_str = id1.uuid().to_string();
        let found = persistence
            .find_by_identifier(&uuid_str[..8])
            .await
            .unwrap();
        assert!(found.is_some());
        assert_eq!(found.unwrap().uuid(), id1.uuid());

        // Test not found
        let not_found = persistence.find_by_identifier("nonexistent").await.unwrap();
        assert!(not_found.is_none());

        // Test save
        let new_uuid = Uuid::new_v4();
        let new_id = MockNodeIdentity {
            uuid: new_uuid,
            created_at: SystemTime::now(),
        };
        persistence.save(&new_id).await.unwrap();

        // Verify saved
        let found = persistence
            .find_by_identifier(&new_uuid.to_string()[..8])
            .await
            .unwrap();
        assert!(found.is_some());
        assert_eq!(found.unwrap().uuid(), &new_uuid);
    }

    #[test]
    fn test_identity_display_trait() {
        let uuid = Uuid::new_v4();
        let display = MockIdentityDisplay { uuid };

        // Test format_compact
        assert_eq!(display.format_compact(), uuid.to_string()[..8].to_string());

        // Test format_verbose
        let verbose = display.format_verbose();
        assert!(verbose.contains(&uuid.to_string()));

        // Test format_json
        let json = display.format_json().unwrap();
        assert!(json.contains(&format!(r#""uuid":"{}""#, uuid)));
    }

    #[test]
    fn test_word_provider_trait() {
        let provider = MockWordProvider::new();

        // Test get methods
        assert_eq!(provider.get_adjective(0), Some("swift"));
        assert_eq!(provider.get_adjective(1), Some("brave"));
        assert_eq!(provider.get_adjective(2), Some("clever"));
        assert_eq!(provider.get_adjective(3), None);

        assert_eq!(provider.get_noun(0), Some("falcon"));
        assert_eq!(provider.get_noun(1), Some("tiger"));
        assert_eq!(provider.get_noun(2), Some("eagle"));
        assert_eq!(provider.get_noun(3), None);

        // Test counts
        assert_eq!(provider.adjective_count(), 3);
        assert_eq!(provider.noun_count(), 3);

        // Test validation
        assert!(provider.validate_word_lists().is_ok());

        // Test validation failure
        let empty_provider = MockWordProvider {
            adjectives: vec![],
            nouns: vec!["test".to_string()],
        };
        assert!(empty_provider.validate_word_lists().is_err());
    }

    #[test]
    fn test_node_identity_edge_cases() {
        let id = MockNodeIdentity {
            uuid: Uuid::nil(),
            created_at: UNIX_EPOCH,
        };

        // Test with nil UUID
        assert_eq!(id.uuid(), &Uuid::nil());
        assert_eq!(id.short_uuid(), "00000000");

        // Test with epoch time
        assert_eq!(id.created_at(), UNIX_EPOCH);
    }

    #[test]
    fn test_word_provider_boundary_conditions() {
        let provider = MockWordProvider::new();

        // Test boundary indices
        assert!(provider.get_adjective(usize::MAX).is_none());
        assert!(provider.get_noun(usize::MAX).is_none());

        // Test empty provider
        let empty = MockWordProvider {
            adjectives: vec![],
            nouns: vec![],
        };
        assert_eq!(empty.adjective_count(), 0);
        assert_eq!(empty.noun_count(), 0);
        assert!(empty.validate_word_lists().is_err());
    }

    #[test]
    fn test_identity_display_json_format() {
        let uuid = Uuid::new_v4();
        let display = MockIdentityDisplay { uuid };

        let json = display.format_json().unwrap();

        // Verify it's valid JSON format
        assert!(json.starts_with('{'));
        assert!(json.ends_with('}'));
        assert!(json.contains(r#""uuid""#));

        // Verify the values
        assert!(json.contains(&uuid.to_string()));
    }

    #[tokio::test]
    async fn test_persistence_concurrent_access() {
        let persistence = std::sync::Arc::new(MockIdentityPersistence::new());

        // Create multiple tasks accessing persistence
        let tasks: Vec<_> = (0..5)
            .map(|_| {
                let p = persistence.clone();
                tokio::spawn(async move { p.get_or_create().await.unwrap() })
            })
            .collect();

        // Wait for all tasks
        // Wait for all tasks to complete
        let mut results = Vec::new();
        for task in tasks {
            results.push(task.await.unwrap());
        }

        // All should return the same identity
        let first_uuid = results[0].uuid();
        for result in &results {
            assert_eq!(result.uuid(), first_uuid);
        }
    }
}
