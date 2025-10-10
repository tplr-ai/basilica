use sqlx::SqlitePool;
use tracing::info;

use crate::persistence::types::RentalFilter;
use crate::persistence::ValidatorPersistence;
use crate::rental::{RentalInfo, RentalState};

#[derive(Debug, Clone)]
pub struct SimplePersistence {
    pub(crate) pool: SqlitePool,
}

impl SimplePersistence {
    pub fn pool(&self) -> &SqlitePool {
        &self.pool
    }
}

impl SimplePersistence {
    pub fn with_pool(pool: SqlitePool) -> Self {
        Self { pool }
    }

    #[cfg(test)]
    pub async fn for_testing() -> Result<Self, anyhow::Error> {
        let pool = SqlitePool::connect(":memory:").await?;

        sqlx::query("PRAGMA journal_mode = WAL")
            .execute(&pool)
            .await?;
        sqlx::query("PRAGMA busy_timeout = 5000")
            .execute(&pool)
            .await?;
        sqlx::query("PRAGMA synchronous = NORMAL")
            .execute(&pool)
            .await?;

        let instance = Self { pool };
        instance.run_migrations().await?;

        Ok(instance)
    }

    pub async fn new(
        database_path: &str,
        _validator_hotkey: String,
    ) -> Result<Self, anyhow::Error> {
        let db_url = if database_path.starts_with("sqlite:") {
            database_path.to_string()
        } else {
            format!("sqlite:{database_path}")
        };

        let final_url = if db_url.contains("?") {
            db_url
        } else {
            format!("{db_url}?mode=rwc")
        };

        let pool = sqlx::SqlitePool::connect(&final_url).await?;

        sqlx::query("PRAGMA journal_mode = WAL")
            .execute(&pool)
            .await?;
        sqlx::query("PRAGMA busy_timeout = 5000")
            .execute(&pool)
            .await?;
        sqlx::query("PRAGMA synchronous = NORMAL")
            .execute(&pool)
            .await?;

        Ok(Self { pool })
    }

    pub async fn run_migrations(&self) -> Result<(), anyhow::Error> {
        info!("Running database migrations");

        sqlx::migrate!("./migrations")
            .run(&self.pool)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to run migrations: {}", e))?;

        self.initialize_collateral_scan_status().await?;

        info!("Database migrations completed successfully");
        Ok(())
    }

    async fn initialize_collateral_scan_status(&self) -> Result<(), anyhow::Error> {
        use chrono::Utc;
        use collateral_contract::config::CONTRACT_DEPLOYED_BLOCK_NUMBER;
        use tracing::warn;

        let now = Utc::now().to_rfc3339();
        let insert_query = r#"
            INSERT OR IGNORE INTO collateral_scan_status (last_scanned_block_number, updated_at, id)
            VALUES (?, ?, 1)
        "#;

        let result = sqlx::query(insert_query)
            .bind(CONTRACT_DEPLOYED_BLOCK_NUMBER as i64)
            .bind(now)
            .execute(&self.pool)
            .await;

        if let Err(e) = result {
            warn!(
                "Error initializing collateral scan status (may already exist): {}",
                e
            );
        }

        Ok(())
    }
}

#[async_trait::async_trait]
impl ValidatorPersistence for SimplePersistence {
    async fn save_rental(&self, rental: &RentalInfo) -> anyhow::Result<()> {
        sqlx::query(
            "INSERT INTO rentals (
                id, validator_hotkey, node_id, container_id, ssh_session_id,
                ssh_credentials, state, created_at, container_spec, miner_id, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                state = excluded.state,
                container_id = excluded.container_id,
                ssh_session_id = excluded.ssh_session_id,
                ssh_credentials = excluded.ssh_credentials,
                miner_id = excluded.miner_id,
                metadata = excluded.metadata",
        )
        .bind(&rental.rental_id)
        .bind(&rental.validator_hotkey)
        .bind(&rental.node_id)
        .bind(&rental.container_id)
        .bind(&rental.ssh_session_id)
        .bind(&rental.ssh_credentials)
        .bind(match &rental.state {
            RentalState::Provisioning => "provisioning",
            RentalState::Active => "active",
            RentalState::Stopping => "stopping",
            RentalState::Stopped => "stopped",
            RentalState::Failed => "failed",
        })
        .bind(rental.created_at.to_rfc3339())
        .bind(serde_json::to_string(&rental.container_spec)?)
        .bind(&rental.miner_id)
        .bind(serde_json::to_string(&rental.metadata)?)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    async fn load_rental(&self, rental_id: &str) -> anyhow::Result<Option<RentalInfo>> {
        let filter = RentalFilter {
            rental_id: Some(rental_id.to_string()),
            ..Default::default()
        };
        self.query_rentals(filter)
            .await
            .map(|mut rentals| rentals.pop())
    }

    async fn list_validator_rentals(
        &self,
        validator_hotkey: &str,
    ) -> anyhow::Result<Vec<RentalInfo>> {
        let filter = RentalFilter {
            validator_hotkey: Some(validator_hotkey.to_string()),
            order_by_created_desc: true,
            ..Default::default()
        };
        self.query_rentals(filter).await
    }

    async fn query_non_terminated_rentals(&self) -> anyhow::Result<Vec<RentalInfo>> {
        let filter = RentalFilter {
            exclude_states: Some(vec![RentalState::Stopped, RentalState::Failed]),
            order_by_created_desc: true,
            ..Default::default()
        };
        self.query_rentals(filter).await
    }

    async fn delete_rental(&self, rental_id: &str) -> anyhow::Result<()> {
        sqlx::query("DELETE FROM rentals WHERE id = ?")
            .bind(rental_id)
            .execute(&self.pool)
            .await?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::types::{CpuSpec, GpuSpec, NodeRegistration, UpdateMinerRequest};

    #[tokio::test]
    async fn test_prevent_duplicate_ssh_endpoint_registration() {
        let persistence = SimplePersistence::for_testing()
            .await
            .expect("Failed to create persistence");

        let nodes1 = vec![NodeRegistration {
            node_id: "exec1".to_string(),
            ssh_endpoint: "http://192.168.1.1:8080".to_string(),
            gpu_count: 2,
            gpu_specs: vec![GpuSpec {
                name: "RTX 4090".to_string(),
                memory_gb: 24,
                compute_capability: "8.9".to_string(),
            }],
            cpu_specs: CpuSpec {
                cores: 16,
                model: "Intel i9".to_string(),
                memory_gb: 32,
            },
        }];

        let result = persistence
            .register_miner("miner1", "hotkey1", "http://miner1.com", &nodes1)
            .await;
        assert!(result.is_ok());

        let nodes2 = vec![NodeRegistration {
            node_id: "exec2".to_string(),
            ssh_endpoint: "http://192.168.1.1:8080".to_string(),
            gpu_count: 1,
            gpu_specs: vec![GpuSpec {
                name: "RTX 3090".to_string(),
                memory_gb: 24,
                compute_capability: "8.6".to_string(),
            }],
            cpu_specs: CpuSpec {
                cores: 8,
                model: "Intel i7".to_string(),
                memory_gb: 16,
            },
        }];

        let result = persistence
            .register_miner("miner2", "hotkey2", "http://miner2.com", &nodes2)
            .await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("already registered"));
    }

    #[tokio::test]
    async fn test_prevent_duplicate_ssh_endpoint_update() {
        let persistence = SimplePersistence::for_testing()
            .await
            .expect("Failed to create persistence");

        let nodes1 = vec![NodeRegistration {
            node_id: "exec1".to_string(),
            ssh_endpoint: "http://192.168.1.1:8080".to_string(),
            gpu_count: 2,
            gpu_specs: vec![],
            cpu_specs: CpuSpec {
                cores: 16,
                model: "Intel i9".to_string(),
                memory_gb: 32,
            },
        }];

        persistence
            .register_miner("miner1", "hotkey1", "http://miner1.com", &nodes1)
            .await
            .expect("Failed to register miner1");

        let nodes2 = vec![NodeRegistration {
            node_id: "exec2".to_string(),
            ssh_endpoint: "http://192.168.1.2:8080".to_string(),
            gpu_count: 1,
            gpu_specs: vec![],
            cpu_specs: CpuSpec {
                cores: 8,
                model: "Intel i7".to_string(),
                memory_gb: 16,
            },
        }];

        persistence
            .register_miner("miner2", "hotkey2", "http://miner2.com", &nodes2)
            .await
            .expect("Failed to register miner2");

        let update_request = UpdateMinerRequest {
            endpoint: None,
            nodes: Some(vec![NodeRegistration {
                node_id: "exec2_updated".to_string(),
                ssh_endpoint: "http://192.168.1.1:8080".to_string(),
                gpu_count: 1,
                gpu_specs: vec![],
                cpu_specs: CpuSpec {
                    cores: 8,
                    model: "Intel i7".to_string(),
                    memory_gb: 16,
                },
            }]),
            signature: "test_signature".to_string(),
        };

        let result = persistence.update_miner("miner2", &update_request).await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("already registered by another miner"));
    }

    #[tokio::test]
    async fn test_allow_same_miner_update_with_same_ssh_endpoint() {
        let persistence = SimplePersistence::for_testing()
            .await
            .expect("Failed to create persistence");

        let nodes = vec![NodeRegistration {
            node_id: "exec1".to_string(),
            ssh_endpoint: "http://192.168.1.1:8080".to_string(),
            gpu_count: 2,
            gpu_specs: vec![],
            cpu_specs: CpuSpec {
                cores: 16,
                model: "Intel i9".to_string(),
                memory_gb: 32,
            },
        }];

        persistence
            .register_miner("miner1", "hotkey1", "http://miner1.com", &nodes)
            .await
            .expect("Failed to register miner");

        let update_request = UpdateMinerRequest {
            endpoint: Some("http://miner1-updated.com".to_string()),
            nodes: Some(vec![NodeRegistration {
                node_id: "exec1_updated".to_string(),
                ssh_endpoint: "http://192.168.1.1:8080".to_string(),
                gpu_count: 3,
                gpu_specs: vec![],
                cpu_specs: CpuSpec {
                    cores: 16,
                    model: "Intel i9".to_string(),
                    memory_gb: 64,
                },
            }]),
            signature: "test_signature".to_string(),
        };

        let result = persistence.update_miner("miner1", &update_request).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_gpu_uuid_duplicate_prevention() {
        let persistence = SimplePersistence::for_testing().await.unwrap();

        let node1 = NodeRegistration {
            node_id: "exec1".to_string(),
            ssh_endpoint: "root@192.168.1.100:50051".to_string(),
            gpu_count: 1,
            gpu_specs: vec![],
            cpu_specs: CpuSpec {
                cores: 8,
                model: "Intel i7".to_string(),
                memory_gb: 32,
            },
        };

        persistence
            .register_miner("miner1", "hotkey1", "http://miner1.com", &[node1])
            .await
            .unwrap();

        let gpu_uuid = "GPU-550e8400-e29b-41d4-a716-446655440000";
        sqlx::query("UPDATE miner_nodes SET gpu_uuids = ? WHERE miner_id = ? AND node_id = ?")
            .bind(gpu_uuid)
            .bind("miner1")
            .bind("exec1")
            .execute(&persistence.pool)
            .await
            .unwrap();

        let node2 = NodeRegistration {
            node_id: "exec2".to_string(),
            ssh_endpoint: "root@192.168.1.101:50051".to_string(),
            gpu_count: 1,
            gpu_specs: vec![],
            cpu_specs: CpuSpec {
                cores: 8,
                model: "Intel i7".to_string(),
                memory_gb: 32,
            },
        };

        persistence
            .register_miner("miner2", "hotkey2", "http://miner2.com", &[node2])
            .await
            .unwrap();

        let count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM miner_nodes")
            .fetch_one(&persistence.pool)
            .await
            .unwrap();
        assert_eq!(count, 2);

        let gpu_count: i64 =
            sqlx::query_scalar("SELECT COUNT(*) FROM miner_nodes WHERE gpu_uuids = ?")
                .bind(gpu_uuid)
                .fetch_one(&persistence.pool)
                .await
                .unwrap();
        assert_eq!(gpu_count, 1);
    }

    #[tokio::test]
    async fn test_hardware_profile_enrichment() {
        let persistence = SimplePersistence::for_testing()
            .await
            .expect("Failed to create persistence");

        let node = NodeRegistration {
            node_id: "exec1".to_string(),
            ssh_endpoint: "root@192.168.1.100:50051".to_string(),
            gpu_count: 2,
            gpu_specs: vec![],
            cpu_specs: CpuSpec {
                cores: 8,
                model: "Intel i7".to_string(),
                memory_gb: 32,
            },
        };

        persistence
            .register_miner("miner_1", "hotkey1", "http://miner1.com", &[node])
            .await
            .unwrap();

        persistence
            .store_node_hardware_profile(
                1,
                "exec1",
                Some("AMD EPYC 7763".to_string()),
                Some(64),
                Some(256),
                Some(1000),
                r#"{"cpu": "AMD EPYC 7763", "cores": 64, "ram": 256}"#,
            )
            .await
            .unwrap();

        persistence
            .store_node_network_profile(
                1,
                "exec1",
                Some("192.168.1.100".to_string()),
                Some("exec1.example.com".to_string()),
                Some("San Francisco".to_string()),
                Some("California".to_string()),
                Some("US".to_string()),
                Some("37.7749,-122.4194".to_string()),
                Some("AS12345 Example ISP".to_string()),
                Some("94102".to_string()),
                Some("America/Los_Angeles".to_string()),
                &chrono::Utc::now().to_rfc3339(),
                r#"{"city": "San Francisco", "region": "California", "country": "US"}"#,
            )
            .await
            .unwrap();

        let nodes = persistence.get_miner_nodes("miner_1").await.unwrap();
        assert_eq!(nodes.len(), 1);

        let node = &nodes[0];
        assert_eq!(node.node_id, "exec1");
        assert_eq!(node.cpu_specs.model, "AMD EPYC 7763");
        assert_eq!(node.cpu_specs.cores, 64);
        assert_eq!(node.cpu_specs.memory_gb, 256);
        assert_eq!(
            node.location,
            Some("San Francisco/California/US".to_string())
        );

        let available = persistence
            .get_available_nodes(None, None, None, None)
            .await
            .unwrap();

        assert_eq!(available.len(), 1);
        let available_exec = &available[0];
        assert_eq!(available_exec.cpu_specs.model, "AMD EPYC 7763");
        assert_eq!(available_exec.cpu_specs.cores, 64);
        assert_eq!(available_exec.cpu_specs.memory_gb, 256);
        assert_eq!(
            available_exec.location,
            Some("San Francisco/California/US".to_string())
        );
    }
}
