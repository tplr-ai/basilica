use crate::error::Result;
use crate::models::{Deployment, DeploymentStatus, GpuOffering, Provider, ProviderSshKey, SshKey};
use chrono::{DateTime, Utc};
use sqlx::postgres::{PgPool, PgPoolOptions};
use sqlx::Row;
use std::str::FromStr;

pub struct Database {
    pool: PgPool,
}

impl Database {
    /// Create new database connection
    pub async fn new(database_url: &str) -> Result<Self> {
        let pool = PgPoolOptions::new()
            .max_connections(5)
            .connect(database_url)
            .await?;

        Ok(Self { pool })
    }

    /// Create database from an existing pool (for testing/mocking)
    pub fn from_pool(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Insert or update GPU offerings
    pub async fn upsert_offerings(&self, offerings: &[GpuOffering]) -> Result<()> {
        let mut tx = self.pool.begin().await?;

        for offering in offerings {
            sqlx::query(
                r#"
                INSERT INTO gpu_offerings
                (id, provider, gpu_type, gpu_memory_gb_per_gpu, gpu_count, interconnect, storage, deployment_type,
                 system_memory_gb, vcpu_count, region, hourly_rate_per_gpu, availability, raw_metadata, fetched_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
                ON CONFLICT(id) DO UPDATE SET
                    interconnect = EXCLUDED.interconnect,
                    storage = EXCLUDED.storage,
                    deployment_type = EXCLUDED.deployment_type,
                    hourly_rate_per_gpu = EXCLUDED.hourly_rate_per_gpu,
                    availability = EXCLUDED.availability,
                    raw_metadata = EXCLUDED.raw_metadata,
                    fetched_at = EXCLUDED.fetched_at
                "#,
            )
            .bind(&offering.id)
            .bind(offering.provider.as_str())
            .bind(offering.gpu_type.as_str())
            .bind(offering.gpu_memory_gb_per_gpu.map(|m| m as i32))
            .bind(offering.gpu_count as i32)
            .bind(offering.interconnect.as_ref())
            .bind(offering.storage.as_ref())
            .bind(offering.deployment_type.as_ref())
            .bind(offering.system_memory_gb as i32)
            .bind(offering.vcpu_count as i32)
            .bind(&offering.region)
            .bind(offering.hourly_rate_per_gpu)  // Bind as Decimal, not string
            .bind(offering.availability)
            .bind(&offering.raw_metadata)
            .bind(offering.fetched_at)
            .execute(&mut *tx)
            .await?;
        }

        tx.commit().await?;
        Ok(())
    }

    /// Get all offerings for a provider
    pub async fn get_offerings(&self, provider: Option<Provider>) -> Result<Vec<GpuOffering>> {
        let query = if let Some(p) = provider {
            sqlx::query(
                "SELECT id, provider, gpu_type, gpu_memory_gb_per_gpu, gpu_count, interconnect, storage, deployment_type,
                        system_memory_gb, vcpu_count, region, hourly_rate_per_gpu, availability, raw_metadata, fetched_at
                 FROM gpu_offerings WHERE provider = $1 ORDER BY fetched_at DESC",
            )
            .bind(p.as_str())
        } else {
            sqlx::query(
                "SELECT id, provider, gpu_type, gpu_memory_gb_per_gpu, gpu_count, interconnect, storage, deployment_type,
                        system_memory_gb, vcpu_count, region, hourly_rate_per_gpu, availability, raw_metadata, fetched_at
                 FROM gpu_offerings ORDER BY fetched_at DESC",
            )
        };

        let rows = query.fetch_all(&self.pool).await?;

        let offerings = rows
            .into_iter()
            .filter_map(|row| {
                let id: String = row.get("id");
                let provider_str: String = row.get("provider");
                let gpu_type_str: String = row.get("gpu_type");
                let raw_metadata: serde_json::Value = row.get("raw_metadata");

                let provider = match provider_str.parse() {
                    Ok(p) => p,
                    Err(e) => {
                        tracing::error!(
                            row_id = %id,
                            provider = %provider_str,
                            error = %e,
                            "Failed to parse provider from gpu_offerings row, skipping"
                        );
                        return None;
                    }
                };

                let gpu_type = match gpu_type_str.parse() {
                    Ok(g) => g,
                    Err(e) => {
                        tracing::error!(
                            row_id = %id,
                            gpu_type = %gpu_type_str,
                            error = %e,
                            "Failed to parse gpu_type from gpu_offerings row, skipping"
                        );
                        return None;
                    }
                };

                Some(GpuOffering {
                    id,
                    provider,
                    gpu_type,
                    gpu_memory_gb_per_gpu: row
                        .get::<Option<i32>, _>("gpu_memory_gb_per_gpu")
                        .map(|m| m as u32),
                    gpu_count: row.get::<i32, _>("gpu_count") as u32,
                    interconnect: row.get("interconnect"),
                    storage: row.get("storage"),
                    deployment_type: row.get("deployment_type"),
                    system_memory_gb: row.get::<i32, _>("system_memory_gb") as u32,
                    vcpu_count: row.get::<i32, _>("vcpu_count") as u32,
                    region: row.get("region"),
                    hourly_rate_per_gpu: row.get("hourly_rate_per_gpu"), // Get as Decimal directly
                    availability: row.get("availability"),
                    raw_metadata,
                    fetched_at: row.get("fetched_at"),
                })
            })
            .collect();

        Ok(offerings)
    }

    /// Get last fetch time for provider from offerings table
    pub async fn get_last_fetch_time(&self, provider: Provider) -> Result<Option<DateTime<Utc>>> {
        let row = sqlx::query(
            "SELECT MAX(fetched_at) as last_fetch FROM gpu_offerings WHERE provider = $1",
        )
        .bind(provider.as_str())
        .fetch_optional(&self.pool)
        .await?;

        Ok(row.and_then(|r| r.get("last_fetch")))
    }

    // ========================================================================
    // Deployment Management
    // ========================================================================

    /// Create a new deployment record
    pub async fn create_deployment(&self, deployment: &Deployment) -> Result<()> {
        sqlx::query(
            r#"
            INSERT INTO secure_cloud_rentals
            (id, user_id, provider, provider_instance_id, offering_id, instance_type, location_code,
             status, hostname, ssh_public_key, ip_address, connection_info, raw_response,
             error_message, created_at, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
            "#,
        )
        .bind(&deployment.id)
        .bind(&deployment.user_id)
        .bind(deployment.provider.as_str())
        .bind(&deployment.provider_instance_id)
        .bind(&deployment.offering_id)
        .bind(&deployment.instance_type)
        .bind(&deployment.location_code)
        .bind(deployment.status.as_str())
        .bind(&deployment.hostname)
        .bind(&deployment.ssh_public_key)
        .bind(&deployment.ip_address)
        .bind(&deployment.connection_info)
        .bind(&deployment.raw_response)
        .bind(&deployment.error_message)
        .bind(deployment.created_at)
        .bind(deployment.updated_at)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// Update deployment status and details
    #[allow(clippy::too_many_arguments)]
    pub async fn update_deployment(
        &self,
        id: &str,
        provider_instance_id: Option<String>,
        status: DeploymentStatus,
        ip_address: Option<String>,
        connection_info: Option<serde_json::Value>,
        raw_response: Option<serde_json::Value>,
        error_message: Option<String>,
    ) -> Result<()> {
        let now = Utc::now();

        sqlx::query(
            r#"
            UPDATE secure_cloud_rentals
            SET provider_instance_id = $1,
                status = $2,
                ip_address = $3,
                connection_info = $4,
                raw_response = $5,
                error_message = $6,
                updated_at = $7
            WHERE id = $8
            "#,
        )
        .bind(provider_instance_id)
        .bind(status.as_str())
        .bind(ip_address)
        .bind(connection_info)
        .bind(raw_response)
        .bind(error_message)
        .bind(now)
        .bind(id)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// Helper function to parse a deployment from a database row with error logging
    fn parse_deployment_from_row(row: &sqlx::postgres::PgRow) -> Option<Deployment> {
        let id: String = row.get("id");
        let provider_str: String = row.get("provider");
        let status_str: String = row.get("status");

        let provider = match provider_str.parse() {
            Ok(p) => p,
            Err(e) => {
                tracing::error!(
                    deployment_id = %id,
                    provider = %provider_str,
                    error = %e,
                    "Failed to parse provider from deployment row, skipping"
                );
                return None;
            }
        };

        let status = match status_str.parse() {
            Ok(s) => s,
            Err(e) => {
                tracing::error!(
                    deployment_id = %id,
                    status = %status_str,
                    error = %e,
                    "Failed to parse status from deployment row, skipping"
                );
                return None;
            }
        };

        Some(Deployment {
            id,
            user_id: row.get("user_id"),
            provider,
            provider_instance_id: row.get("provider_instance_id"),
            offering_id: row.get("offering_id"),
            instance_type: row.get("instance_type"),
            location_code: row.get("location_code"),
            status,
            hostname: row.get("hostname"),
            ssh_public_key: row.get("ssh_public_key"),
            ip_address: row.get("ip_address"),
            connection_info: row.get("connection_info"),
            raw_response: row.get("raw_response"),
            error_message: row.get("error_message"),
            created_at: row.get("created_at"),
            updated_at: row.get("updated_at"),
        })
    }

    /// Get deployment by ID
    pub async fn get_deployment(&self, id: &str) -> Result<Option<Deployment>> {
        let row = sqlx::query(
            r#"
            SELECT id, user_id, provider, provider_instance_id, offering_id, instance_type, location_code,
                   status, hostname, ssh_public_key, ip_address, connection_info, raw_response,
                   error_message, created_at, updated_at
            FROM secure_cloud_rentals
            WHERE id = $1
            "#,
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await?;

        Ok(row.and_then(|r| Self::parse_deployment_from_row(&r)))
    }

    /// Get deployment by ID and user ID (ownership check)
    pub async fn get_deployment_by_user(
        &self,
        id: &str,
        user_id: &str,
    ) -> Result<Option<Deployment>> {
        let row = sqlx::query(
            r#"
            SELECT id, user_id, provider, provider_instance_id, offering_id, instance_type, location_code,
                   status, hostname, ssh_public_key, ip_address, connection_info, raw_response,
                   error_message, created_at, updated_at
            FROM secure_cloud_rentals
            WHERE id = $1 AND user_id = $2
            "#,
        )
        .bind(id)
        .bind(user_id)
        .fetch_optional(&self.pool)
        .await?;

        Ok(row.and_then(|r| Self::parse_deployment_from_row(&r)))
    }

    /// List deployments by user
    pub async fn list_deployments_by_user(&self, user_id: &str) -> Result<Vec<Deployment>> {
        let rows = sqlx::query(
            r#"
            SELECT id, user_id, provider, provider_instance_id, offering_id, instance_type, location_code,
                   status, hostname, ssh_public_key, ip_address, connection_info, raw_response,
                   error_message, created_at, updated_at
            FROM secure_cloud_rentals
            WHERE user_id = $1
            ORDER BY created_at DESC
            "#,
        )
        .bind(user_id)
        .fetch_all(&self.pool)
        .await?;

        let deployments = rows
            .iter()
            .filter_map(Self::parse_deployment_from_row)
            .collect();

        Ok(deployments)
    }

    /// List all deployments with optional filters
    pub async fn list_deployments(
        &self,
        provider: Option<Provider>,
        status: Option<DeploymentStatus>,
    ) -> Result<Vec<Deployment>> {
        let mut query_str = String::from(
            r#"
            SELECT id, user_id, provider, provider_instance_id, offering_id, instance_type, location_code,
                   status, hostname, ssh_public_key, ip_address, connection_info, raw_response,
                   error_message, created_at, updated_at
            FROM secure_cloud_rentals
            WHERE 1=1
            "#,
        );

        let mut conditions = Vec::new();
        let mut bind_index = 1;

        if provider.is_some() {
            conditions.push(format!("provider = ${}", bind_index));
            bind_index += 1;
        }

        if status.is_some() {
            conditions.push(format!("status = ${}", bind_index));
        }

        if !conditions.is_empty() {
            query_str.push_str(" AND ");
            query_str.push_str(&conditions.join(" AND "));
        }

        query_str.push_str(" ORDER BY created_at DESC");

        let mut query = sqlx::query(&query_str);

        if let Some(p) = provider {
            query = query.bind(p.as_str());
        }

        if let Some(s) = status {
            query = query.bind(s.as_str());
        }

        let rows = query.fetch_all(&self.pool).await?;

        let deployments = rows
            .iter()
            .filter_map(Self::parse_deployment_from_row)
            .collect();

        Ok(deployments)
    }

    /// Get all active deployments (for telemetry monitoring)
    pub async fn get_all_active_deployments(&self) -> Result<Vec<Deployment>> {
        let active_statuses: Vec<&str> = vec![
            DeploymentStatus::Pending.as_str(),
            DeploymentStatus::Provisioning.as_str(),
            DeploymentStatus::Running.as_str(),
        ];

        let rows = sqlx::query(
            r#"
            SELECT id, user_id, provider, provider_instance_id, offering_id, instance_type, location_code,
                   status, hostname, ssh_public_key, ip_address, connection_info, raw_response,
                   error_message, created_at, updated_at
            FROM secure_cloud_rentals
            WHERE status = ANY($1)
            ORDER BY created_at DESC
            "#,
        )
        .bind(&active_statuses)
        .fetch_all(&self.pool)
        .await?;

        let deployments = rows
            .iter()
            .filter_map(Self::parse_deployment_from_row)
            .collect();

        Ok(deployments)
    }

    /// Delete deployment record
    pub async fn delete_deployment(&self, id: &str) -> Result<()> {
        sqlx::query("DELETE FROM secure_cloud_rentals WHERE id = $1")
            .bind(id)
            .execute(&self.pool)
            .await?;

        Ok(())
    }

    // ========================================================================
    // SSH Key Management
    // ========================================================================

    /// Create a new SSH key
    pub async fn create_ssh_key(&self, ssh_key: &SshKey) -> Result<()> {
        sqlx::query(
            r#"
            INSERT INTO ssh_keys (id, user_id, name, public_key, created_at, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6)
            "#,
        )
        .bind(&ssh_key.id)
        .bind(&ssh_key.user_id)
        .bind(&ssh_key.name)
        .bind(&ssh_key.public_key)
        .bind(ssh_key.created_at)
        .bind(ssh_key.updated_at)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// Get SSH key by user ID
    pub async fn get_ssh_key_by_user(&self, user_id: &str) -> Result<Option<SshKey>> {
        let row = sqlx::query(
            "SELECT id, user_id, name, public_key, created_at, updated_at
             FROM ssh_keys WHERE user_id = $1",
        )
        .bind(user_id)
        .fetch_optional(&self.pool)
        .await?;

        Ok(row.map(|r| SshKey {
            id: r.get("id"),
            user_id: r.get("user_id"),
            name: r.get("name"),
            public_key: r.get("public_key"),
            created_at: r.get("created_at"),
            updated_at: r.get("updated_at"),
        }))
    }

    /// Get SSH key by ID
    pub async fn get_ssh_key_by_id(&self, id: &str) -> Result<Option<SshKey>> {
        let row = sqlx::query(
            "SELECT id, user_id, name, public_key, created_at, updated_at
             FROM ssh_keys WHERE id = $1",
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await?;

        Ok(row.map(|r| SshKey {
            id: r.get("id"),
            user_id: r.get("user_id"),
            name: r.get("name"),
            public_key: r.get("public_key"),
            created_at: r.get("created_at"),
            updated_at: r.get("updated_at"),
        }))
    }

    /// Delete SSH key (cascades to provider_ssh_keys)
    pub async fn delete_ssh_key(&self, id: &str) -> Result<()> {
        sqlx::query("DELETE FROM ssh_keys WHERE id = $1")
            .bind(id)
            .execute(&self.pool)
            .await?;

        Ok(())
    }

    // ========================================================================
    // Provider SSH Key Mappings
    // ========================================================================

    /// Get provider SSH key mapping
    pub async fn get_provider_ssh_key(
        &self,
        ssh_key_id: &str,
        provider: Provider,
    ) -> Result<Option<ProviderSshKey>> {
        let row = sqlx::query(
            "SELECT id, ssh_key_id, provider, provider_key_id, created_at, metadata
             FROM provider_ssh_keys WHERE ssh_key_id = $1 AND provider = $2",
        )
        .bind(ssh_key_id)
        .bind(provider.as_str())
        .fetch_optional(&self.pool)
        .await?;

        Ok(row.and_then(|r| {
            let id: String = r.get("id");
            let provider_str: String = r.get("provider");

            let provider = match Provider::from_str(&provider_str) {
                Ok(p) => p,
                Err(e) => {
                    tracing::error!(
                        provider_ssh_key_id = %id,
                        provider = %provider_str,
                        error = %e,
                        "Failed to parse provider from provider_ssh_keys row, skipping"
                    );
                    return None;
                }
            };

            Some(ProviderSshKey {
                id,
                ssh_key_id: r.get("ssh_key_id"),
                provider,
                provider_key_id: r.get("provider_key_id"),
                created_at: r.get("created_at"),
                metadata: r.get("metadata"),
            })
        }))
    }

    /// Create provider SSH key mapping
    pub async fn create_provider_ssh_key(&self, mapping: &ProviderSshKey) -> Result<()> {
        sqlx::query(
            r#"
            INSERT INTO provider_ssh_keys (id, ssh_key_id, provider, provider_key_id, created_at, metadata)
            VALUES ($1, $2, $3, $4, $5, $6)
            "#,
        )
        .bind(&mapping.id)
        .bind(&mapping.ssh_key_id)
        .bind(mapping.provider.as_str())
        .bind(&mapping.provider_key_id)
        .bind(mapping.created_at)
        .bind(&mapping.metadata)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// List all provider SSH key mappings for a given SSH key
    pub async fn list_provider_ssh_keys_for_key(
        &self,
        ssh_key_id: &str,
    ) -> Result<Vec<ProviderSshKey>> {
        let rows = sqlx::query(
            "SELECT id, ssh_key_id, provider, provider_key_id, created_at, metadata
             FROM provider_ssh_keys WHERE ssh_key_id = $1",
        )
        .bind(ssh_key_id)
        .fetch_all(&self.pool)
        .await?;

        let mappings = rows
            .into_iter()
            .filter_map(|r| {
                let id: String = r.get("id");
                let provider_str: String = r.get("provider");

                let provider = match Provider::from_str(&provider_str) {
                    Ok(p) => p,
                    Err(e) => {
                        tracing::error!(
                            provider_ssh_key_id = %id,
                            provider = %provider_str,
                            error = %e,
                            "Failed to parse provider from provider_ssh_keys row, skipping"
                        );
                        return None;
                    }
                };

                Some(ProviderSshKey {
                    id,
                    ssh_key_id: r.get("ssh_key_id"),
                    provider,
                    provider_key_id: r.get("provider_key_id"),
                    created_at: r.get("created_at"),
                    metadata: r.get("metadata"),
                })
            })
            .collect();

        Ok(mappings)
    }
}
