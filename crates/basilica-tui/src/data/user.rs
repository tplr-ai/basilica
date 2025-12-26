//! User mode data fetching - connects to Basilica API via SDK

use anyhow::Result;
use basilica_sdk::BasilicaClient;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{debug, error, info};

/// User mode data state
#[derive(Debug, Default, Clone)]
pub struct UserData {
    /// Active rentals
    pub rentals: Vec<RentalInfo>,
    /// Available GPU offerings
    pub offerings: Vec<GpuOffering>,
    /// Active deployments
    pub deployments: Vec<DeploymentInfo>,
    /// Account balance
    pub balance: Option<BalanceInfo>,
    /// Transaction history
    pub transactions: Vec<Transaction>,
    /// Loading states
    pub loading: LoadingState,
    /// Last error message
    pub last_error: Option<String>,
}

#[derive(Debug, Default, Clone)]
pub struct LoadingState {
    pub rentals: bool,
    pub offerings: bool,
    pub deployments: bool,
    pub balance: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RentalInfo {
    pub id: String,
    pub gpu_type: String,
    pub gpu_count: u32,
    pub status: String,
    pub uptime_minutes: u64,
    pub cost: f64,
    pub container_image: String,
    pub ssh_host: Option<String>,
    pub ssh_port: Option<u16>,
    pub ssh_user: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuOffering {
    pub gpu_type: String,
    pub gpu_count: u32,
    pub memory_gb: u32,
    pub price_per_hour: f64,
    pub source: String,
    pub available: u32,
    pub node_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentInfo {
    pub name: String,
    pub deployment_type: String,
    pub status: String,
    pub replicas_ready: u32,
    pub replicas_desired: u32,
    pub gpu_type: String,
    pub gpu_count: u32,
    pub url: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BalanceInfo {
    pub available_tao: f64,
    pub available_usd: f64,
    pub spent_today: f64,
    pub spent_this_month: f64,
    pub active_spend_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transaction {
    pub timestamp: String,
    pub transaction_type: String,
    pub description: String,
    pub amount: f64,
    pub is_credit: bool,
}

impl UserData {
    /// Create new user data instance
    pub fn new() -> Self {
        Self::default()
    }

    /// Refresh all data from API
    pub async fn refresh_all(&mut self, client: &Arc<BasilicaClient>) -> Result<()> {
        debug!("Refreshing all user data...");
        self.loading = LoadingState {
            rentals: true,
            offerings: true,
            deployments: true,
            balance: true,
        };
        self.last_error = None;

        // Fetch in parallel using standalone functions
        let client_clone = client.clone();
        let (rentals_result, offerings_result, deployments_result, balance_result) = tokio::join!(
            fetch_rentals_data(&client_clone),
            fetch_offerings_data(&client_clone),
            fetch_deployments_data(&client_clone),
            fetch_balance_data(&client_clone),
        );

        // Process results
        match rentals_result {
            Ok(rentals) => self.rentals = rentals,
            Err(e) => {
                error!("Failed to fetch rentals: {}", e);
                self.last_error = Some(format!("Rentals: {}", e));
            }
        }
        match offerings_result {
            Ok(offerings) => self.offerings = offerings,
            Err(e) => {
                error!("Failed to fetch offerings: {}", e);
                self.last_error = Some(format!("Offerings: {}", e));
            }
        }
        match deployments_result {
            Ok(deployments) => self.deployments = deployments,
            Err(e) => {
                error!("Failed to fetch deployments: {}", e);
                self.last_error = Some(format!("Deployments: {}", e));
            }
        }
        match balance_result {
            Ok(balance) => self.balance = Some(balance),
            Err(e) => {
                error!("Failed to fetch balance: {}", e);
                self.last_error = Some(format!("Balance: {}", e));
            }
        }

        self.loading = LoadingState::default();
        info!(
            "Refresh complete: {} rentals, {} offerings, {} deployments",
            self.rentals.len(),
            self.offerings.len(),
            self.deployments.len()
        );
        Ok(())
    }

    /// Fetch rentals from API
    async fn fetch_rentals(&mut self, client: &Arc<BasilicaClient>) -> Result<()> {
        self.loading.rentals = true;

        match client.list_rentals(None).await {
            Ok(response) => {
                self.rentals = response
                    .rentals
                    .into_iter()
                    .map(|r| {
                        // Parse created_at to compute uptime
                        let uptime_minutes = chrono::DateTime::parse_from_rfc3339(&r.created_at)
                            .map(|start| {
                                let now = chrono::Utc::now();
                                let duration = now.signed_duration_since(start);
                                duration.num_minutes().max(0) as u64
                            })
                            .unwrap_or(0);

                        // Extract GPU info from gpu_specs
                        let (gpu_type, gpu_count) = if let Some(spec) = r.gpu_specs.first() {
                            (spec.name.clone(), r.gpu_specs.len() as u32)
                        } else {
                            ("Unknown".to_string(), 0)
                        };

                        RentalInfo {
                            id: r.rental_id,
                            gpu_type,
                            gpu_count,
                            status: format!("{:?}", r.state),
                            uptime_minutes,
                            cost: 0.0, // TODO: Get from usage API
                            container_image: r.container_image,
                            ssh_host: None, // Need to fetch from rental details
                            ssh_port: None,
                            ssh_user: None,
                        }
                    })
                    .collect();
                self.loading.rentals = false;
                Ok(())
            }
            Err(e) => {
                self.loading.rentals = false;
                Err(anyhow::anyhow!("Failed to list rentals: {}", e))
            }
        }
    }

    /// Fetch available GPU offerings from API
    async fn fetch_offerings(&mut self, client: &Arc<BasilicaClient>) -> Result<()> {
        self.loading.offerings = true;

        match client.list_available_nodes(None).await {
            Ok(response) => {
                self.offerings = response
                    .available_nodes
                    .into_iter()
                    .map(|n| {
                        // Extract GPU info from node details
                        let (gpu_type, memory_gb, gpu_count) =
                            if let Some(spec) = n.node.gpu_specs.first() {
                                (
                                    spec.name.clone(),
                                    spec.memory_gb,
                                    n.node.gpu_specs.len() as u32,
                                )
                            } else {
                                ("Unknown".to_string(), 0, 0)
                            };

                        let price_per_hour = n
                            .node
                            .hourly_rate_cents
                            .map(|c| c as f64 / 100.0)
                            .unwrap_or(0.0);

                        GpuOffering {
                            gpu_type,
                            gpu_count,
                            memory_gb,
                            price_per_hour,
                            source: "basilica".to_string(),
                            available: 1,
                            node_id: Some(n.node.id),
                        }
                    })
                    .collect();
                self.loading.offerings = false;
                Ok(())
            }
            Err(e) => {
                self.loading.offerings = false;
                Err(anyhow::anyhow!("Failed to list offerings: {}", e))
            }
        }
    }

    /// Fetch deployments from API
    async fn fetch_deployments(&mut self, client: &Arc<BasilicaClient>) -> Result<()> {
        self.loading.deployments = true;

        match client.list_deployments().await {
            Ok(response) => {
                self.deployments = response
                    .deployments
                    .into_iter()
                    .map(|d| DeploymentInfo {
                        name: d.instance_name,
                        deployment_type: "deployment".to_string(),
                        status: d.state,
                        replicas_ready: d.replicas.ready,
                        replicas_desired: d.replicas.desired,
                        gpu_type: String::new(),
                        gpu_count: 0,
                        url: Some(d.url),
                    })
                    .collect();
                self.loading.deployments = false;
                Ok(())
            }
            Err(e) => {
                self.loading.deployments = false;
                Err(anyhow::anyhow!("Failed to list deployments: {}", e))
            }
        }
    }

    /// Fetch balance from API
    async fn fetch_balance(&mut self, client: &Arc<BasilicaClient>) -> Result<()> {
        self.loading.balance = true;

        match client.get_balance().await {
            Ok(response) => {
                // Parse balance string (e.g., "12.5 TAO" or just a number)
                let balance_tao: f64 = response
                    .balance
                    .split_whitespace()
                    .next()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0.0);

                self.balance = Some(BalanceInfo {
                    available_tao: balance_tao,
                    available_usd: balance_tao * 10.0, // TODO: Get actual TAO/USD rate
                    spent_today: 0.0,
                    spent_this_month: 0.0,
                    active_spend_rate: 0.0,
                });
                self.loading.balance = false;
                Ok(())
            }
            Err(e) => {
                self.loading.balance = false;
                Err(anyhow::anyhow!("Failed to fetch balance: {}", e))
            }
        }
    }

    /// Refresh rentals only
    pub async fn refresh_rentals(&mut self, client: &Arc<BasilicaClient>) -> Result<()> {
        self.fetch_rentals(client).await
    }

    /// Refresh balance only
    pub async fn refresh_balance(&mut self, client: &Arc<BasilicaClient>) -> Result<()> {
        self.fetch_balance(client).await
    }

    /// Refresh offerings only
    pub async fn refresh_offerings(&mut self, client: &Arc<BasilicaClient>) -> Result<()> {
        self.fetch_offerings(client).await
    }

    /// Refresh deployments only
    pub async fn refresh_deployments(&mut self, client: &Arc<BasilicaClient>) -> Result<()> {
        self.fetch_deployments(client).await
    }

    /// Get total active rentals count
    pub fn active_rentals_count(&self) -> usize {
        self.rentals
            .iter()
            .filter(|r| r.status.contains("Running") || r.status.contains("Active"))
            .count()
    }

    /// Get total GPU count across all rentals
    pub fn total_gpus(&self) -> u32 {
        self.rentals.iter().map(|r| r.gpu_count).sum()
    }

    /// Get estimated hourly spend
    pub fn hourly_spend(&self) -> f64 {
        self.rentals.iter().map(|r| r.cost).sum()
    }
}

// Standalone fetch functions for parallel execution
async fn fetch_rentals_data(client: &Arc<BasilicaClient>) -> Result<Vec<RentalInfo>> {
    match client.list_rentals(None).await {
        Ok(response) => Ok(response
            .rentals
            .into_iter()
            .map(|r| {
                let uptime_minutes = chrono::DateTime::parse_from_rfc3339(&r.created_at)
                    .map(|start| {
                        let now = chrono::Utc::now();
                        let duration = now.signed_duration_since(start);
                        duration.num_minutes().max(0) as u64
                    })
                    .unwrap_or(0);

                let (gpu_type, gpu_count) = if let Some(spec) = r.gpu_specs.first() {
                    (spec.name.clone(), r.gpu_specs.len() as u32)
                } else {
                    ("Unknown".to_string(), 0)
                };

                RentalInfo {
                    id: r.rental_id,
                    gpu_type,
                    gpu_count,
                    status: format!("{:?}", r.state),
                    uptime_minutes,
                    cost: 0.0,
                    container_image: r.container_image,
                    ssh_host: None,
                    ssh_port: None,
                    ssh_user: None,
                }
            })
            .collect()),
        Err(e) => Err(anyhow::anyhow!("Failed to list rentals: {}", e)),
    }
}

async fn fetch_offerings_data(client: &Arc<BasilicaClient>) -> Result<Vec<GpuOffering>> {
    match client.list_available_nodes(None).await {
        Ok(response) => Ok(response
            .available_nodes
            .into_iter()
            .map(|n| {
                let (gpu_type, memory_gb, gpu_count) = if let Some(spec) = n.node.gpu_specs.first()
                {
                    (
                        spec.name.clone(),
                        spec.memory_gb,
                        n.node.gpu_specs.len() as u32,
                    )
                } else {
                    ("Unknown".to_string(), 0, 0)
                };

                let price_per_hour = n
                    .node
                    .hourly_rate_cents
                    .map(|c| c as f64 / 100.0)
                    .unwrap_or(0.0);

                GpuOffering {
                    gpu_type,
                    gpu_count,
                    memory_gb,
                    price_per_hour,
                    source: "basilica".to_string(),
                    available: 1,
                    node_id: Some(n.node.id),
                }
            })
            .collect()),
        Err(e) => Err(anyhow::anyhow!("Failed to list offerings: {}", e)),
    }
}

async fn fetch_deployments_data(client: &Arc<BasilicaClient>) -> Result<Vec<DeploymentInfo>> {
    match client.list_deployments().await {
        Ok(response) => Ok(response
            .deployments
            .into_iter()
            .map(|d| DeploymentInfo {
                name: d.instance_name,
                deployment_type: "deployment".to_string(),
                status: d.state,
                replicas_ready: d.replicas.ready,
                replicas_desired: d.replicas.desired,
                gpu_type: String::new(),
                gpu_count: 0,
                url: Some(d.url),
            })
            .collect()),
        Err(e) => Err(anyhow::anyhow!("Failed to list deployments: {}", e)),
    }
}

async fn fetch_balance_data(client: &Arc<BasilicaClient>) -> Result<BalanceInfo> {
    match client.get_balance().await {
        Ok(response) => {
            let balance_tao: f64 = response
                .balance
                .split_whitespace()
                .next()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0.0);

            Ok(BalanceInfo {
                available_tao: balance_tao,
                available_usd: balance_tao * 10.0, // TODO: Get actual TAO/USD rate
                spent_today: 0.0,
                spent_this_month: 0.0,
                active_spend_rate: 0.0,
            })
        }
        Err(e) => Err(anyhow::anyhow!("Failed to fetch balance: {}", e)),
    }
}
