//! User mode data fetching

use anyhow::Result;
use serde::{Deserialize, Serialize};

/// User mode data state
#[derive(Debug, Default)]
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
}

#[derive(Debug, Default)]
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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuOffering {
    pub gpu_type: String,
    pub gpu_count: u32,
    pub memory_gb: u32,
    pub price_per_hour: f64,
    pub source: String,
    pub available: u32,
    pub use_case: String,
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
    pub async fn refresh_all(&mut self, _api_url: &str) -> Result<()> {
        // TODO: Implement actual API calls using basilica-sdk
        self.loading = LoadingState {
            rentals: true,
            offerings: true,
            deployments: true,
            balance: true,
        };

        // Simulate API response
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        self.loading = LoadingState::default();
        Ok(())
    }

    /// Refresh rentals only
    pub async fn refresh_rentals(&mut self, _api_url: &str) -> Result<()> {
        self.loading.rentals = true;
        // TODO: Implement actual API call
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
        self.loading.rentals = false;
        Ok(())
    }

    /// Refresh balance only
    pub async fn refresh_balance(&mut self, _api_url: &str) -> Result<()> {
        self.loading.balance = true;
        // TODO: Implement actual API call
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
        self.loading.balance = false;
        Ok(())
    }
}

