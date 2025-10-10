//! # Chain Registration
//!
//! Common chain registration functionality for neurons (miners and validators) to register
//! on-chain and publish their axon endpoints for discovery.

use crate::service::Service;
use anyhow::Result;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{error, info, warn};

/// Configuration for chain registration
#[derive(Debug, Clone)]
pub struct RegistrationConfig {
    /// Network ID
    pub netuid: u16,
    /// Network name (e.g., "finney", "local")
    pub network: String,
    /// Axon port
    pub axon_port: u16,
    /// External IP (optional - will auto-detect if not provided)
    pub external_ip: Option<String>,
    /// Spoofed IP for local development (e.g., "10.0.0.1" for miners, "10.0.0.2" for validators)
    pub local_spoofed_ip: String,
    /// Neuron type for logging (e.g., "miner", "validator")
    pub neuron_type: String,
}

/// Chain registration service for one-time startup registration
///
/// This service is used by both miners and validators to:
/// 1. Register their presence on the Bittensor network
/// 2. Publish their axon endpoint for discovery
/// 3. Discover their assigned UID from the metagraph
///
/// The UID is dynamically discovered from the chain based on the neuron's hotkey,
/// ensuring it's always accurate and up-to-date. UIDs should never be hardcoded
/// in configuration files.
#[derive(Clone)]
pub struct ChainRegistration {
    config: RegistrationConfig,
    bittensor_service: Arc<Service>,
    state: Arc<RwLock<RegistrationState>>,
}

/// Internal registration state
#[derive(Debug)]
struct RegistrationState {
    /// Current registration status
    is_registered: bool,
    /// Registration timestamp
    registration_time: Option<chrono::DateTime<chrono::Utc>>,
    /// Discovered UID from metagraph
    discovered_uid: Option<u16>,
}

/// Snapshot of the current registration state
#[derive(Debug, Clone)]
pub struct RegistrationStateSnapshot {
    pub is_registered: bool,
    pub registration_time: Option<chrono::DateTime<chrono::Utc>>,
    pub discovered_uid: Option<u16>,
}

impl ChainRegistration {
    /// Create a new chain registration service
    pub fn new(config: RegistrationConfig, bittensor_service: Arc<Service>) -> Self {
        info!(
            "Initializing chain registration for {} on netuid: {}",
            config.neuron_type, config.netuid
        );

        let state = Arc::new(RwLock::new(RegistrationState {
            is_registered: false,
            registration_time: None,
            discovered_uid: None,
        }));

        Self {
            config,
            bittensor_service,
            state,
        }
    }

    /// Perform one-time startup registration
    pub async fn register_startup(&self) -> Result<()> {
        info!(
            "Performing one-time startup chain registration for {}",
            self.config.neuron_type
        );

        // First, check if our hotkey is registered in the metagraph
        let our_account_id = self.bittensor_service.get_account_id();
        info!(
            "Checking registration for {} hotkey: {}",
            self.config.neuron_type, our_account_id
        );

        // Get metagraph and use discovery to find our neuron
        let metagraph = self
            .bittensor_service
            .get_metagraph(self.config.netuid)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to get metagraph: {}", e))?;

        let discovery = crate::discovery::NeuronDiscovery::new(&metagraph);
        let our_neuron = discovery.find_neuron_by_hotkey(&our_account_id.to_string());

        if let Some(neuron) = our_neuron {
            // Update our state with the discovered UID
            let mut state = self.state.write().await;
            state.discovered_uid = Some(neuron.uid);
            drop(state);

            info!(
                "Found {} hotkey registered with UID: {}",
                self.config.neuron_type, neuron.uid
            );

            // Create the socket address for the axon
            let axon_ip = self.determine_axon_ip().await?;
            let axon_addr = format!("{}:{}", axon_ip, self.config.axon_port)
                .parse()
                .map_err(|e| anyhow::anyhow!("Invalid axon address: {}", e))?;

            // Check if axon info has changed
            let needs_update = if let Some(current_axon) = &neuron.axon_info {
                let current_addr = &current_axon.socket_addr;
                if current_addr != &axon_addr {
                    info!(
                        "Axon address has changed from {} to {} - updating registration",
                        current_addr, axon_addr
                    );
                    true
                } else {
                    info!(
                        "Axon already registered at {} - skipping serve_axon to avoid rate limiting",
                        current_addr
                    );
                    false
                }
            } else {
                info!(
                    "No axon info found for neuron - registering new axon at {}",
                    axon_addr
                );
                true
            };

            if needs_update {
                info!(
                    "Registering {} axon at address: {}",
                    self.config.neuron_type, axon_addr
                );

                // Register axon once at startup
                match self
                    .bittensor_service
                    .serve_axon(self.config.netuid, axon_addr)
                    .await
                {
                    Ok(()) => {
                        let mut state = self.state.write().await;
                        state.is_registered = true;
                        state.registration_time = Some(chrono::Utc::now());
                        info!(
                            "{} startup chain registration successful",
                            self.config.neuron_type
                        );
                    }
                    Err(e) => {
                        // Check if this is a "Transaction is outdated" error
                        let error_str = e.to_string();
                        if error_str.contains("Transaction is outdated") {
                            warn!(
                                "Axon registration skipped - likely already registered at {}. Error: {}",
                                axon_addr, e
                            );
                            // Mark as registered anyway since this error usually means it's already registered
                            let mut state = self.state.write().await;
                            state.is_registered = true;
                            state.registration_time = Some(chrono::Utc::now());
                        } else if error_str.contains("Custom error: 12") {
                            // Handle rate limiting error specifically
                            error!(
                                "Rate limited when calling serve_axon (Custom error: 12). This typically happens when calling serve_axon too frequently. Current axon: {:?}, attempted: {}",
                                neuron.axon_info, axon_addr
                            );
                            return Err(anyhow::anyhow!(
                                "Rate limited when updating axon registration. Please wait before retrying."
                            ));
                        } else {
                            error!(
                                "{} startup chain registration failed: {}",
                                self.config.neuron_type, e
                            );
                            return Err(anyhow::anyhow!(
                                "Failed to register {} axon: {}",
                                self.config.neuron_type,
                                e
                            ));
                        }
                    }
                }
            } else {
                // Axon info hasn't changed, mark as registered without calling serve_axon
                let mut state = self.state.write().await;
                state.is_registered = true;
                state.registration_time = Some(chrono::Utc::now());
                info!(
                    "{} registration completed - using existing axon endpoint",
                    self.config.neuron_type
                );
            }
        } else {
            error!(
                "Hotkey {} is not registered on subnet {} - please register your {} first",
                our_account_id, self.config.netuid, self.config.neuron_type
            );
            return Err(anyhow::anyhow!(
                "{} hotkey {} is not registered on subnet {}. Please register your {} using btcli before starting.",
                self.config.neuron_type, our_account_id, self.config.netuid, self.config.neuron_type
            ));
        }

        Ok(())
    }

    /// Determine the appropriate IP address for the axon
    async fn determine_axon_ip(&self) -> Result<String> {
        if let Some(external_ip) = &self.config.external_ip {
            // Use configured external IP if provided
            info!("Using configured external IP: {}", external_ip);
            Ok(external_ip.clone())
        } else if self.config.network == "local" {
            // For local development, use a private network IP that will pass chain validation
            warn!(
                "Using spoofed IP {} for local development - axon won't be reachable at this address",
                self.config.local_spoofed_ip
            );
            Ok(self.config.local_spoofed_ip.clone())
        } else {
            // For production without external_ip, auto-detect public IP
            info!("No external_ip configured, auto-detecting public IP address...");
            let detected_ip = basilica_common::network::get_public_ip_with_timeout(5).await;

            if detected_ip == "<unknown-ip>" {
                error!("Failed to auto-detect public IP address");
                return Err(anyhow::anyhow!(
                    "Could not auto-detect public IP address. Please set external_ip in configuration or check your internet connection"
                ));
            }

            info!("Auto-detected public IP: {}", detected_ip);
            warn!(
                "Using auto-detected IP {}. For production, consider setting external_ip in configuration for reliability",
                detected_ip
            );

            Ok(detected_ip)
        }
    }

    /// Get current registration state
    pub async fn get_state(&self) -> RegistrationStateSnapshot {
        let state = self.state.read().await;
        RegistrationStateSnapshot {
            is_registered: state.is_registered,
            registration_time: state.registration_time,
            discovered_uid: state.discovered_uid,
        }
    }

    /// Get the UID discovered from the metagraph
    ///
    /// Returns the neuron's UID as discovered from the chain based on its hotkey.
    /// This is the authoritative source for a neuron's UID and should be used
    /// instead of any hardcoded configuration values.
    ///
    /// Returns `None` if:
    /// - The neuron hasn't been registered yet
    /// - Registration is skipped (local testing mode)
    /// - The hotkey wasn't found in the metagraph
    pub async fn get_discovered_uid(&self) -> Option<u16> {
        self.state.read().await.discovered_uid
    }

    /// Health check for registration service
    pub async fn health_check(&self) -> Result<()> {
        let state = self.state.read().await;

        if !state.is_registered {
            return Err(anyhow::anyhow!("Chain registration not completed"));
        }

        // Check if registration is too old (warn but don't fail)
        if let Some(reg_time) = state.registration_time {
            let elapsed = chrono::Utc::now().signed_duration_since(reg_time);
            if elapsed > chrono::Duration::hours(24) {
                warn!(
                    "Chain registration is old (registered {} hours ago)",
                    elapsed.num_hours()
                );
            }
        }

        Ok(())
    }
}

/// Builder for RegistrationConfig
pub struct RegistrationConfigBuilder {
    netuid: u16,
    network: String,
    axon_port: u16,
    external_ip: Option<String>,
    local_spoofed_ip: String,
    neuron_type: String,
}

impl RegistrationConfigBuilder {
    pub fn new(netuid: u16, network: String, axon_port: u16) -> Self {
        Self {
            netuid,
            network,
            axon_port,
            external_ip: None,
            local_spoofed_ip: "10.0.0.1".to_string(),
            neuron_type: "neuron".to_string(),
        }
    }

    pub fn external_ip(mut self, ip: Option<String>) -> Self {
        self.external_ip = ip;
        self
    }

    pub fn local_spoofed_ip(mut self, ip: String) -> Self {
        self.local_spoofed_ip = ip;
        self
    }

    pub fn neuron_type(mut self, neuron_type: String) -> Self {
        self.neuron_type = neuron_type;
        self
    }

    pub fn build(self) -> RegistrationConfig {
        RegistrationConfig {
            netuid: self.netuid,
            network: self.network,
            axon_port: self.axon_port,
            external_ip: self.external_ip,
            local_spoofed_ip: self.local_spoofed_ip,
            neuron_type: self.neuron_type,
        }
    }
}
