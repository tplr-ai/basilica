//! TDX Host Validation and Setup
//!
//! Self-contained module for validating and configuring TDX on bare-metal hosts.
//! All setup logic is embedded - no external dependencies, repos, or reference files.
//!
//! This module handles:
//! - Checking TDX host status (CPU capability, kernel, services)
//! - Installing TDX host packages (Intel kernel, QEMU, libvirt)
//! - Configuring GRUB for TDX (nohibernate)
//! - Installing attestation packages (PCCS, QGS)
//! - Configuring PCCS with Intel API key
//! - Verifying TDX host setup

use anyhow::{Context, Result};
use std::sync::Arc;
use std::time::Duration;
use tokio::time::{sleep, Instant};
use tracing::{debug, info, warn};

use basilica_common::ssh::SshConnectionDetails;
use basilica_tee::bootstrap::commands::{tdx_guest, tdx_host};

use crate::ssh::ValidatorSshClient;

/// Default timeout waiting for node to come back after reboot
const DEFAULT_REBOOT_TIMEOUT: Duration = Duration::from_secs(300); // 5 minutes
/// How often to poll for node availability
const REBOOT_POLL_INTERVAL: Duration = Duration::from_secs(10);

/// TDX host status information
#[derive(Debug, Clone, Default)]
pub struct TdxHostStatus {
    /// Whether TDX module is initialized
    pub tdx_initialized: bool,
    /// Whether CPU supports TDX
    pub cpu_tdx_capable: bool,
    /// Current kernel version
    pub kernel_version: String,
    /// Whether Intel kernel is installed
    pub intel_kernel_installed: bool,
    /// PCCS service status
    pub pccs_running: bool,
    /// QGS service status
    pub qgs_running: bool,
    /// Whether BIOS has TDX enabled
    pub bios_tdx_enabled: bool,
    /// Whether a reboot is required
    pub reboot_required: bool,
}

/// Result of TDX host setup
#[derive(Debug, Clone, Default)]
pub struct TdxHostSetupResult {
    /// Whether packages were installed
    pub packages_installed: bool,
    /// Whether GRUB was configured
    pub grub_configured: bool,
    /// Whether attestation packages were installed
    pub attestation_installed: bool,
    /// Whether PCCS was configured
    pub pccs_configured: bool,
    /// Whether a reboot is required after setup
    pub reboot_required: bool,
    /// Kernel version that will be active after reboot
    pub target_kernel: Option<String>,
}

/// TDX VM readiness status
#[derive(Debug, Clone, Default)]
pub struct TdxVmReadiness {
    /// Overall readiness for TDX VMs
    pub ready: bool,
    /// KVM device available
    pub kvm_available: bool,
    /// QEMU installed
    pub qemu_installed: bool,
    /// QEMU has TDX support
    pub qemu_tdx_support: bool,
    /// virsh installed
    pub virsh_installed: bool,
    /// libvirt has TDX capability
    pub libvirt_tdx_support: bool,
    /// OVMF firmware available
    pub ovmf_available: bool,
    /// Path to OVMF firmware
    pub ovmf_path: Option<String>,
    /// libvirtd service status
    pub libvirtd_running: bool,
    /// List of errors
    pub errors: Vec<String>,
    /// List of warnings
    pub warnings: Vec<String>,
}

/// Result of TDX guest quote generation test
#[derive(Debug, Clone, Default)]
pub struct TdxGuestQuoteResult {
    /// Whether the test was successful
    pub success: bool,
    /// TDX device path inside guest (if found)
    pub tdx_device: Option<String>,
    /// Quote size in bytes (if generated)
    pub quote_size: Option<usize>,
    /// Preview of quote hex (first 200 chars)
    pub quote_preview: Option<String>,
    /// Method used for quote generation
    pub method: Option<String>,
    /// Error message if failed
    pub error: Option<String>,
}

/// TDX Host Validator
///
/// Validates and sets up TDX on bare-metal hosts via SSH.
/// All commands are embedded in the code - no external dependencies.
pub struct TdxHostValidator {
    ssh_client: Arc<ValidatorSshClient>,
}

impl TdxHostValidator {
    /// Create a new TDX host validator
    pub fn new(ssh_client: Arc<ValidatorSshClient>) -> Self {
        Self { ssh_client }
    }

    /// Check current TDX host status
    ///
    /// Returns detailed status including TDX initialization, kernel info, and service status.
    pub async fn check_status(&self, connection: &SshConnectionDetails) -> Result<TdxHostStatus> {
        info!("[TDX Host] Checking TDX host status");

        let output = self
            .ssh_client
            .execute_command(connection, tdx_host::CHECK_TDX_HOST_STATUS, true)
            .await
            .context("Failed to check TDX host status")?;

        let status = self.parse_status_output(&output);

        // Also check if reboot is required
        let reboot_output = self
            .ssh_client
            .execute_command(connection, tdx_host::CHECK_REBOOT_REQUIRED, true)
            .await
            .unwrap_or_default();

        let mut status = status;
        status.reboot_required = reboot_output.contains("REBOOT_REQUIRED:");

        debug!("[TDX Host] Status: {:?}", status);
        Ok(status)
    }

    /// Parse the output of CHECK_TDX_HOST_STATUS command
    fn parse_status_output(&self, output: &str) -> TdxHostStatus {
        let mut status = TdxHostStatus::default();

        // Parse: TDX_STATUS:init=yes,cpu=yes,kernel=6.8.0-intel,intel=yes,pccs=active,qgs=active,bios=yes
        for line in output.lines() {
            if let Some(data) = line.strip_prefix("TDX_STATUS:") {
                for pair in data.split(',') {
                    let parts: Vec<&str> = pair.split('=').collect();
                    if parts.len() == 2 {
                        let key = parts[0].trim();
                        let value = parts[1].trim();
                        match key {
                            "init" => status.tdx_initialized = value == "yes",
                            "cpu" => status.cpu_tdx_capable = value == "yes",
                            "kernel" => status.kernel_version = value.to_string(),
                            "intel" => status.intel_kernel_installed = value == "yes",
                            "pccs" => status.pccs_running = value == "active",
                            "qgs" => status.qgs_running = value == "active",
                            "bios" => status.bios_tdx_enabled = value == "yes",
                            _ => {}
                        }
                    }
                }
            }
        }

        status
    }

    /// Install TDX host packages
    ///
    /// Installs: linux-image-intel, qemu-system-x86, libvirt, ovmf
    pub async fn install_packages(&self, connection: &SshConnectionDetails) -> Result<bool> {
        info!("[TDX Host] Installing TDX host packages");

        let output = self
            .ssh_client
            .execute_command(connection, tdx_host::INSTALL_TDX_HOST_PACKAGES, true)
            .await
            .context("Failed to install TDX host packages")?;

        let success = output.contains("PACKAGES_INSTALLED:");

        if success {
            info!("[TDX Host] TDX host packages installed successfully");
        } else if output.contains("UNSUPPORTED_UBUNTU:") {
            warn!("[TDX Host] Unsupported Ubuntu version for TDX");
            return Ok(false);
        } else {
            warn!(
                "[TDX Host] Package installation may have failed: {}",
                output
            );
        }

        Ok(success)
    }

    /// Configure GRUB for TDX
    ///
    /// Adds nohibernate to GRUB command line (required for TDX).
    pub async fn configure_grub(&self, connection: &SshConnectionDetails) -> Result<bool> {
        info!("[TDX Host] Configuring GRUB for TDX");

        let output = self
            .ssh_client
            .execute_command(connection, tdx_host::CONFIGURE_GRUB_TDX, true)
            .await
            .context("Failed to configure GRUB")?;

        let reboot_required = output.contains("GRUB_UPDATED:reboot_required");
        let already_configured = output.contains("GRUB_OK:no_changes");

        if reboot_required {
            info!("[TDX Host] GRUB configured, reboot required");
        } else if already_configured {
            info!("[TDX Host] GRUB already configured");
        }

        Ok(reboot_required || already_configured)
    }

    /// Install attestation packages
    ///
    /// Installs: sgx-dcap-pccs, tdx-qgs, libsgx-dcap-default-qpl
    pub async fn install_attestation_packages(
        &self,
        connection: &SshConnectionDetails,
    ) -> Result<bool> {
        info!("[TDX Host] Installing attestation packages");

        let output = self
            .ssh_client
            .execute_command(connection, tdx_host::INSTALL_ATTESTATION_PACKAGES, true)
            .await
            .context("Failed to install attestation packages")?;

        let success = output.contains("ATTESTATION_PACKAGES_INSTALLED");

        if success {
            info!("[TDX Host] Attestation packages installed successfully");
        } else {
            warn!("[TDX Host] Attestation package installation may have issues");
        }

        Ok(success)
    }

    /// Configure PCCS with Intel API key
    ///
    /// Sets up PCCS config and starts the service.
    pub async fn configure_pccs(
        &self,
        connection: &SshConnectionDetails,
        intel_api_key: &str,
    ) -> Result<bool> {
        info!("[TDX Host] Configuring PCCS service");

        // Create command with environment variables
        let cmd = format!(
            "INTEL_API_KEY='{}' bash -c '{}'",
            intel_api_key,
            tdx_host::CONFIGURE_PCCS.replace('\n', "; ").trim()
        );

        let output = self
            .ssh_client
            .execute_command(connection, &cmd, true)
            .await
            .context("Failed to configure PCCS")?;

        let success = output.contains("PCCS_CONFIGURED:running");
        let failed = output.contains("PCCS_CONFIGURED:failed_to_start");

        if success {
            info!("[TDX Host] PCCS configured and running");
        } else if failed {
            warn!("[TDX Host] PCCS configured but failed to start");
        } else if output.contains("PCCS_CONFIG_ERROR:") {
            warn!("[TDX Host] PCCS configuration error: {}", output);
        }

        Ok(success)
    }

    /// Register platform with Intel PCCS
    pub async fn register_platform(&self, connection: &SshConnectionDetails) -> Result<bool> {
        info!("[TDX Host] Registering platform with Intel PCCS");

        let output = self
            .ssh_client
            .execute_command(connection, tdx_host::REGISTER_PLATFORM, true)
            .await
            .context("Failed to register platform")?;

        let success = output.contains("PLATFORM_REGISTRATION:success")
            || output.contains("PLATFORM_REGISTRATION:attempted");

        if success {
            info!("[TDX Host] Platform registration completed");
        } else if output.contains("PLATFORM_REGISTRATION:tool_not_found") {
            warn!("[TDX Host] PCKIDRetrievalTool not found");
            return Ok(false);
        }

        Ok(success)
    }

    /// Verify TDX host setup
    ///
    /// Checks that TDX is properly initialized and services are running.
    pub async fn verify_host(
        &self,
        connection: &SshConnectionDetails,
    ) -> Result<(bool, Vec<String>, Vec<String>)> {
        info!("[TDX Host] Verifying TDX host setup");

        let output = self
            .ssh_client
            .execute_command(connection, tdx_host::VERIFY_TDX_HOST_FULL, true)
            .await
            .context("Failed to verify TDX host")?;

        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        // Parse: TDX_HOST_VERIFIED:ok|warnings=... or TDX_HOST_VERIFIED:failed|errors=...|warnings=...
        for line in output.lines() {
            if line.starts_with("TDX_HOST_VERIFIED:") {
                let parts: Vec<&str> = line.split('|').collect();
                for part in parts {
                    if let Some(err_list) = part.strip_prefix("errors=") {
                        for err in err_list.split(';').filter(|s| !s.is_empty()) {
                            errors.push(err.to_string());
                        }
                    }
                    if let Some(warn_list) = part.strip_prefix("warnings=") {
                        for warn in warn_list.split(';').filter(|s| !s.is_empty()) {
                            warnings.push(warn.to_string());
                        }
                    }
                }
            }
        }

        let success = output.contains("TDX_HOST_VERIFIED:ok");

        if success {
            if warnings.is_empty() {
                info!("[TDX Host] TDX host verified successfully");
            } else {
                info!("[TDX Host] TDX host verified with warnings: {:?}", warnings);
            }
        } else {
            warn!("[TDX Host] TDX host verification failed: {:?}", errors);
        }

        Ok((success, errors, warnings))
    }

    /// Check if reboot is required
    pub async fn check_reboot_required(&self, connection: &SshConnectionDetails) -> Result<bool> {
        let output = self
            .ssh_client
            .execute_command(connection, tdx_host::CHECK_REBOOT_REQUIRED, true)
            .await
            .context("Failed to check reboot status")?;

        Ok(output.contains("REBOOT_REQUIRED:"))
    }

    /// Full TDX host setup
    ///
    /// Runs complete setup: packages, GRUB, attestation.
    /// Does NOT configure PCCS (requires API key) or reboot (requires manual action).
    pub async fn setup_tdx_host(
        &self,
        connection: &SshConnectionDetails,
    ) -> Result<TdxHostSetupResult> {
        info!("[TDX Host] Starting full TDX host setup");

        let output = self
            .ssh_client
            .execute_command(connection, tdx_host::FULL_TDX_HOST_SETUP, true)
            .await
            .context("Failed to run TDX host setup")?;

        let mut result = TdxHostSetupResult::default();

        // Parse output
        for line in output.lines() {
            if line.contains("PPAs added") {
                result.packages_installed = true;
            }
            if line.contains("TDX host packages installed") {
                result.packages_installed = true;
            }
            if line.contains("Attestation packages installed") {
                result.attestation_installed = true;
            }
            if line.contains("GRUB configured") || line.contains("GRUB already configured") {
                result.grub_configured = true;
            }
            if line.starts_with("SETUP_COMPLETE:reboot_required") {
                result.reboot_required = true;
                // Extract target kernel if present
                if let Some(kernel) = line.split("target_kernel=").nth(1) {
                    result.target_kernel = Some(kernel.trim().to_string());
                }
            }
            if line.starts_with("SETUP_COMPLETE:no_reboot_needed") {
                result.reboot_required = false;
            }
            if line.starts_with("SETUP_FAILED:") {
                warn!("[TDX Host] Setup failed: {}", line);
            }
        }

        if result.packages_installed {
            info!("[TDX Host] TDX host setup completed");
            if result.reboot_required {
                info!("[TDX Host] Reboot required to activate Intel kernel");
            }
        } else {
            warn!("[TDX Host] TDX host setup may have issues");
        }

        Ok(result)
    }

    /// Check if host is TDX-ready (no further setup needed)
    pub async fn is_tdx_ready(&self, connection: &SshConnectionDetails) -> Result<bool> {
        let status = self.check_status(connection).await?;

        // TDX is ready if:
        // - TDX is initialized
        // - Intel kernel is running
        // - No reboot required
        Ok(status.tdx_initialized && !status.reboot_required)
    }

    /// Reboot the node and wait for it to come back online
    ///
    /// Returns Ok(new_kernel_version) on success, or error if timeout.
    pub async fn reboot_and_wait(
        &self,
        connection: &SshConnectionDetails,
        timeout: Option<Duration>,
    ) -> Result<String> {
        let timeout = timeout.unwrap_or(DEFAULT_REBOOT_TIMEOUT);
        info!("[TDX Host] Initiating reboot (timeout: {:?})", timeout);

        // Get current boot time to detect actual reboot
        let pre_boot_id = self.get_boot_id(connection).await.unwrap_or_default();
        debug!("[TDX Host] Pre-reboot boot_id: {}", pre_boot_id);

        // Initiate reboot (this will disconnect us)
        let reboot_result = self
            .ssh_client
            .execute_command(
                connection,
                "nohup sh -c 'sleep 2 && reboot' &>/dev/null &",
                true,
            )
            .await;

        // It's OK if this fails - the connection may drop immediately
        if let Err(e) = reboot_result {
            debug!(
                "[TDX Host] Reboot command result (expected disconnect): {}",
                e
            );
        }

        info!("[TDX Host] Reboot initiated, waiting for node to go down...");

        // Wait a moment for the node to actually start rebooting
        sleep(Duration::from_secs(5)).await;

        // Now wait for it to come back
        self.wait_for_node(connection, timeout, Some(&pre_boot_id))
            .await
    }

    /// Wait for a node to become available via SSH
    ///
    /// If `expected_different_boot_id` is provided, waits until boot_id changes
    /// (to ensure a real reboot happened, not just SSH reconnection).
    pub async fn wait_for_node(
        &self,
        connection: &SshConnectionDetails,
        timeout: Duration,
        expected_different_boot_id: Option<&str>,
    ) -> Result<String> {
        let start = Instant::now();
        let mut last_error = String::new();
        let mut node_was_down = false;

        info!(
            "[TDX Host] Waiting for node to come back online (timeout: {:?})",
            timeout
        );

        while start.elapsed() < timeout {
            // Try to connect and run a simple command
            match self
                .ssh_client
                .execute_command(connection, "uname -r", true)
                .await
            {
                Ok(kernel_version) => {
                    let kernel = kernel_version.trim().to_string();

                    // If we need to verify a reboot happened, check boot_id
                    if let Some(old_boot_id) = expected_different_boot_id {
                        match self.get_boot_id(connection).await {
                            Ok(new_boot_id) => {
                                if new_boot_id == old_boot_id && !node_was_down {
                                    // Same boot ID and node never went down - hasn't rebooted yet
                                    debug!(
                                        "[TDX Host] Boot ID unchanged, node hasn't rebooted yet"
                                    );
                                    sleep(REBOOT_POLL_INTERVAL).await;
                                    continue;
                                }
                                // Different boot ID or node was down - reboot completed
                                info!("[TDX Host] Node is back online with kernel: {}", kernel);
                                return Ok(kernel);
                            }
                            Err(e) => {
                                debug!("[TDX Host] Failed to get boot_id: {}", e);
                                // Continue waiting
                            }
                        }
                    } else {
                        // No boot_id check needed
                        info!("[TDX Host] Node is back online with kernel: {}", kernel);
                        return Ok(kernel);
                    }
                }
                Err(e) => {
                    node_was_down = true;
                    last_error = e.to_string();
                    debug!(
                        "[TDX Host] Node not yet available ({:.0}s elapsed): {}",
                        start.elapsed().as_secs(),
                        e
                    );
                }
            }

            sleep(REBOOT_POLL_INTERVAL).await;
        }

        anyhow::bail!(
            "Timeout waiting for node after {:?}. Last error: {}",
            timeout,
            last_error
        )
    }

    /// Get the system boot ID (changes on each reboot)
    async fn get_boot_id(&self, connection: &SshConnectionDetails) -> Result<String> {
        let output = self
            .ssh_client
            .execute_command(connection, "cat /proc/sys/kernel/random/boot_id", true)
            .await
            .context("Failed to get boot_id")?;
        Ok(output.trim().to_string())
    }

    /// Full TDX setup with automatic reboot handling
    ///
    /// This is the main entry point for automated TDX host provisioning.
    /// It will:
    /// 1. Check current status
    /// 2. Install packages if needed
    /// 3. Reboot if needed (and wait for node to come back)
    /// 4. Verify TDX is initialized
    ///
    /// Returns the final TdxHostStatus after all setup is complete.
    pub async fn setup_tdx_host_full(
        &self,
        connection: &SshConnectionDetails,
        intel_api_key: Option<&str>,
        reboot_timeout: Option<Duration>,
    ) -> Result<TdxHostStatus> {
        info!("[TDX Host] Starting full TDX host setup with auto-reboot");

        // Step 1: Check initial status
        let initial_status = self.check_status(connection).await?;
        info!("[TDX Host] Initial status: {:?}", initial_status);

        if initial_status.tdx_initialized {
            info!("[TDX Host] TDX already initialized, checking services...");
            // Just ensure services are running
            if !initial_status.pccs_running || !initial_status.qgs_running {
                self.install_attestation_packages(connection).await?;
            }
            if let Some(api_key) = intel_api_key {
                if !initial_status.pccs_running {
                    self.configure_pccs(connection, api_key).await?;
                }
            }
            return self.check_status(connection).await;
        }

        // Step 2: Run setup if needed
        if !initial_status.intel_kernel_installed {
            info!("[TDX Host] Installing TDX packages...");
            let setup_result = self.setup_tdx_host(connection).await?;
            info!("[TDX Host] Setup result: {:?}", setup_result);
        }

        // Step 3: Check if reboot is needed
        let reboot_required = self.check_reboot_required(connection).await?;

        if reboot_required {
            info!("[TDX Host] Reboot required, initiating...");
            let new_kernel = self.reboot_and_wait(connection, reboot_timeout).await?;
            info!("[TDX Host] Node back online with kernel: {}", new_kernel);
        }

        // Step 4: Post-reboot setup (PCCS, etc.)
        let post_reboot_status = self.check_status(connection).await?;

        if !post_reboot_status.pccs_running {
            info!("[TDX Host] Installing attestation packages...");
            self.install_attestation_packages(connection).await?;

            if let Some(api_key) = intel_api_key {
                info!("[TDX Host] Configuring PCCS...");
                self.configure_pccs(connection, api_key).await?;
                self.register_platform(connection).await?;
            }
        }

        // Step 5: Final verification
        let final_status = self.check_status(connection).await?;
        info!("[TDX Host] Final status: {:?}", final_status);

        if final_status.tdx_initialized {
            info!("[TDX Host] ✅ TDX host setup complete!");
        } else {
            warn!("[TDX Host] ⚠️ TDX not initialized - may need BIOS configuration");
        }

        Ok(final_status)
    }

    // =========================================================================
    // TDX Guest VM Methods
    // =========================================================================

    /// Check if host is ready to launch TDX VMs
    ///
    /// Verifies: /dev/kvm, QEMU TDX support, libvirt capabilities, OVMF firmware
    pub async fn check_vm_readiness(
        &self,
        connection: &SshConnectionDetails,
    ) -> Result<TdxVmReadiness> {
        info!("[TDX VM] Checking VM readiness");

        let output = self
            .ssh_client
            .execute_command(connection, tdx_host::CHECK_TDX_VM_READINESS, true)
            .await
            .context("Failed to check TDX VM readiness")?;

        let mut readiness = TdxVmReadiness::default();

        // Parse: VM_READINESS:ready=yes,kvm=yes,qemu=yes,qemu_tdx=yes,...
        for line in output.lines() {
            if let Some(data) = line.strip_prefix("VM_READINESS:") {
                for pair in data.split(',') {
                    let parts: Vec<&str> = pair.split('=').collect();
                    if parts.len() >= 2 {
                        let key = parts[0].trim();
                        let value = parts[1..].join("="); // Handle paths with =
                        match key {
                            "ready" => readiness.ready = value == "yes",
                            "kvm" => readiness.kvm_available = value == "yes",
                            "qemu" => readiness.qemu_installed = value == "yes",
                            "qemu_tdx" => readiness.qemu_tdx_support = value == "yes",
                            "virsh" => readiness.virsh_installed = value == "yes",
                            "libvirt_tdx" => readiness.libvirt_tdx_support = value == "yes",
                            "ovmf" => readiness.ovmf_available = value == "yes",
                            "ovmf_path" => {
                                if !value.is_empty() {
                                    readiness.ovmf_path = Some(value.to_string());
                                }
                            }
                            "libvirtd" => readiness.libvirtd_running = value == "active",
                            _ => {}
                        }
                    }
                }
            } else if let Some(errors) = line.strip_prefix("VM_READINESS_ERRORS:") {
                readiness.errors = errors
                    .split(';')
                    .filter(|s| !s.is_empty())
                    .map(|s| s.to_string())
                    .collect();
            } else if let Some(warnings) = line.strip_prefix("VM_READINESS_WARNINGS:") {
                readiness.warnings = warnings
                    .split(';')
                    .filter(|s| !s.is_empty())
                    .map(|s| s.to_string())
                    .collect();
            }
        }

        debug!("[TDX VM] Readiness: {:?}", readiness);
        Ok(readiness)
    }

    /// Download TDX guest image if not present
    pub async fn download_guest_image(&self, connection: &SshConnectionDetails) -> Result<String> {
        info!("[TDX VM] Downloading guest image");

        let output = self
            .ssh_client
            .execute_command(connection, tdx_guest::DOWNLOAD_TDX_GUEST_IMAGE, true)
            .await
            .context("Failed to download guest image")?;

        // Parse: GUEST_IMAGE:downloaded:/path or GUEST_IMAGE:exists:/path
        for line in output.lines() {
            if let Some(rest) = line.strip_prefix("GUEST_IMAGE:") {
                let parts: Vec<&str> = rest.splitn(2, ':').collect();
                if parts.len() == 2 {
                    let path = parts[1].to_string();
                    info!("[TDX VM] Guest image at: {}", path);
                    return Ok(path);
                }
            }
        }

        anyhow::bail!("Failed to download guest image: {}", output)
    }

    /// Create and launch a test TDX VM
    ///
    /// Returns the SSH port to connect to the VM
    pub async fn create_test_tdx_vm(
        &self,
        connection: &SshConnectionDetails,
        vm_name: &str,
    ) -> Result<u16> {
        info!("[TDX VM] Creating test VM: {}", vm_name);

        // Step 1: Ensure guest image exists
        self.download_guest_image(connection).await?;

        // Step 2: Create VM disk
        let create_disk_cmd = format!(
            "VM_NAME='{}' bash -c '{}'",
            vm_name,
            tdx_guest::CREATE_TDX_VM_DISK.replace('\n', "; ").trim()
        );
        self.ssh_client
            .execute_command(connection, &create_disk_cmd, true)
            .await
            .context("Failed to create VM disk")?;

        // Step 3: Create cloud-init
        let create_cloudinit_cmd = format!(
            "VM_NAME='{}' bash -c '{}'",
            vm_name,
            tdx_guest::CREATE_CLOUD_INIT.replace('\n', "; ").trim()
        );
        self.ssh_client
            .execute_command(connection, &create_cloudinit_cmd, true)
            .await
            .context("Failed to create cloud-init")?;

        // Step 4: Launch VM
        let launch_cmd = format!(
            "VM_NAME='{}' bash -c '{}'",
            vm_name,
            tdx_guest::LAUNCH_TDX_VM_QEMU.replace('\n', "; ").trim()
        );
        let output = self
            .ssh_client
            .execute_command(connection, &launch_cmd, true)
            .await
            .context("Failed to launch TDX VM")?;

        // Parse: TDX_VM_LAUNCHED:name=...,ssh_port=2222,...
        for line in output.lines() {
            if let Some(data) = line.strip_prefix("TDX_VM_LAUNCHED:") {
                for pair in data.split(',') {
                    if let Some(port_str) = pair.strip_prefix("ssh_port=") {
                        let port: u16 = port_str.parse().context("Invalid SSH port")?;
                        info!("[TDX VM] VM launched, SSH port: {}", port);
                        return Ok(port);
                    }
                }
            }
        }

        anyhow::bail!("Failed to parse VM launch output: {}", output)
    }

    /// Wait for TDX VM to be ready (SSH accessible)
    pub async fn wait_for_vm(
        &self,
        connection: &SshConnectionDetails,
        ssh_port: u16,
        timeout: Duration,
    ) -> Result<()> {
        info!("[TDX VM] Waiting for VM to be ready (port {})", ssh_port);

        let wait_cmd = format!(
            "SSH_PORT='{}' TIMEOUT='{}' bash -c '{}'",
            ssh_port,
            timeout.as_secs(),
            tdx_guest::WAIT_FOR_TDX_VM.replace('\n', "; ").trim()
        );

        let output = self
            .ssh_client
            .execute_command(connection, &wait_cmd, true)
            .await
            .context("Failed while waiting for VM")?;

        if output.contains("TDX_VM_WAIT:ready") {
            info!("[TDX VM] VM is ready");
            Ok(())
        } else if output.contains("TDX_VM_WAIT:timeout") {
            anyhow::bail!("Timeout waiting for VM to become ready")
        } else {
            anyhow::bail!("Unexpected wait result: {}", output)
        }
    }

    /// Test TDX quote generation inside a guest VM
    pub async fn test_guest_quote_generation(
        &self,
        connection: &SshConnectionDetails,
        ssh_port: u16,
    ) -> Result<TdxGuestQuoteResult> {
        info!(
            "[TDX VM] Testing quote generation in guest (port {})",
            ssh_port
        );

        // First install attestation tools
        let install_cmd = format!(
            "SSH_PORT='{}' bash -c '{}'",
            ssh_port,
            tdx_guest::INSTALL_GUEST_ATTESTATION
                .replace('\n', "; ")
                .trim()
        );
        let install_output = self
            .ssh_client
            .execute_command(connection, &install_cmd, true)
            .await
            .unwrap_or_default();
        debug!("[TDX VM] Install output: {}", install_output);

        // Now test quote generation
        let test_cmd = format!(
            "SSH_PORT='{}' bash -c '{}'",
            ssh_port,
            tdx_guest::TEST_GUEST_QUOTE_GEN.replace('\n', "; ").trim()
        );
        let output = self
            .ssh_client
            .execute_command(connection, &test_cmd, true)
            .await
            .context("Failed to test quote generation")?;

        let mut result = TdxGuestQuoteResult::default();

        for line in output.lines() {
            if let Some(dev) = line.strip_prefix("TDX_DEVICE:") {
                result.tdx_device = Some(dev.to_string());
            } else if let Some(data) = line.strip_prefix("QUOTE_GEN:") {
                if data.starts_with("success") {
                    result.success = true;
                    result.method = Some("tdx_attest".to_string());
                    // Parse size
                    if let Some(size_part) = data.split("size=").nth(1) {
                        result.quote_size = size_part
                            .split(&[',', ' '][..])
                            .next()
                            .and_then(|s| s.parse().ok());
                    }
                } else if data == "no_tdx_device" {
                    result.error = Some("No TDX device found in guest".to_string());
                } else if data == "no_method_available" {
                    result.error = Some(
                        "TDX device found but no quote generation method available".to_string(),
                    );
                } else {
                    result.error = Some(data.to_string());
                }
            } else if let Some(preview) = line.strip_prefix("QUOTE_PREVIEW:") {
                result.quote_preview = Some(preview.to_string());
            }
        }

        if result.success {
            info!(
                "[TDX VM] Quote generation successful, size: {:?}",
                result.quote_size
            );
        } else {
            warn!("[TDX VM] Quote generation failed: {:?}", result.error);
        }

        Ok(result)
    }

    /// Cleanup a test TDX VM
    pub async fn cleanup_test_vm(
        &self,
        connection: &SshConnectionDetails,
        vm_name: &str,
    ) -> Result<()> {
        info!("[TDX VM] Cleaning up VM: {}", vm_name);

        let cleanup_cmd = format!(
            "VM_NAME='{}' bash -c '{}'",
            vm_name,
            tdx_guest::CLEANUP_TDX_VM.replace('\n', "; ").trim()
        );

        self.ssh_client
            .execute_command(connection, &cleanup_cmd, true)
            .await
            .context("Failed to cleanup VM")?;

        info!("[TDX VM] VM cleaned up");
        Ok(())
    }

    /// Run full TDX guest VM test
    ///
    /// Creates a VM, tests quote generation, and cleans up.
    /// This is the main entry point for testing TDX guest attestation.
    pub async fn run_full_guest_test(
        &self,
        connection: &SshConnectionDetails,
    ) -> Result<TdxGuestQuoteResult> {
        info!("[TDX VM] Starting full guest test");

        let output = self
            .ssh_client
            .execute_command(connection, tdx_guest::FULL_TDX_GUEST_TEST, true)
            .await
            .context("Failed to run full guest test")?;

        let mut result = TdxGuestQuoteResult::default();

        // Parse the comprehensive test output
        for line in output.lines() {
            if let Some(dev) = line.strip_prefix("TDX_DEVICE:") {
                if dev != "not_found" {
                    result.tdx_device = Some(dev.to_string());
                }
            } else if let Some(data) = line.strip_prefix("QUOTE_TEST:") {
                if data.starts_with("success") {
                    result.success = true;
                    if let Some(size_part) = data.split("size=").nth(1) {
                        result.quote_size = size_part
                            .split(&[',', ' '][..])
                            .next()
                            .and_then(|s| s.parse().ok());
                    }
                } else if data == "no_device" {
                    result.error = Some("No TDX device in guest".to_string());
                } else if data == "device_found_but_no_method" {
                    result.tdx_device = Some("/dev/tdx_guest".to_string());
                    result.error =
                        Some("TDX device found but quote generation not available".to_string());
                } else {
                    result.error = Some(data.to_string());
                }
            }
        }

        if result.success {
            info!("[TDX VM] Full guest test PASSED");
        } else {
            warn!("[TDX VM] Full guest test: {:?}", result);
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_status_output() {
        let validator = TdxHostValidator {
            ssh_client: Arc::new(ValidatorSshClient::new()),
        };

        let output = "TDX_STATUS:init=yes,cpu=yes,kernel=6.8.0-1011-intel,intel=yes,pccs=active,qgs=active,bios=yes";
        let status = validator.parse_status_output(output);

        assert!(status.tdx_initialized);
        assert!(status.cpu_tdx_capable);
        assert_eq!(status.kernel_version, "6.8.0-1011-intel");
        assert!(status.intel_kernel_installed);
        assert!(status.pccs_running);
        assert!(status.qgs_running);
        assert!(status.bios_tdx_enabled);
    }

    #[test]
    fn test_parse_status_output_not_initialized() {
        let validator = TdxHostValidator {
            ssh_client: Arc::new(ValidatorSshClient::new()),
        };

        let output = "TDX_STATUS:init=no,cpu=yes,kernel=6.8.0-generic,intel=no,pccs=inactive,qgs=inactive,bios=no";
        let status = validator.parse_status_output(output);

        assert!(!status.tdx_initialized);
        assert!(status.cpu_tdx_capable);
        assert_eq!(status.kernel_version, "6.8.0-generic");
        assert!(!status.intel_kernel_installed);
        assert!(!status.pccs_running);
        assert!(!status.qgs_running);
        assert!(!status.bios_tdx_enabled);
    }

    #[test]
    fn test_default_status() {
        let status = TdxHostStatus::default();
        assert!(!status.tdx_initialized);
        assert!(!status.cpu_tdx_capable);
        assert!(status.kernel_version.is_empty());
        assert!(!status.intel_kernel_installed);
        assert!(!status.pccs_running);
        assert!(!status.qgs_running);
        assert!(!status.bios_tdx_enabled);
        assert!(!status.reboot_required);
    }

    #[test]
    fn test_default_setup_result() {
        let result = TdxHostSetupResult::default();
        assert!(!result.packages_installed);
        assert!(!result.grub_configured);
        assert!(!result.attestation_installed);
        assert!(!result.pccs_configured);
        assert!(!result.reboot_required);
        assert!(result.target_kernel.is_none());
    }

    #[test]
    fn test_default_vm_readiness() {
        let readiness = TdxVmReadiness::default();
        assert!(!readiness.ready);
        assert!(!readiness.kvm_available);
        assert!(!readiness.qemu_installed);
        assert!(readiness.ovmf_path.is_none());
        assert!(readiness.errors.is_empty());
    }

    #[test]
    fn test_default_guest_quote_result() {
        let result = TdxGuestQuoteResult::default();
        assert!(!result.success);
        assert!(result.tdx_device.is_none());
        assert!(result.quote_size.is_none());
        assert!(result.error.is_none());
    }
}
