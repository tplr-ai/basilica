//! TDX Host Integration Tests
//!
//! Live integration tests for TDX host setup and validation.
//! These tests connect to real hardware and should only be run manually.
//!
//! # Running the tests
//!
//! Set environment variables and run with:
//! ```bash
//! TDX_TEST_HOST=151.185.43.58 \
//! TDX_TEST_USER=root \
//! TDX_TEST_KEY_PATH=~/.ssh/id_rsa \
//! cargo test --package basilica-validator --test tdx_host_integration_test -- --ignored --nocapture
//! ```
//!
//! Or use the justfile target:
//! ```bash
//! just test-tdx-host 151.185.43.58 root ~/.ssh/id_rsa
//! ```

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use basilica_common::ssh::SshConnectionDetails;
use basilica_validator::miner_prover::validation_tdx_host::{
    TdxGuestQuoteResult, TdxHostStatus, TdxHostValidator, TdxVmReadiness,
};
use basilica_validator::ssh::ValidatorSshClient;

/// Get test configuration from environment variables
fn get_test_config() -> Option<SshConnectionDetails> {
    let host = std::env::var("TDX_TEST_HOST").ok()?;
    let user = std::env::var("TDX_TEST_USER").unwrap_or_else(|_| "root".to_string());
    let key_path = std::env::var("TDX_TEST_KEY_PATH")
        .map(|p| {
            if p.starts_with('~') {
                let home = std::env::var("HOME").unwrap_or_else(|_| "/root".to_string());
                PathBuf::from(p.replacen('~', &home, 1))
            } else {
                PathBuf::from(p)
            }
        })
        .ok()?;
    let port = std::env::var("TDX_TEST_PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(22);

    Some(SshConnectionDetails {
        host,
        username: user,
        port,
        private_key_path: key_path,
        timeout: Duration::from_secs(60),
    })
}

/// Print TDX host status in a readable format
fn print_status(status: &TdxHostStatus) {
    println!("\n╔══════════════════════════════════════════╗");
    println!("║          TDX Host Status Report          ║");
    println!("╠══════════════════════════════════════════╣");
    println!(
        "║ TDX Initialized:     {:>18} ║",
        if status.tdx_initialized {
            "✓ YES"
        } else {
            "✗ NO"
        }
    );
    println!(
        "║ CPU TDX Capable:     {:>18} ║",
        if status.cpu_tdx_capable {
            "✓ YES"
        } else {
            "✗ NO"
        }
    );
    println!(
        "║ Intel Kernel:        {:>18} ║",
        if status.intel_kernel_installed {
            "✓ YES"
        } else {
            "✗ NO"
        }
    );
    println!(
        "║ BIOS TDX Enabled:    {:>18} ║",
        if status.bios_tdx_enabled {
            "✓ YES"
        } else {
            "✗ NO"
        }
    );
    println!(
        "║ PCCS Running:        {:>18} ║",
        if status.pccs_running {
            "✓ YES"
        } else {
            "✗ NO"
        }
    );
    println!(
        "║ QGS Running:         {:>18} ║",
        if status.qgs_running {
            "✓ YES"
        } else {
            "✗ NO"
        }
    );
    println!(
        "║ Kernel Version:      {:>18} ║",
        if status.kernel_version.is_empty() {
            "unknown"
        } else {
            &status.kernel_version
        }
    );
    println!(
        "║ Reboot Required:     {:>18} ║",
        if status.reboot_required {
            "⚠ YES"
        } else {
            "✓ NO"
        }
    );
    println!("╚══════════════════════════════════════════╝\n");
}

// =============================================================================
// LIVE TESTS - Only run with TDX_TEST_* environment variables set
// =============================================================================

/// Test: Check TDX host status on real hardware
#[tokio::test]
#[ignore = "requires TDX_TEST_HOST environment variable"]
async fn test_check_tdx_host_status() {
    let connection = get_test_config()
        .expect("Set TDX_TEST_HOST, TDX_TEST_USER, TDX_TEST_KEY_PATH environment variables");

    println!(
        "\n🔍 Connecting to {}@{}:{}",
        connection.username, connection.host, connection.port
    );

    let ssh_client = Arc::new(ValidatorSshClient::new());
    let validator = TdxHostValidator::new(ssh_client);

    // Check status
    let status = validator
        .check_status(&connection)
        .await
        .expect("Failed to check TDX host status");

    print_status(&status);

    // Print diagnostic info without failing - this test is for inspection
    if !status.cpu_tdx_capable {
        println!(
            "⚠️  CPU TDX capability not detected - this may be a detection issue or BIOS setting"
        );
        println!("   Check BIOS for TDX/TME settings");
    }

    if !status.tdx_initialized && status.cpu_tdx_capable {
        println!("ℹ️  TDX-capable CPU detected but TDX not initialized");
        println!("   This usually means Intel kernel is not installed or system needs reboot");
    }
}

/// Test: Check if TDX is already initialized
#[tokio::test]
#[ignore = "requires TDX_TEST_HOST environment variable"]
async fn test_tdx_already_initialized() {
    let connection = get_test_config()
        .expect("Set TDX_TEST_HOST, TDX_TEST_USER, TDX_TEST_KEY_PATH environment variables");

    println!(
        "\n🔍 Checking if TDX is already initialized on {}",
        connection.host
    );

    let ssh_client = Arc::new(ValidatorSshClient::new());
    let validator = TdxHostValidator::new(ssh_client);

    let status = validator
        .check_status(&connection)
        .await
        .expect("Failed to check TDX host status");

    if status.tdx_initialized {
        println!("✅ TDX is already initialized!");
        println!("   Kernel: {}", status.kernel_version);
        println!(
            "   PCCS: {}",
            if status.pccs_running {
                "running"
            } else {
                "not running"
            }
        );
        println!(
            "   QGS: {}",
            if status.qgs_running {
                "running"
            } else {
                "not running"
            }
        );
    } else {
        println!("⚠️  TDX is NOT initialized");
        println!(
            "   Intel kernel installed: {}",
            status.intel_kernel_installed
        );
        println!("   BIOS TDX enabled: {}", status.bios_tdx_enabled);
        if status.reboot_required {
            println!("   ⚡ Reboot required to complete initialization");
        }
    }
}

/// Test: Verify TDX host full setup status
#[tokio::test]
#[ignore = "requires TDX_TEST_HOST environment variable"]
async fn test_verify_tdx_host() {
    let connection = get_test_config()
        .expect("Set TDX_TEST_HOST, TDX_TEST_USER, TDX_TEST_KEY_PATH environment variables");

    println!("\n🔍 Verifying TDX host setup on {}", connection.host);

    let ssh_client = Arc::new(ValidatorSshClient::new());
    let validator = TdxHostValidator::new(ssh_client);

    let (success, errors, warnings) = validator
        .verify_host(&connection)
        .await
        .expect("Failed to verify TDX host");

    println!("\n📋 Verification Results:");
    println!(
        "   Status: {}",
        if success { "✅ PASSED" } else { "❌ FAILED" }
    );

    if !errors.is_empty() {
        println!("\n   Errors:");
        for err in &errors {
            println!("     ❌ {}", err);
        }
    }

    if !warnings.is_empty() {
        println!("\n   Warnings:");
        for warn in &warnings {
            println!("     ⚠️  {}", warn);
        }
    }
}

/// Test: Check if reboot is required
#[tokio::test]
#[ignore = "requires TDX_TEST_HOST environment variable"]
async fn test_check_reboot_required() {
    let connection = get_test_config()
        .expect("Set TDX_TEST_HOST, TDX_TEST_USER, TDX_TEST_KEY_PATH environment variables");

    println!("\n🔍 Checking reboot status on {}", connection.host);

    let ssh_client = Arc::new(ValidatorSshClient::new());
    let validator = TdxHostValidator::new(ssh_client);

    let reboot_required = validator
        .check_reboot_required(&connection)
        .await
        .expect("Failed to check reboot status");

    if reboot_required {
        println!("⚠️  Reboot is REQUIRED");
        println!("   Run: sudo reboot");
    } else {
        println!("✅ No reboot required");
    }
}

// =============================================================================
// SETUP TESTS - Only run when explicitly requested (modifies system!)
// =============================================================================

/// Test: Run full TDX host setup (CAUTION: modifies system!)
///
/// This test will:
/// 1. Install TDX packages (Intel kernel, QEMU, libvirt)
/// 2. Configure GRUB
/// 3. Install attestation packages
///
/// After this test, the system will need a REBOOT.
#[tokio::test]
#[ignore = "modifies system - run with TDX_RUN_SETUP=1"]
async fn test_run_tdx_host_setup() {
    // Extra safety check
    if std::env::var("TDX_RUN_SETUP").unwrap_or_default() != "1" {
        println!("⚠️  Skipping setup test - set TDX_RUN_SETUP=1 to run");
        return;
    }

    let connection = get_test_config()
        .expect("Set TDX_TEST_HOST, TDX_TEST_USER, TDX_TEST_KEY_PATH environment variables");

    println!("\n🚀 Running TDX host setup on {}", connection.host);
    println!("   ⚠️  This will modify the system!");

    let ssh_client = Arc::new(ValidatorSshClient::new());
    let validator = TdxHostValidator::new(ssh_client);

    // Check initial status
    let initial_status = validator.check_status(&connection).await.unwrap();
    println!("\n📋 Initial Status:");
    print_status(&initial_status);

    // Run setup
    println!("🔧 Running setup...");
    let result = validator
        .setup_tdx_host(&connection)
        .await
        .expect("TDX host setup failed");

    println!("\n📋 Setup Results:");
    println!("   Packages installed: {}", result.packages_installed);
    println!("   GRUB configured: {}", result.grub_configured);
    println!("   Attestation installed: {}", result.attestation_installed);
    println!("   Reboot required: {}", result.reboot_required);
    if let Some(kernel) = &result.target_kernel {
        println!("   Target kernel: {}", kernel);
    }

    // Check final status
    let final_status = validator.check_status(&connection).await.unwrap();
    println!("\n📋 Final Status:");
    print_status(&final_status);

    if result.reboot_required {
        println!("\n⚡ REBOOT REQUIRED!");
        println!(
            "   Run: ssh {}@{} 'sudo reboot'",
            connection.username, connection.host
        );
    }
}

/// Test: Full TDX setup with automatic reboot (CAUTION: reboots node!)
///
/// This test will:
/// 1. Install TDX packages
/// 2. Automatically reboot the node
/// 3. Wait for it to come back (up to 5 minutes)
/// 4. Verify TDX is initialized
/// 5. Configure PCCS if API key provided
#[tokio::test]
#[ignore = "reboots node - run with TDX_RUN_SETUP=1 and TDX_AUTO_REBOOT=1"]
async fn test_full_tdx_setup_with_reboot() {
    // Safety checks
    if std::env::var("TDX_RUN_SETUP").unwrap_or_default() != "1" {
        println!("⚠️  Skipping - set TDX_RUN_SETUP=1 to run");
        return;
    }
    if std::env::var("TDX_AUTO_REBOOT").unwrap_or_default() != "1" {
        println!("⚠️  Skipping - set TDX_AUTO_REBOOT=1 to allow automatic reboot");
        return;
    }

    let connection = get_test_config()
        .expect("Set TDX_TEST_HOST, TDX_TEST_USER, TDX_TEST_KEY_PATH environment variables");
    let intel_api_key = std::env::var("INTEL_API_KEY").ok();

    println!("\n╔═══════════════════════════════════════════════════════════════╗");
    println!("║     Full TDX Setup with Auto-Reboot                          ║");
    println!("║     Target: {:43}    ║", connection.host);
    println!("╚═══════════════════════════════════════════════════════════════╝");

    if intel_api_key.is_none() {
        println!("⚠️  INTEL_API_KEY not set - PCCS won't be configured");
    }

    let ssh_client = Arc::new(ValidatorSshClient::new());
    let validator = TdxHostValidator::new(ssh_client);

    println!("\n📍 Starting full TDX setup (this may take several minutes)...\n");

    let final_status = validator
        .setup_tdx_host_full(
            &connection,
            intel_api_key.as_deref(),
            Some(Duration::from_secs(600)), // 10 minute timeout for reboot
        )
        .await
        .expect("Full TDX setup failed");

    print_status(&final_status);

    if final_status.tdx_initialized {
        println!("✅ SUCCESS: TDX is initialized and ready!");
    } else {
        println!("❌ TDX not initialized - check BIOS settings");
        println!("   BIOS must have:");
        println!("   - Intel TME (Total Memory Encryption) enabled");
        println!("   - Intel TDX enabled");
        println!("   - MKTME keys allocated for TDX");
    }

    // Print ready status
    let ready = validator.is_tdx_ready(&connection).await.unwrap_or(false);
    println!("\n   TDX Ready: {}", if ready { "✅ YES" } else { "❌ NO" });
}

/// Test: Configure PCCS with Intel API key (requires API key!)
#[tokio::test]
#[ignore = "requires INTEL_API_KEY environment variable"]
async fn test_configure_pccs() {
    let connection = get_test_config()
        .expect("Set TDX_TEST_HOST, TDX_TEST_USER, TDX_TEST_KEY_PATH environment variables");
    let intel_api_key = std::env::var("INTEL_API_KEY").expect(
        "Set INTEL_API_KEY environment variable (get from https://api.portal.trustedservices.intel.com/)",
    );

    println!("\n🔧 Configuring PCCS on {}", connection.host);

    let ssh_client = Arc::new(ValidatorSshClient::new());
    let validator = TdxHostValidator::new(ssh_client);

    let success = validator
        .configure_pccs(&connection, &intel_api_key)
        .await
        .expect("Failed to configure PCCS");

    if success {
        println!("✅ PCCS configured and running!");
    } else {
        println!("❌ PCCS configuration failed");
    }

    // Also try to register platform
    println!("\n🔧 Registering platform with Intel...");
    let registered = validator
        .register_platform(&connection)
        .await
        .unwrap_or(false);
    if registered {
        println!("✅ Platform registered with Intel PCCS");
    } else {
        println!("⚠️  Platform registration may have issues (check manually)");
    }
}

/// Test: Just reboot and wait (for testing reboot logic)
#[tokio::test]
#[ignore = "reboots node - run with TDX_AUTO_REBOOT=1"]
async fn test_reboot_and_wait() {
    if std::env::var("TDX_AUTO_REBOOT").unwrap_or_default() != "1" {
        println!("⚠️  Skipping - set TDX_AUTO_REBOOT=1 to allow reboot");
        return;
    }

    let connection = get_test_config()
        .expect("Set TDX_TEST_HOST, TDX_TEST_USER, TDX_TEST_KEY_PATH environment variables");

    println!(
        "\n🔄 Rebooting {} and waiting for it to come back...",
        connection.host
    );

    let ssh_client = Arc::new(ValidatorSshClient::new());
    let validator = TdxHostValidator::new(ssh_client);

    // Get pre-reboot status
    let pre_status = validator.check_status(&connection).await.unwrap();
    println!("   Pre-reboot kernel: {}", pre_status.kernel_version);

    // Reboot and wait
    let new_kernel = validator
        .reboot_and_wait(&connection, Some(Duration::from_secs(300)))
        .await
        .expect("Reboot failed or timed out");

    println!("   ✅ Node back online!");
    println!("   Post-reboot kernel: {}", new_kernel);

    // Check status after reboot
    let post_status = validator.check_status(&connection).await.unwrap();
    print_status(&post_status);
}

// =============================================================================
// COMBINED TESTS
// =============================================================================

/// Test: Full integration flow - check status, setup if needed, verify
#[tokio::test]
#[ignore = "full integration test - run manually"]
async fn test_full_tdx_host_integration() {
    let connection = get_test_config()
        .expect("Set TDX_TEST_HOST, TDX_TEST_USER, TDX_TEST_KEY_PATH environment variables");

    println!("\n╔═══════════════════════════════════════════════════════════════╗");
    println!("║          TDX Host Full Integration Test                       ║");
    println!("║          Target: {:43} ║", connection.host);
    println!("╚═══════════════════════════════════════════════════════════════╝");

    let ssh_client = Arc::new(ValidatorSshClient::new());
    let validator = TdxHostValidator::new(ssh_client);

    // Step 1: Check status
    println!("\n📍 Step 1: Checking TDX host status...");
    let status = validator.check_status(&connection).await.unwrap();
    print_status(&status);

    // Step 2: Verify
    println!("📍 Step 2: Verifying TDX host setup...");
    let (success, errors, warnings) = validator.verify_host(&connection).await.unwrap();

    if success {
        println!("   ✅ TDX host is ready!");
    } else {
        println!("   ❌ TDX host has issues:");
        for err in errors {
            println!("      - {}", err);
        }
    }

    if !warnings.is_empty() {
        println!("   ⚠️  Warnings:");
        for warn in warnings {
            println!("      - {}", warn);
        }
    }

    // Step 3: Summary
    println!("\n📍 Summary:");
    println!(
        "   TDX Initialized: {}",
        if status.tdx_initialized { "✅" } else { "❌" }
    );
    println!(
        "   Intel Kernel: {}",
        if status.intel_kernel_installed {
            "✅"
        } else {
            "❌"
        }
    );
    println!(
        "   PCCS Running: {}",
        if status.pccs_running { "✅" } else { "⚠️" }
    );
    println!(
        "   QGS Running: {}",
        if status.qgs_running { "✅" } else { "⚠️" }
    );

    if status.reboot_required {
        println!("\n   ⚡ Reboot required to complete setup");
    }

    // Step 4: Ready check
    let ready = validator.is_tdx_ready(&connection).await.unwrap_or(false);
    println!(
        "\n   TDX Ready: {}",
        if ready {
            "✅ YES - can run TDX VMs"
        } else {
            "❌ NO"
        }
    );
}

// =============================================================================
// TDX GUEST VM TESTS
// =============================================================================

/// Print VM readiness status in a readable format
fn print_vm_readiness(readiness: &TdxVmReadiness) {
    println!("\n╔══════════════════════════════════════════╗");
    println!("║        TDX VM Readiness Report           ║");
    println!("╠══════════════════════════════════════════╣");
    println!(
        "║ Overall Ready:       {:>18} ║",
        if readiness.ready { "✓ YES" } else { "✗ NO" }
    );
    println!(
        "║ KVM Available:       {:>18} ║",
        if readiness.kvm_available {
            "✓ YES"
        } else {
            "✗ NO"
        }
    );
    println!(
        "║ QEMU Installed:      {:>18} ║",
        if readiness.qemu_installed {
            "✓ YES"
        } else {
            "✗ NO"
        }
    );
    println!(
        "║ QEMU TDX Support:    {:>18} ║",
        if readiness.qemu_tdx_support {
            "✓ YES"
        } else {
            "✗ NO"
        }
    );
    println!(
        "║ virsh Installed:     {:>18} ║",
        if readiness.virsh_installed {
            "✓ YES"
        } else {
            "✗ NO"
        }
    );
    println!(
        "║ libvirt TDX:         {:>18} ║",
        if readiness.libvirt_tdx_support {
            "✓ YES"
        } else {
            "⚠ NO"
        }
    );
    println!(
        "║ OVMF Available:      {:>18} ║",
        if readiness.ovmf_available {
            "✓ YES"
        } else {
            "✗ NO"
        }
    );
    println!(
        "║ libvirtd Running:    {:>18} ║",
        if readiness.libvirtd_running {
            "✓ YES"
        } else {
            "⚠ NO"
        }
    );
    println!("╚══════════════════════════════════════════╝");

    if let Some(path) = &readiness.ovmf_path {
        println!("   OVMF Path: {}", path);
    }

    if !readiness.errors.is_empty() {
        println!("\n   Errors:");
        for err in &readiness.errors {
            println!("     ❌ {}", err);
        }
    }

    if !readiness.warnings.is_empty() {
        println!("\n   Warnings:");
        for warn in &readiness.warnings {
            println!("     ⚠️  {}", warn);
        }
    }
}

/// Print guest quote result
fn print_quote_result(result: &TdxGuestQuoteResult) {
    println!("\n╔══════════════════════════════════════════╗");
    println!("║      TDX Guest Quote Generation          ║");
    println!("╠══════════════════════════════════════════╣");
    println!(
        "║ Success:             {:>18} ║",
        if result.success { "✓ YES" } else { "✗ NO" }
    );
    if let Some(dev) = &result.tdx_device {
        println!("║ TDX Device:          {:>18} ║", dev);
    }
    if let Some(size) = result.quote_size {
        println!("║ Quote Size:          {:>14} bytes ║", size);
    }
    if let Some(method) = &result.method {
        println!("║ Method:              {:>18} ║", method);
    }
    println!("╚══════════════════════════════════════════╝");

    if let Some(preview) = &result.quote_preview {
        println!("\n   Quote Preview (first 100 chars):");
        println!("   {}", &preview[..preview.len().min(100)]);
    }

    if let Some(err) = &result.error {
        println!("\n   ❌ Error: {}", err);
    }
}

/// Test: Check if TDX host is ready to run TDX VMs
#[tokio::test]
#[ignore = "requires TDX_TEST_HOST environment variable"]
async fn test_check_vm_readiness() {
    let connection = get_test_config()
        .expect("Set TDX_TEST_HOST, TDX_TEST_USER, TDX_TEST_KEY_PATH environment variables");

    println!("\n🔍 Checking TDX VM readiness on {}", connection.host);

    let ssh_client = Arc::new(ValidatorSshClient::new());
    let validator = TdxHostValidator::new(ssh_client);

    let readiness = validator
        .check_vm_readiness(&connection)
        .await
        .expect("Failed to check VM readiness");

    print_vm_readiness(&readiness);

    if readiness.ready {
        println!("\n✅ Host is ready to launch TDX VMs!");
    } else {
        println!("\n❌ Host is NOT ready for TDX VMs");
        if !readiness.errors.is_empty() {
            println!("   Fix the errors listed above");
        }
    }
}

/// Test: Download TDX guest image
#[tokio::test]
#[ignore = "requires TDX_TEST_HOST, downloads ~700MB image"]
async fn test_download_guest_image() {
    let connection = get_test_config()
        .expect("Set TDX_TEST_HOST, TDX_TEST_USER, TDX_TEST_KEY_PATH environment variables");

    println!("\n📥 Downloading TDX guest image on {}", connection.host);
    println!("   This may take a few minutes for ~700MB download...");

    let ssh_client = Arc::new(ValidatorSshClient::new());
    let validator = TdxHostValidator::new(ssh_client);

    let path = validator
        .download_guest_image(&connection)
        .await
        .expect("Failed to download guest image");

    println!("✅ Guest image ready at: {}", path);
}

/// Test: Full TDX guest VM quote generation test
///
/// This test will:
/// 1. Check VM readiness
/// 2. Download guest image (if needed)
/// 3. Create a TDX VM
/// 4. Wait for VM to boot
/// 5. Test quote generation inside VM
/// 6. Cleanup
#[tokio::test]
#[ignore = "creates TDX VM - run with TDX_RUN_VM_TEST=1"]
async fn test_tdx_guest_vm_quote_generation() {
    if std::env::var("TDX_RUN_VM_TEST").unwrap_or_default() != "1" {
        println!("⚠️  Skipping - set TDX_RUN_VM_TEST=1 to run TDX VM test");
        return;
    }

    let connection = get_test_config()
        .expect("Set TDX_TEST_HOST, TDX_TEST_USER, TDX_TEST_KEY_PATH environment variables");

    println!("\n╔═══════════════════════════════════════════════════════════════╗");
    println!("║     TDX Guest VM Quote Generation Test                        ║");
    println!("║     Target: {:43}    ║", connection.host);
    println!("╚═══════════════════════════════════════════════════════════════╝");

    let ssh_client = Arc::new(ValidatorSshClient::new());
    let validator = TdxHostValidator::new(ssh_client);

    // Step 1: Check VM readiness
    println!("\n📍 Step 1: Checking VM readiness...");
    let readiness = validator.check_vm_readiness(&connection).await.unwrap();
    print_vm_readiness(&readiness);

    if !readiness.ready {
        println!("\n❌ Host is not ready for TDX VMs. Aborting test.");
        return;
    }

    // Step 2: Run full guest test
    println!("\n📍 Step 2: Running full TDX guest test...");
    println!("   This will create a TDX VM, test quote generation, and cleanup.");
    println!("   This may take 5-10 minutes...\n");

    let result = validator
        .run_full_guest_test(&connection)
        .await
        .expect("Failed to run guest test");

    print_quote_result(&result);

    // Summary
    println!("\n═══════════════════════════════════════════════════════════════");
    if result.success {
        println!("✅ TDX GUEST QUOTE GENERATION: SUCCESS");
        println!("   Quote size: {} bytes", result.quote_size.unwrap_or(0));
        println!("   TDX attestation is working end-to-end!");
    } else if result.tdx_device.is_some() {
        println!("⚠️  TDX GUEST QUOTE GENERATION: PARTIAL SUCCESS");
        println!("   TDX device found in guest but quote generation failed");
        println!("   This may require additional attestation tools");
    } else {
        println!("❌ TDX GUEST QUOTE GENERATION: FAILED");
        println!("   No TDX device found in guest");
        println!("   Possible causes:");
        println!("   - QEMU not using TDX properly");
        println!("   - Kernel in guest doesn't support TDX");
        println!("   - TDX module not initialized on host");
    }
}

/// Test: Create and cleanup a test TDX VM manually (step by step)
#[tokio::test]
#[ignore = "creates TDX VM - run with TDX_RUN_VM_TEST=1"]
async fn test_create_cleanup_tdx_vm() {
    if std::env::var("TDX_RUN_VM_TEST").unwrap_or_default() != "1" {
        println!("⚠️  Skipping - set TDX_RUN_VM_TEST=1 to run");
        return;
    }

    let connection = get_test_config()
        .expect("Set TDX_TEST_HOST, TDX_TEST_USER, TDX_TEST_KEY_PATH environment variables");

    let vm_name = format!("tdx-test-{}", std::process::id());
    println!("\n🔧 Creating TDX VM: {}", vm_name);

    let ssh_client = Arc::new(ValidatorSshClient::new());
    let validator = TdxHostValidator::new(ssh_client);

    // Create VM
    println!("\n📍 Creating VM...");
    let ssh_port = match validator.create_test_tdx_vm(&connection, &vm_name).await {
        Ok(port) => {
            println!("   ✅ VM created, SSH port: {}", port);
            port
        }
        Err(e) => {
            println!("   ❌ Failed to create VM: {}", e);
            // Try cleanup anyway
            let _ = validator.cleanup_test_vm(&connection, &vm_name).await;
            panic!("VM creation failed");
        }
    };

    // Wait for VM
    println!("\n📍 Waiting for VM to boot (up to 5 minutes)...");
    match validator
        .wait_for_vm(&connection, ssh_port, Duration::from_secs(300))
        .await
    {
        Ok(()) => println!("   ✅ VM is ready!"),
        Err(e) => {
            println!("   ❌ VM did not become ready: {}", e);
            let _ = validator.cleanup_test_vm(&connection, &vm_name).await;
            panic!("VM boot timeout");
        }
    }

    // Test quote generation
    println!("\n📍 Testing quote generation...");
    let result = validator
        .test_guest_quote_generation(&connection, ssh_port)
        .await
        .unwrap_or_default();
    print_quote_result(&result);

    // Cleanup
    println!("\n📍 Cleaning up VM...");
    validator
        .cleanup_test_vm(&connection, &vm_name)
        .await
        .expect("Failed to cleanup VM");
    println!("   ✅ VM cleaned up");

    if result.success {
        println!("\n✅ Test PASSED: TDX quote generation works!");
    } else {
        println!("\n⚠️  Test completed but quote generation failed");
    }
}

/// Test: Full two-phase TDX attestation with Intel kernel
/// This test:
/// 1. Creates a TDX VM with generic kernel
/// 2. Installs Intel kernel inside the VM
/// 3. Destroys and recreates VM (TDX VMs can't reboot)
/// 4. Generates TDX Report from /dev/tdx_guest
#[tokio::test]
#[ignore = "requires TDX_RUN_VM_TEST=1, takes 10-15 minutes"]
async fn test_two_phase_attestation() {
    if std::env::var("TDX_RUN_VM_TEST").unwrap_or_default() != "1" {
        println!("⚠️  Skipping - set TDX_RUN_VM_TEST=1 to run");
        return;
    }

    let connection = get_test_config()
        .expect("Set TDX_TEST_HOST, TDX_TEST_USER, TDX_TEST_KEY_PATH environment variables");

    println!("\n╔═══════════════════════════════════════════════════════════════╗");
    println!("║     TDX Two-Phase Attestation Test                            ║");
    println!("║     Target: {:50} ║", connection.host);
    println!("╚═══════════════════════════════════════════════════════════════╝");

    let ssh_client = Arc::new(ValidatorSshClient::new());
    let _validator = TdxHostValidator::new(ssh_client.clone());

    let vm_name = format!("tdx-attest-{}", std::process::id());
    let ssh_port = 2230;

    // Phase 1: Create VM with generic kernel
    println!("\n📍 Phase 1: Creating TDX VM with generic kernel...");

    let phase1_script = format!(
        r#"
        set -e
        cd /var/lib/basilica-tdx-test
        VM_NAME="{vm_name}"
        SSH_PORT={ssh_port}
        
        # Cleanup any existing
        rm -f ${{VM_NAME}}* 2>/dev/null || true
        rm -rf ${{VM_NAME}}-seed 2>/dev/null || true
        
        # Create disk
        qemu-img create -f qcow2 -F qcow2 -b ubuntu-noble-cloudimg.img ${{VM_NAME}}.qcow2 20G >/dev/null
        
        # Create cloud-init
        mkdir -p ${{VM_NAME}}-seed
        cat > ${{VM_NAME}}-seed/meta-data << EOF
instance-id: ${{VM_NAME}}
local-hostname: ${{VM_NAME}}
EOF
        cat > ${{VM_NAME}}-seed/user-data << EOF
#cloud-config
users:
  - name: ubuntu
    sudo: ALL=(ALL) NOPASSWD:ALL
    ssh_authorized_keys:
      - $(cat /root/.ssh/id_rsa.pub 2>/dev/null || cat ~/.ssh/id_rsa.pub)
package_update: false
EOF
        cloud-localds ${{VM_NAME}}-seed.iso ${{VM_NAME}}-seed/user-data ${{VM_NAME}}-seed/meta-data
        
        # Start VM
        nohup qemu-system-x86_64 \
            -name "$VM_NAME" \
            -enable-kvm \
            -machine q35,kernel-irqchip=split,confidential-guest-support=tdx0 \
            -object tdx-guest,id=tdx0 \
            -cpu host \
            -m 4096 \
            -smp 2 \
            -bios /usr/share/ovmf/OVMF.fd \
            -drive file="${{VM_NAME}}.qcow2",if=virtio,format=qcow2 \
            -drive file="${{VM_NAME}}-seed.iso",if=virtio,format=raw,readonly=on \
            -netdev user,id=net0,hostfwd=tcp::${{SSH_PORT}}-:22 \
            -device virtio-net-pci,netdev=net0 \
            -nographic \
            -serial none \
            -pidfile "${{VM_NAME}}.pid" \
            > ${{VM_NAME}}.log 2>&1 &
        
        sleep 3
        cat ${{VM_NAME}}.pid
    "#
    );

    let pid = ssh_client
        .execute_command(&connection, &phase1_script, true)
        .await
        .expect("Failed to create Phase 1 VM");
    println!("   ✅ VM started (PID: {})", pid.trim());

    // Wait for VM to be accessible
    println!("\n📍 Waiting for VM to boot...");
    tokio::time::sleep(tokio::time::Duration::from_secs(90)).await;

    // Install Intel kernel
    println!("\n📍 Installing Intel kernel (this may take a few minutes)...");
    let install_script = format!(
        r#"
        ssh -i /root/.ssh/id_rsa -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
            -o ConnectTimeout=30 -p {ssh_port} ubuntu@localhost \
            "sudo add-apt-repository -y ppa:kobuk-team/tdx-release && \
             sudo apt-get update -qq && \
             sudo apt-get install -y linux-image-intel 2>&1 | tail -5"
    "#
    );

    let install_result = ssh_client
        .execute_command(&connection, &install_script, true)
        .await
        .expect("Failed to install Intel kernel");
    println!("   {}", install_result.lines().last().unwrap_or("Done"));

    // Phase 2: Destroy and recreate VM
    println!("\n📍 Phase 2: Destroying VM and recreating with Intel kernel...");
    let phase2_script = format!(
        r#"
        set -e
        cd /var/lib/basilica-tdx-test
        VM_NAME="{vm_name}"
        SSH_PORT={ssh_port}
        
        # Kill old VM
        kill $(cat ${{VM_NAME}}.pid) 2>/dev/null || true
        sleep 2
        rm -f ${{VM_NAME}}.pid ${{VM_NAME}}.log
        
        # Restart with vsock for quote generation
        nohup qemu-system-x86_64 \
            -name "$VM_NAME" \
            -enable-kvm \
            -machine q35,kernel-irqchip=split,confidential-guest-support=tdx0 \
            -object tdx-guest,id=tdx0 \
            -cpu host \
            -m 4096 \
            -smp 2 \
            -bios /usr/share/ovmf/OVMF.fd \
            -drive file="${{VM_NAME}}.qcow2",if=virtio,format=qcow2 \
            -drive file="${{VM_NAME}}-seed.iso",if=virtio,format=raw,readonly=on \
            -netdev user,id=net0,hostfwd=tcp::${{SSH_PORT}}-:22 \
            -device virtio-net-pci,netdev=net0 \
            -device vhost-vsock-pci,guest-cid=3 \
            -nographic \
            -serial none \
            -pidfile "${{VM_NAME}}.pid" \
            > ${{VM_NAME}}.log 2>&1 &
        
        sleep 3
        cat ${{VM_NAME}}.pid
    "#
    );

    let pid2 = ssh_client
        .execute_command(&connection, &phase2_script, true)
        .await
        .expect("Failed to create Phase 2 VM");
    println!("   ✅ VM restarted (PID: {})", pid2.trim());

    // Wait for VM with Intel kernel
    println!("\n📍 Waiting for VM to boot with Intel kernel...");
    tokio::time::sleep(tokio::time::Duration::from_secs(90)).await;

    // Verify kernel and get TDX report
    println!("\n📍 Verifying Intel kernel and generating TDX Report...");
    let report_script = format!(
        r#"
        ssh -i /root/.ssh/id_rsa -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
            -o ConnectTimeout=30 -p {ssh_port} ubuntu@localhost "
            echo 'KERNEL:' \$(uname -r)
            echo 'TDX_DEVICE:' \$(ls /dev/tdx_guest 2>/dev/null || echo 'NOT_FOUND')
            echo 'VSOCK:' \$(ls /dev/vsock 2>/dev/null || echo 'NOT_FOUND')
            
            # Generate TDX Report
            sudo python3 << 'PYEOF'
import os, fcntl, base64

TDX_CMD_GET_REPORT0 = 0xc4405401
buf = bytearray(64 + 1024)

try:
    fd = os.open('/dev/tdx_guest', os.O_RDWR)
    fcntl.ioctl(fd, TDX_CMD_GET_REPORT0, buf)
    os.close(fd)
    report = buf[64:]
    print('TDX_REPORT_OK')
    print(f'REPORT_SIZE: {{len(report)}}')
    print(f'REPORT_VERSION: {{report[0]}}')
    print(f'REPORT_BASE64: {{base64.b64encode(report[:256]).decode()}}')
except Exception as e:
    print(f'TDX_REPORT_ERROR: {{e}}')
PYEOF
        "
    "#
    );

    let report_result = ssh_client
        .execute_command(&connection, &report_script, true)
        .await
        .expect("Failed to generate TDX report");

    println!("\n╔══════════════════════════════════════════╗");
    println!("║      TDX Attestation Result              ║");
    println!("╠══════════════════════════════════════════╣");

    let mut success = false;
    for line in report_result.lines() {
        if line.starts_with("KERNEL:") {
            let kernel = line.trim_start_matches("KERNEL:").trim();
            let is_intel = kernel.contains("intel");
            println!("║ Kernel: {:>28} ║", kernel);
            println!(
                "║ Intel Kernel: {:>22} ║",
                if is_intel { "✓ YES" } else { "✗ NO" }
            );
        } else if line.starts_with("TDX_DEVICE:") {
            let dev = line.trim_start_matches("TDX_DEVICE:").trim();
            println!("║ TDX Device: {:>24} ║", dev);
        } else if line.starts_with("TDX_REPORT_OK") {
            success = true;
            println!("║ Report Generated: {:>18} ║", "✓ YES");
        } else if line.starts_with("REPORT_SIZE:") {
            let size = line.trim_start_matches("REPORT_SIZE:").trim();
            println!("║ Report Size: {:>19} bytes ║", size);
        } else if line.starts_with("REPORT_VERSION:") {
            let ver = line.trim_start_matches("REPORT_VERSION:").trim();
            println!("║ Report Version: {:>20} ║", ver);
        } else if line.starts_with("REPORT_BASE64:") {
            let b64 = line.trim_start_matches("REPORT_BASE64:").trim();
            println!("╚══════════════════════════════════════════╝");
            println!("\n📄 TDX Report (Base64, first 100 chars):");
            println!("   {}...", &b64[..b64.len().min(100)]);
        } else if line.starts_with("TDX_REPORT_ERROR:") {
            println!("║ Report Generated: {:>18} ║", "✗ NO");
            println!("╚══════════════════════════════════════════╝");
            println!(
                "\n❌ Error: {}",
                line.trim_start_matches("TDX_REPORT_ERROR:").trim()
            );
        }
    }

    // Cleanup
    println!("\n📍 Cleaning up...");
    let cleanup_script = format!(
        r#"
        cd /var/lib/basilica-tdx-test
        VM_NAME="{vm_name}"
        kill $(cat ${{VM_NAME}}.pid) 2>/dev/null || true
        rm -f ${{VM_NAME}}* 2>/dev/null || true
        rm -rf ${{VM_NAME}}-seed 2>/dev/null || true
        echo "Cleaned up"
    "#
    );
    let _ = ssh_client
        .execute_command(&connection, &cleanup_script, true)
        .await;
    println!("   ✅ Cleanup complete");

    if success {
        println!("\n✅ SUCCESS: TDX attestation report generated!");
    } else {
        println!("\n❌ FAILED: Could not generate TDX attestation report");
    }

    assert!(success, "TDX attestation report generation failed");
}
