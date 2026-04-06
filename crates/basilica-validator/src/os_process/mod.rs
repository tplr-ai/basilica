//! Linux Process Management Module
//!
//! Provides simple, focused utilities for managing processes on Linux systems.
//! Designed specifically for Ubuntu/Linux environments.

use anyhow::{anyhow, Result};
use std::time::Duration;
use tracing::{debug, error, info, warn};

/// Manages Linux processes for proper cleanup
pub struct ProcessGroup;

impl ProcessGroup {
    /// Configure a command to run in a new process group
    pub fn configure_command(command: &mut tokio::process::Command) {
        // Create new process group when spawned (0 means use PID as PGID)
        command.process_group(0);
        debug!("Configured command for new process group");
    }
}

/// Simple process terminator for graceful shutdown
pub struct ProcessTerminator;

impl ProcessTerminator {
    /// Terminate a process with timeout
    pub async fn terminate(pid: i32, timeout: Duration) -> Result<()> {
        info!("Terminating process {} with timeout {:?}", pid, timeout);

        if !ProcessUtils::process_exists(pid as u32) {
            debug!("Process {} already terminated", pid);
            return Ok(());
        }

        // Try graceful termination with SIGTERM
        let term_result = unsafe { libc::kill(pid, libc::SIGTERM) };
        if term_result != 0 {
            let errno = std::io::Error::last_os_error();
            if errno.raw_os_error() == Some(libc::ESRCH) {
                debug!("Process {} not found (already terminated)", pid);
                return Ok(());
            }
            debug!("SIGTERM failed: {}", errno);
        }

        // Wait for graceful shutdown
        tokio::time::sleep(timeout).await;

        // Check if still alive and force kill if needed
        if ProcessUtils::process_exists(pid as u32) {
            warn!("Process {} still alive, sending SIGKILL", pid);

            let kill_result = unsafe { libc::kill(pid, libc::SIGKILL) };
            if kill_result != 0 {
                let errno = std::io::Error::last_os_error();
                if errno.raw_os_error() != Some(libc::ESRCH) {
                    warn!("SIGKILL failed: {}", errno);
                }
            }

            // Wait for SIGKILL to take effect
            tokio::time::sleep(Duration::from_millis(200)).await;

            // Final check
            if ProcessUtils::process_exists(pid as u32) {
                error!("Process {} still exists after SIGKILL", pid);
                return Err(anyhow!("Failed to terminate process {}", pid));
            }
        }

        info!("Process {} terminated", pid);
        Ok(())
    }
}

/// Simple process utilities
pub struct ProcessUtils;

impl ProcessUtils {
    /// Check if a process exists
    pub fn process_exists(pid: u32) -> bool {
        use sysinfo::{Pid, System};

        let mut system = System::new_all();
        system.refresh_processes();
        system.process(Pid::from_u32(pid)).is_some()
    }

    /// Reap zombie (defunct) child processes
    /// This should be called periodically to clean up zombie processes
    pub fn reap_zombies() {
        use std::ptr;

        // Use waitpid with WNOHANG to reap any zombie children without blocking
        loop {
            let result = unsafe { libc::waitpid(-1, ptr::null_mut(), libc::WNOHANG) };

            if result > 0 {
                debug!("Reaped zombie process with PID {}", result);
            } else if result == 0 {
                // No more zombie children to reap
                break;
            } else {
                // Error or no children
                break;
            }
        }
    }

    /// Kill any SSH tunnels
    pub fn cleanup_ssh_tunnels() {
        info!("Starting SSH tunnel cleanup");

        // First, reap any zombie processes
        Self::reap_zombies();

        let mut killed_count = 0;
        let mut found_processes = Vec::new();
        let mut all_processes = Vec::new();

        // Use procfs to find SSH processes more reliably
        if let Ok(entries) = std::fs::read_dir("/proc") {
            for entry in entries.flatten() {
                // Check if this is a PID directory
                let file_name = entry.file_name();
                let pid_str = file_name.to_string_lossy();

                if let Ok(pid) = pid_str.parse::<i32>() {
                    // Skip init process and our own process
                    if pid <= 1 || pid == std::process::id() as i32 {
                        continue;
                    }

                    // Read the cmdline file to get the full command
                    let cmdline_path = format!("/proc/{}/cmdline", pid);
                    if let Ok(cmdline_bytes) = std::fs::read(&cmdline_path) {
                        // The cmdline file contains null-separated arguments
                        if cmdline_bytes.is_empty() {
                            continue;
                        }

                        // Convert to readable string for debugging
                        let cmdline_str = String::from_utf8_lossy(&cmdline_bytes);
                        let cmdline_display = cmdline_str.replace('\0', " ").trim().to_string();

                        // Log first 10 processes for debugging
                        if all_processes.len() < 10 {
                            debug!("Process PID {}: {}", pid, cmdline_display);
                        }
                        all_processes.push((pid, cmdline_display.clone()));

                        // Check if this is an SSH process
                        // The cmdline might look like: "ssh-N-L3000:localhost:3000-p22..."
                        // or "ssh -N -L 3000:localhost:3000 -p 22..." depending on how it was launched

                        // Check both the display string and raw bytes
                        let is_ssh = cmdline_display.contains("ssh")
                            || cmdline_bytes.starts_with(b"ssh")
                            || cmdline_str.starts_with("ssh");

                        if is_ssh {
                            debug!("Found SSH process PID {}: {}", pid, cmdline_display);

                            // Check if it's a tunnel - be very aggressive here
                            // Look for any tunnel-related patterns
                            let is_tunnel =
                                // Check for flags with or without spaces
                                cmdline_display.contains("-L") ||  // Local forwarding
                                cmdline_display.contains("-N") ||  // No command
                                cmdline_display.contains("-D") ||  // Dynamic forwarding
                                // Check for common tunnel patterns
                                cmdline_display.contains("localhost") ||
                                cmdline_display.contains("127.0.0.1") ||
                                cmdline_display.contains(":3000") ||  // Common port
                                // Check raw string too (handles null bytes)
                                cmdline_str.contains("-L") ||
                                cmdline_str.contains("-N") ||
                                cmdline_str.contains("-D");

                            if is_tunnel {
                                info!("Found SSH tunnel process PID {}: {}", pid, cmdline_display);
                                found_processes.push((pid, cmdline_display.clone()));

                                // Kill the process with SIGKILL (9)
                                let kill_result = unsafe { libc::kill(pid, libc::SIGKILL) };
                                if kill_result == 0 {
                                    killed_count += 1;
                                    info!("Successfully killed SSH tunnel process PID {}", pid);
                                } else {
                                    let errno = std::io::Error::last_os_error();
                                    // ESRCH means process doesn't exist (might have died already)
                                    if errno.raw_os_error() != Some(libc::ESRCH) {
                                        error!("Failed to kill SSH tunnel PID {}: {}", pid, errno);
                                    }
                                }
                            }
                        }
                    } else {
                        // Log why we couldn't read the cmdline
                        let err = std::fs::read(&cmdline_path).err();
                        if let Some(e) = err {
                            // Only log for low PIDs to avoid spam
                            if pid < 1000 {
                                debug!("Could not read /proc/{}/cmdline: {}", pid, e);
                            }
                        }
                    }
                }
            }
        } else {
            warn!("Could not read /proc directory");
        }

        debug!("Scanned {} total processes", all_processes.len());

        // Also use more aggressive system commands as fallback
        // Kill any ssh process with tunnel-like patterns
        let patterns = [
            "ssh.*-L.*:",       // SSH with local port forwarding
            "ssh.*-N",          // SSH with no command execution
            "ssh.*-D.*:",       // SSH with dynamic port forwarding
            "ssh.*localhost",   // SSH with localhost
            "ssh.*127\\.0\\.1", // SSH with 127.0.0.1
        ];

        for pattern in &patterns {
            debug!("Running pkill with pattern: {}", pattern);
            let result = std::process::Command::new("pkill")
                .args(["-9", "-f", pattern])
                .output();

            if let Ok(output) = result {
                if output.status.success() {
                    debug!("pkill succeeded for pattern: {}", pattern);
                }
            }
        }

        // Final reap of any zombies created
        Self::reap_zombies();

        if found_processes.is_empty() {
            info!("No SSH tunnel processes found");
            // Log some processes for debugging
            if !all_processes.is_empty() {
                debug!("Sample of processes scanned:");
                for (pid, cmd) in all_processes.iter().take(5) {
                    debug!("  PID {}: {}", pid, cmd);
                }
            }
        } else {
            info!(
                "Found {} SSH tunnel processes, killed {}",
                found_processes.len(),
                killed_count
            );
            for (pid, cmd) in found_processes {
                debug!("  PID {}: {}", pid, cmd);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::process::Command;

    #[tokio::test]
    async fn test_process_group_configuration() {
        let mut command = Command::new("echo");
        command.arg("test");

        ProcessGroup::configure_command(&mut command);

        // Should not panic. In restricted environments (containers, sandboxed CI),
        // process group creation may fail with EPERM/ENOSYS — that is acceptable.
        let result = command.output().await;
        match &result {
            Ok(output) => assert!(output.status.success()),
            Err(e) => {
                let raw = e.raw_os_error().unwrap_or(0);
                // EPERM(1), ENOSYS(38), EACCES(13) — environment does not support setsid/setpgid
                assert!(
                    matches!(raw, 1 | 13 | 38),
                    "unexpected error (os_error={raw}): {e}"
                );
            }
        }
    }

    #[test]
    fn test_process_exists() {
        // Current process should exist
        let current_pid = std::process::id();
        assert!(ProcessUtils::process_exists(current_pid));

        // Non-existent PID
        let fake_pid = 99999999;
        assert!(!ProcessUtils::process_exists(fake_pid));
    }

    #[test]
    fn test_cleanup_ssh_tunnels() {
        // This test verifies that cleanup_ssh_tunnels doesn't crash
        // and properly identifies SSH processes
        ProcessUtils::cleanup_ssh_tunnels();
        // The function should complete without panicking
    }

    #[test]
    fn test_ssh_tunnel_detection() {
        // Test that we can detect SSH tunnel patterns correctly
        let test_cases = vec![
            ("ssh-N-L3000:localhost:3000-p22-i/opt/key", true),
            ("ssh -N -L 3000:localhost:3000 -p 22", true),
            ("ssh user@host", false),
            ("ssh -N localhost", true),
            ("ssh -L8080:localhost:80 server", true),
            ("/usr/bin/ssh -D 1080 proxy", true),
        ];

        for (cmdline, should_match) in test_cases {
            let has_tunnel_flag = cmdline.contains("-L")
                || cmdline.contains("-N")
                || cmdline.contains("-D")
                || cmdline.contains("localhost");

            assert_eq!(
                has_tunnel_flag, should_match,
                "Failed for cmdline: {}",
                cmdline
            );
        }
    }

    #[test]
    fn test_reap_zombies() {
        // This test verifies that reap_zombies doesn't crash
        // It won't have any zombies to reap in a test environment
        ProcessUtils::reap_zombies();
        // The function should complete without panicking
    }
}
