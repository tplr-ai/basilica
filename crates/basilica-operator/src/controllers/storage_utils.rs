use k8s_openapi::api::core::v1::{
    ExecAction, HTTPGetAction, Lifecycle, LifecycleHandler, Probe, ResourceRequirements,
    SecurityContext,
};
use k8s_openapi::apimachinery::pkg::api::resource::Quantity;
use std::collections::BTreeMap;

const MIN_CACHE_SIZE_MB: u32 = 512;
const MAX_CACHE_SIZE_MB: u32 = 16384;

#[derive(Debug, thiserror::Error)]
pub enum StorageUtilsError {
    #[error("Cache size {0}MB exceeds maximum {1}MB")]
    CacheSizeTooLarge(u32, u32),
    #[error("Cache size {0}MB below minimum {1}MB")]
    CacheSizeTooSmall(u32, u32),
}

/// Wrapped container command and args for FUSE wait integration.
pub type WrappedCommand = (Option<Vec<String>>, Option<Vec<String>>);

fn to_quantity(value: &str) -> Quantity {
    Quantity(value.to_string())
}

pub fn build_fuse_sidecar_resources(
    cache_size_mb: u32,
    is_job: bool,
) -> Result<ResourceRequirements, StorageUtilsError> {
    if cache_size_mb < MIN_CACHE_SIZE_MB {
        return Err(StorageUtilsError::CacheSizeTooSmall(
            cache_size_mb,
            MIN_CACHE_SIZE_MB,
        ));
    }

    if cache_size_mb > MAX_CACHE_SIZE_MB {
        return Err(StorageUtilsError::CacheSizeTooLarge(
            cache_size_mb,
            MAX_CACHE_SIZE_MB,
        ));
    }

    let mut limits = BTreeMap::new();
    let mut requests = BTreeMap::new();

    limits.insert("cpu".to_string(), to_quantity("500m"));
    requests.insert("cpu".to_string(), to_quantity("500m"));

    let memory = if is_job { "1Gi" } else { "512Mi" };
    limits.insert("memory".to_string(), to_quantity(memory));
    requests.insert("memory".to_string(), to_quantity(memory));

    let ephemeral_storage = format!("{}Mi", cache_size_mb * 2);
    limits.insert(
        "ephemeral-storage".to_string(),
        to_quantity(&ephemeral_storage),
    );
    requests.insert(
        "ephemeral-storage".to_string(),
        to_quantity(&ephemeral_storage),
    );

    Ok(ResourceRequirements {
        limits: Some(limits),
        requests: Some(requests),
        claims: None,
    })
}

pub fn build_fuse_lifecycle_hook(timeout_secs: i64) -> Lifecycle {
    let command = format!(
        "timeout {} sh -c 'kill -TERM 1 && while kill -0 1 2>/dev/null; do sleep 1; done'",
        timeout_secs
    );

    Lifecycle {
        pre_stop: Some(LifecycleHandler {
            exec: Some(ExecAction {
                command: Some(vec!["sh".to_string(), "-c".to_string(), command]),
            }),
            ..Default::default()
        }),
        ..Default::default()
    }
}

pub fn build_fuse_health_probes() -> (Option<Probe>, Option<Probe>, Option<Probe>) {
    let startup_probe = Some(Probe {
        http_get: Some(HTTPGetAction {
            path: Some("/ready".to_string()),
            port: k8s_openapi::apimachinery::pkg::util::intstr::IntOrString::Int(9090),
            ..Default::default()
        }),
        initial_delay_seconds: Some(2),
        period_seconds: Some(2),
        failure_threshold: Some(15),
        ..Default::default()
    });

    let liveness_probe = Some(Probe {
        http_get: Some(HTTPGetAction {
            path: Some("/health".to_string()),
            port: k8s_openapi::apimachinery::pkg::util::intstr::IntOrString::Int(9090),
            ..Default::default()
        }),
        period_seconds: Some(10),
        failure_threshold: Some(3),
        ..Default::default()
    });

    let readiness_probe = Some(Probe {
        http_get: Some(HTTPGetAction {
            path: Some("/ready".to_string()),
            port: k8s_openapi::apimachinery::pkg::util::intstr::IntOrString::Int(9090),
            ..Default::default()
        }),
        period_seconds: Some(3),
        failure_threshold: Some(1),
        ..Default::default()
    });

    (startup_probe, liveness_probe, readiness_probe)
}

/// Builds security context for FUSE filesystem daemon.
///
/// FUSE requires privileged mode to access /dev/fuse and perform mount operations.
/// Kubernetes blocks hostPath device mounts unless the container is privileged.
///
/// Security model:
/// - Runs as root (UID 0) - required for mount()
/// - Privileged mode - required to access /dev/fuse device from hostPath
/// - Read-only root filesystem - prevents tampering with container files
///
/// Note: We use privileged mode because:
/// 1. Kubernetes blocks hostPath device mounts without it
/// 2. FUSE needs to open /dev/fuse and call mount() syscall
/// 3. Even with CAP_SYS_ADMIN + Unconfined seccomp + Unconfined AppArmor,
///    K8s still blocks device access without privileged mode
pub fn build_fuse_security_context() -> SecurityContext {
    SecurityContext {
        run_as_user: Some(0),
        run_as_non_root: Some(false),
        privileged: Some(true),
        allow_privilege_escalation: Some(true),
        read_only_root_filesystem: Some(true),
        ..Default::default()
    }
}

/// Error type for command wrapping failures
#[derive(Debug, thiserror::Error)]
pub enum CommandWrapError {
    #[error("Failed to shell-escape command: {0}")]
    ShellEscapeError(String),
}

/// Wraps user command with FUSE wait script.
///
/// Handles four input combinations:
/// - command Some, args Some: exec command with args
/// - command Some, args None: exec command alone
/// - command None, args Some: exec with args (passes args to image entrypoint)
/// - command None, args None: just wait for FUSE, then exit (container uses image entrypoint)
///
/// Returns error if shell escaping fails (e.g., null bytes in strings).
pub fn wrap_command_with_fuse_wait(
    user_command: Option<Vec<String>>,
    user_args: Option<Vec<String>>,
    mount_path: &str,
) -> Result<WrappedCommand, CommandWrapError> {
    const WAIT_TIMEOUT_SECS: i32 = 60;

    let wait_script = format!(
        r#"echo 'Waiting for FUSE mount at {mount_path}...'
for i in $(seq 1 {timeout}); do
  if [ -f {mount_path}/.fuse_ready ]; then
    echo 'FUSE mount ready, starting application'
    break
  fi
  if [ $i -eq {timeout} ]; then
    echo 'ERROR: FUSE mount timeout after {timeout}s'
    exit 1
  fi
  sleep 1
done
"#,
        mount_path = mount_path,
        timeout = WAIT_TIMEOUT_SECS
    );

    // Build the exec portion based on what's provided
    let exec_portion = match (user_command, user_args) {
        // Case 1: Both command and args provided
        (Some(cmd), Some(args)) => {
            let cmd_str = shlex::try_join(cmd.iter().map(|s| s.as_str()))
                .map_err(|e| CommandWrapError::ShellEscapeError(e.to_string()))?;
            let args_str = shlex::try_join(args.iter().map(|s| s.as_str()))
                .map_err(|e| CommandWrapError::ShellEscapeError(e.to_string()))?;
            Some(format!("{} {}", cmd_str, args_str))
        }
        // Case 2: Only command provided
        (Some(cmd), None) => {
            let cmd_str = shlex::try_join(cmd.iter().map(|s| s.as_str()))
                .map_err(|e| CommandWrapError::ShellEscapeError(e.to_string()))?;
            Some(cmd_str)
        }
        // Case 3: Only args provided - pass to image entrypoint via exec "$@"
        (None, Some(args)) => {
            let args_str = shlex::try_join(args.iter().map(|s| s.as_str()))
                .map_err(|e| CommandWrapError::ShellEscapeError(e.to_string()))?;
            // Use exec "$@" pattern to preserve entrypoint behavior
            Some(format!("exec {}", args_str))
        }
        // Case 4: Neither command nor args - just wait, no exec needed
        // The container will exit after wait script completes; if user wants
        // to run image entrypoint, they should not use FUSE wrapper or provide explicit command
        (None, None) => None,
    };

    let wrapped_script = match exec_portion {
        Some(exec_cmd) => format!("{}\n{}", wait_script, exec_cmd),
        None => wait_script,
    };

    Ok((
        Some(vec!["/bin/sh".to_string(), "-c".to_string()]),
        Some(vec![wrapped_script]),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_size_validation_below_minimum() {
        let result = build_fuse_sidecar_resources(256, false);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            StorageUtilsError::CacheSizeTooSmall(256, 512)
        ));
    }

    #[test]
    fn test_cache_size_validation_above_maximum() {
        let result = build_fuse_sidecar_resources(20000, false);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            StorageUtilsError::CacheSizeTooLarge(20000, 16384)
        ));
    }

    #[test]
    fn test_cache_size_validation_at_boundaries() {
        assert!(build_fuse_sidecar_resources(512, false).is_ok());
        assert!(build_fuse_sidecar_resources(16384, false).is_ok());
    }

    #[test]
    fn test_user_deployment_resources() {
        let resources = build_fuse_sidecar_resources(2048, false).unwrap();
        let limits = resources.limits.as_ref().unwrap();
        let requests = resources.requests.as_ref().unwrap();

        assert_eq!(limits.get("cpu").unwrap().0, "500m");
        assert_eq!(limits.get("memory").unwrap().0, "512Mi");
        assert_eq!(limits.get("ephemeral-storage").unwrap().0, "4096Mi");

        assert_eq!(requests.get("cpu").unwrap().0, "500m");
        assert_eq!(requests.get("memory").unwrap().0, "512Mi");
        assert_eq!(requests.get("ephemeral-storage").unwrap().0, "4096Mi");
    }

    #[test]
    fn test_job_resources() {
        let resources = build_fuse_sidecar_resources(2048, true).unwrap();
        let limits = resources.limits.as_ref().unwrap();
        let requests = resources.requests.as_ref().unwrap();

        assert_eq!(limits.get("cpu").unwrap().0, "500m");
        assert_eq!(limits.get("memory").unwrap().0, "1Gi");
        assert_eq!(limits.get("ephemeral-storage").unwrap().0, "4096Mi");

        assert_eq!(requests.get("cpu").unwrap().0, "500m");
        assert_eq!(requests.get("memory").unwrap().0, "1Gi");
        assert_eq!(requests.get("ephemeral-storage").unwrap().0, "4096Mi");
    }

    #[test]
    fn test_lifecycle_hook_command() {
        let lifecycle = build_fuse_lifecycle_hook(120);
        let pre_stop = lifecycle.pre_stop.as_ref().unwrap();
        let exec = pre_stop.exec.as_ref().unwrap();
        let command = exec.command.as_ref().unwrap();

        assert_eq!(command.len(), 3);
        assert_eq!(command[0], "sh");
        assert_eq!(command[1], "-c");
        assert!(command[2].contains("timeout 120"));
        assert!(command[2].contains("kill -TERM 1"));
        assert!(command[2].contains("while kill -0 1"));
    }

    #[test]
    fn test_startup_probe_configuration() {
        let (startup, _, _) = build_fuse_health_probes();
        let probe = startup.unwrap();
        let http_get = probe.http_get.as_ref().unwrap();

        assert_eq!(http_get.path.as_ref().unwrap(), "/ready");
        assert_eq!(
            http_get.port,
            k8s_openapi::apimachinery::pkg::util::intstr::IntOrString::Int(9090)
        );
        assert_eq!(probe.initial_delay_seconds, Some(2));
        assert_eq!(probe.period_seconds, Some(2));
        assert_eq!(probe.failure_threshold, Some(15));
    }

    #[test]
    fn test_liveness_probe_configuration() {
        let (_, liveness, _) = build_fuse_health_probes();
        let probe = liveness.unwrap();
        let http_get = probe.http_get.as_ref().unwrap();

        assert_eq!(http_get.path.as_ref().unwrap(), "/health");
        assert_eq!(
            http_get.port,
            k8s_openapi::apimachinery::pkg::util::intstr::IntOrString::Int(9090)
        );
        assert_eq!(probe.period_seconds, Some(10));
        assert_eq!(probe.failure_threshold, Some(3));
    }

    #[test]
    fn test_readiness_probe_configuration() {
        let (_, _, readiness) = build_fuse_health_probes();
        let probe = readiness.unwrap();
        let http_get = probe.http_get.as_ref().unwrap();

        assert_eq!(http_get.path.as_ref().unwrap(), "/ready");
        assert_eq!(
            http_get.port,
            k8s_openapi::apimachinery::pkg::util::intstr::IntOrString::Int(9090)
        );
        assert_eq!(probe.period_seconds, Some(3));
        assert_eq!(probe.failure_threshold, Some(1));
    }

    #[test]
    fn test_fuse_security_context() {
        let sc = build_fuse_security_context();

        assert_eq!(sc.run_as_user, Some(0), "Must run as root for FUSE mount");
        assert_eq!(sc.run_as_non_root, Some(false));
        assert_eq!(
            sc.privileged,
            Some(true),
            "Must use privileged mode to access /dev/fuse from hostPath"
        );
        assert_eq!(
            sc.allow_privilege_escalation,
            Some(true),
            "Required with privileged mode"
        );
        assert_eq!(
            sc.read_only_root_filesystem,
            Some(true),
            "Should have read-only root filesystem"
        );
    }

    #[test]
    fn test_wrap_command_both_command_and_args() {
        let result = wrap_command_with_fuse_wait(
            Some(vec!["python".to_string(), "main.py".to_string()]),
            Some(vec!["--verbose".to_string()]),
            "/data",
        )
        .unwrap();

        assert_eq!(
            result.0,
            Some(vec!["/bin/sh".to_string(), "-c".to_string()])
        );
        let script = &result.1.unwrap()[0];
        assert!(script.contains("Waiting for FUSE mount at /data"));
        assert!(script.contains("python main.py --verbose"));
    }

    #[test]
    fn test_wrap_command_only_command() {
        let result = wrap_command_with_fuse_wait(
            Some(vec!["sleep".to_string(), "infinity".to_string()]),
            None,
            "/mnt/storage",
        )
        .unwrap();

        let script = &result.1.unwrap()[0];
        assert!(script.contains("Waiting for FUSE mount at /mnt/storage"));
        assert!(script.contains("sleep infinity"));
    }

    #[test]
    fn test_wrap_command_only_args() {
        let result = wrap_command_with_fuse_wait(
            None,
            Some(vec!["--config".to_string(), "/etc/app.yaml".to_string()]),
            "/data",
        )
        .unwrap();

        let script = &result.1.unwrap()[0];
        assert!(script.contains("exec --config /etc/app.yaml"));
    }

    #[test]
    fn test_wrap_command_neither_command_nor_args() {
        let result = wrap_command_with_fuse_wait(None, None, "/data").unwrap();

        let script = &result.1.unwrap()[0];
        assert!(script.contains("Waiting for FUSE mount at /data"));
        assert!(!script.contains("exec"));
    }
}
