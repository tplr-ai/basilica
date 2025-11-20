use k8s_openapi::api::core::v1::{
    Capabilities, ExecAction, HTTPGetAction, Lifecycle, LifecycleHandler, Probe,
    ResourceRequirements, SeccompProfile, SecurityContext,
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
        initial_delay_seconds: Some(10),
        period_seconds: Some(5),
        failure_threshold: Some(30),
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
        period_seconds: Some(5),
        failure_threshold: Some(1),
        ..Default::default()
    });

    (startup_probe, liveness_probe, readiness_probe)
}

/// Builds security context for FUSE filesystem daemon.
///
/// FUSE requires the mount() syscall which is restricted by default.
/// We grant CAP_SYS_ADMIN (minimum capability for FUSE) rather than
/// privileged=true to enable seccomp filtering while allowing FUSE.
///
/// Security model:
/// - Runs as root (UID 0) - required for mount()
/// - CAP_SYS_ADMIN only - minimum capability for FUSE mount/umount
/// - RuntimeDefault seccomp - filters dangerous syscalls
/// - Read-only root filesystem - prevents tampering
/// - allowPrivilegeEscalation=true - required by K8s with CAP_SYS_ADMIN
///
/// This is more restrictive than privileged=true which:
/// - Grants ALL capabilities (not just SYS_ADMIN)
/// - Runs as Unconfined seccomp (no syscall filtering)
/// - Allows access to all host devices
pub fn build_fuse_security_context() -> SecurityContext {
    SecurityContext {
        run_as_user: Some(0),
        run_as_non_root: Some(false),
        privileged: Some(false),
        allow_privilege_escalation: Some(true),
        capabilities: Some(Capabilities {
            drop: Some(vec!["ALL".into()]),
            add: Some(vec!["SYS_ADMIN".into()]),
        }),
        read_only_root_filesystem: Some(true),
        seccomp_profile: Some(SeccompProfile {
            type_: "RuntimeDefault".into(),
            localhost_profile: None,
        }),
        ..Default::default()
    }
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
        assert_eq!(probe.initial_delay_seconds, Some(10));
        assert_eq!(probe.period_seconds, Some(5));
        assert_eq!(probe.failure_threshold, Some(30));
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
        assert_eq!(probe.period_seconds, Some(5));
        assert_eq!(probe.failure_threshold, Some(1));
    }

    #[test]
    fn test_fuse_security_context() {
        let sc = build_fuse_security_context();

        assert_eq!(sc.run_as_user, Some(0), "Must run as root for FUSE mount");
        assert_eq!(sc.run_as_non_root, Some(false));
        assert_eq!(sc.privileged, Some(false), "Should not use privileged mode");
        assert_eq!(
            sc.allow_privilege_escalation,
            Some(true),
            "Required by K8s with CAP_SYS_ADMIN"
        );
        assert_eq!(
            sc.read_only_root_filesystem,
            Some(true),
            "Should have read-only root filesystem"
        );

        let caps = sc.capabilities.as_ref().expect("Capabilities missing");
        assert_eq!(
            caps.drop,
            Some(vec!["ALL".into()]),
            "Should drop all capabilities first"
        );
        assert_eq!(
            caps.add,
            Some(vec!["SYS_ADMIN".into()]),
            "Should add only CAP_SYS_ADMIN for FUSE"
        );

        let seccomp = sc
            .seccomp_profile
            .as_ref()
            .expect("Seccomp profile missing");
        assert_eq!(
            seccomp.type_, "RuntimeDefault",
            "Should use RuntimeDefault seccomp (now actually enforced!)"
        );
        assert_eq!(
            seccomp.localhost_profile, None,
            "Should not use custom seccomp profile"
        );
    }
}
