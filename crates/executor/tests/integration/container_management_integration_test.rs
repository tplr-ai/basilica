use anyhow::Result;
use executor::config::{ContainerConfig, ExecutorConfig, ResourceLimits};
use executor::container::{
    ContainerManager, ContainerRuntime, ContainerState, DockerManager, PodmanManager, RuntimeType,
};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tempfile::TempDir;
use tokio::sync::RwLock;
use tokio::time::{sleep, timeout, Duration};
use uuid::Uuid;

#[tokio::test]
async fn test_docker_container_lifecycle() -> Result<()> {
    let temp_dir = TempDir::new()?;

    let config = ContainerConfig {
        runtime: RuntimeType::Docker,
        image_registry: "docker.io".to_string(),
        image_pull_policy: "IfNotPresent".to_string(),
        network_mode: "bridge".to_string(),
        enable_gpu: true,
        gpu_runtime: Some("nvidia".to_string()),
        resource_limits: ResourceLimits {
            cpu_cores: 4.0,
            memory_gb: 8.0,
            storage_gb: 50.0,
            gpu_count: 1,
        },
        security_opts: vec!["no-new-privileges".to_string()],
        ..Default::default()
    };

    let docker_manager = DockerManager::new(config.clone())?;

    // Check if Docker is available
    if !docker_manager.is_available().await {
        println!("Docker not available, skipping test");
        return Ok(());
    }

    // Create container
    let container_id = docker_manager
        .create_container(
            "alpine:latest",
            vec!["sleep", "300"],
            HashMap::from([("TEST_VAR".to_string(), "test_value".to_string())]),
            vec![],
            Some(temp_dir.path().to_path_buf()),
        )
        .await?;

    // Start container
    docker_manager.start_container(&container_id).await?;

    // Check container state
    let state = docker_manager.get_container_state(&container_id).await?;
    assert_eq!(state, ContainerState::Running);

    // Execute command in container
    let output = docker_manager
        .exec_in_container(
            &container_id,
            vec!["echo", "Hello from container"],
            HashMap::new(),
        )
        .await?;

    assert!(output.contains("Hello from container"));

    // Get container logs
    let logs = docker_manager.get_container_logs(&container_id, 10).await?;
    assert!(!logs.is_empty());

    // Stop container
    docker_manager
        .stop_container(&container_id, Duration::from_secs(10))
        .await?;

    let state = docker_manager.get_container_state(&container_id).await?;
    assert_eq!(state, ContainerState::Stopped);

    // Remove container
    docker_manager.remove_container(&container_id).await?;

    // Verify container is gone
    let state_result = docker_manager.get_container_state(&container_id).await;
    assert!(state_result.is_err());

    Ok(())
}

#[tokio::test]
async fn test_container_resource_enforcement() -> Result<()> {
    let config = ContainerConfig {
        runtime: RuntimeType::Docker,
        resource_limits: ResourceLimits {
            cpu_cores: 2.0,
            memory_gb: 4.0,
            storage_gb: 20.0,
            gpu_count: 0,
        },
        ..Default::default()
    };

    let docker_manager = DockerManager::new(config.clone())?;

    if !docker_manager.is_available().await {
        println!("Docker not available, skipping test");
        return Ok(());
    }

    // Create container with resource limits
    let container_id = docker_manager
        .create_container_with_limits(
            "alpine:latest",
            vec![
                "sh",
                "-c",
                "while true; do echo 'Working...'; sleep 1; done",
            ],
            HashMap::new(),
            &config.resource_limits,
        )
        .await?;

    docker_manager.start_container(&container_id).await?;

    // Get container stats
    let stats = docker_manager.get_container_stats(&container_id).await?;

    // Verify resource limits are applied
    assert!(stats.cpu_limit <= 2.0);
    assert!(stats.memory_limit_mb <= 4096);

    // Cleanup
    docker_manager
        .stop_container(&container_id, Duration::from_secs(5))
        .await?;
    docker_manager.remove_container(&container_id).await?;

    Ok(())
}

#[tokio::test]
async fn test_gpu_container_configuration() -> Result<()> {
    let config = ContainerConfig {
        runtime: RuntimeType::Docker,
        enable_gpu: true,
        gpu_runtime: Some("nvidia".to_string()),
        resource_limits: ResourceLimits {
            cpu_cores: 4.0,
            memory_gb: 16.0,
            storage_gb: 100.0,
            gpu_count: 2,
        },
        ..Default::default()
    };

    let docker_manager = DockerManager::new(config.clone())?;

    if !docker_manager.is_available().await {
        println!("Docker not available, skipping test");
        return Ok(());
    }

    if !docker_manager.has_gpu_support().await {
        println!("GPU support not available, skipping GPU tests");
        return Ok(());
    }

    // Create GPU-enabled container
    let container_id = docker_manager
        .create_gpu_container(
            "nvidia/cuda:12.0-base",
            vec!["nvidia-smi"],
            HashMap::new(),
            2, // Request 2 GPUs
        )
        .await?;

    docker_manager.start_container(&container_id).await?;

    // Wait for container to complete
    sleep(Duration::from_secs(2)).await;

    // Get output
    let logs = docker_manager
        .get_container_logs(&container_id, 100)
        .await?;

    // Should contain nvidia-smi output
    assert!(logs.contains("NVIDIA") || logs.contains("GPU"));

    // Cleanup
    docker_manager.remove_container(&container_id).await?;

    Ok(())
}

#[tokio::test]
async fn test_container_volume_management() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let host_dir = temp_dir.path().join("host_data");
    std::fs::create_dir_all(&host_dir)?;

    // Create test file on host
    let test_file = host_dir.join("test.txt");
    std::fs::write(&test_file, "Hello from host")?;

    let config = ContainerConfig {
        runtime: RuntimeType::Docker,
        ..Default::default()
    };

    let docker_manager = DockerManager::new(config)?;

    if !docker_manager.is_available().await {
        println!("Docker not available, skipping test");
        return Ok(());
    }

    // Create container with volume mount
    let container_id = docker_manager
        .create_container(
            "alpine:latest",
            vec!["cat", "/data/test.txt"],
            HashMap::new(),
            vec![(host_dir.to_string_lossy().to_string(), "/data".to_string())],
            None,
        )
        .await?;

    docker_manager.start_container(&container_id).await?;

    // Wait for container to complete
    sleep(Duration::from_secs(1)).await;

    // Check output
    let logs = docker_manager.get_container_logs(&container_id, 10).await?;
    assert!(logs.contains("Hello from host"));

    // Write from container
    let write_container = docker_manager
        .create_container(
            "alpine:latest",
            vec![
                "sh",
                "-c",
                "echo 'Hello from container' > /data/container.txt",
            ],
            HashMap::new(),
            vec![(host_dir.to_string_lossy().to_string(), "/data".to_string())],
            None,
        )
        .await?;

    docker_manager.start_container(&write_container).await?;
    sleep(Duration::from_secs(1)).await;

    // Verify file was written to host
    let container_file = host_dir.join("container.txt");
    assert!(container_file.exists());
    let content = std::fs::read_to_string(&container_file)?;
    assert!(content.contains("Hello from container"));

    // Cleanup
    docker_manager.remove_container(&container_id).await?;
    docker_manager.remove_container(&write_container).await?;

    Ok(())
}

#[tokio::test]
async fn test_container_network_isolation() -> Result<()> {
    let config = ContainerConfig {
        runtime: RuntimeType::Docker,
        network_mode: "none".to_string(), // Isolated network
        ..Default::default()
    };

    let docker_manager = DockerManager::new(config)?;

    if !docker_manager.is_available().await {
        println!("Docker not available, skipping test");
        return Ok(());
    }

    // Create isolated container
    let container_id = docker_manager
        .create_container(
            "alpine:latest",
            vec![
                "sh",
                "-c",
                "ping -c 1 8.8.8.8 || echo 'Network unreachable'",
            ],
            HashMap::new(),
            vec![],
            None,
        )
        .await?;

    docker_manager.start_container(&container_id).await?;

    // Wait for completion
    sleep(Duration::from_secs(2)).await;

    // Check that network is isolated
    let logs = docker_manager.get_container_logs(&container_id, 10).await?;
    assert!(logs.contains("Network unreachable") || logs.contains("bad address"));

    // Cleanup
    docker_manager.remove_container(&container_id).await?;

    Ok(())
}

#[tokio::test]
async fn test_container_health_monitoring() -> Result<()> {
    let config = ContainerConfig {
        runtime: RuntimeType::Docker,
        health_check_interval: Duration::from_secs(1),
        health_check_timeout: Duration::from_secs(5),
        ..Default::default()
    };

    let docker_manager = DockerManager::new(config)?;

    if !docker_manager.is_available().await {
        println!("Docker not available, skipping test");
        return Ok(());
    }

    // Create container with health check
    let container_id = docker_manager
        .create_container_with_healthcheck(
            "alpine:latest",
            vec!["sh", "-c", "while true; do echo 'healthy'; sleep 1; done"],
            HashMap::new(),
            vec!["CMD", "echo", "healthy"],
            Duration::from_secs(1),
            3,
        )
        .await?;

    docker_manager.start_container(&container_id).await?;

    // Wait for health check to run
    sleep(Duration::from_secs(3)).await;

    // Check health status
    let is_healthy = docker_manager.is_container_healthy(&container_id).await?;
    assert!(is_healthy);

    // Create unhealthy container
    let unhealthy_id = docker_manager
        .create_container_with_healthcheck(
            "alpine:latest",
            vec!["sh", "-c", "while true; do sleep 1; done"],
            HashMap::new(),
            vec!["CMD", "false"], // Always fails
            Duration::from_secs(1),
            2,
        )
        .await?;

    docker_manager.start_container(&unhealthy_id).await?;

    // Wait for health checks
    sleep(Duration::from_secs(3)).await;

    let is_unhealthy = docker_manager.is_container_healthy(&unhealthy_id).await?;
    assert!(!is_unhealthy);

    // Cleanup
    docker_manager
        .stop_container(&container_id, Duration::from_secs(5))
        .await?;
    docker_manager.remove_container(&container_id).await?;
    docker_manager
        .stop_container(&unhealthy_id, Duration::from_secs(5))
        .await?;
    docker_manager.remove_container(&unhealthy_id).await?;

    Ok(())
}

#[tokio::test]
async fn test_concurrent_container_operations() -> Result<()> {
    let config = ContainerConfig {
        runtime: RuntimeType::Docker,
        max_concurrent_containers: 5,
        ..Default::default()
    };

    let docker_manager = Arc::new(DockerManager::new(config)?);

    if !docker_manager.is_available().await {
        println!("Docker not available, skipping test");
        return Ok(());
    }

    let mut handles = vec![];

    // Spawn concurrent container operations
    for i in 0..5 {
        let manager = docker_manager.clone();
        let handle = tokio::spawn(async move {
            let container_id = manager
                .create_container(
                    "alpine:latest",
                    vec!["sh", "-c", &format!("echo 'Container {}' && sleep 2", i)],
                    HashMap::new(),
                    vec![],
                    None,
                )
                .await?;

            manager.start_container(&container_id).await?;

            // Wait and get logs
            sleep(Duration::from_secs(3)).await;
            let logs = manager.get_container_logs(&container_id, 10).await?;

            // Cleanup
            manager
                .stop_container(&container_id, Duration::from_secs(5))
                .await?;
            manager.remove_container(&container_id).await?;

            Ok::<String, anyhow::Error>(logs)
        });

        handles.push(handle);
    }

    // Wait for all operations
    let results = futures::future::try_join_all(handles).await?;

    // Verify all containers ran successfully
    for (i, result) in results.iter().enumerate() {
        match result {
            Ok(logs) => assert!(logs.contains(&format!("Container {}", i))),
            Err(e) => panic!("Container {} failed: {}", i, e),
        }
    }

    Ok(())
}

#[tokio::test]
async fn test_container_cleanup_on_failure() -> Result<()> {
    let config = ContainerConfig {
        runtime: RuntimeType::Docker,
        cleanup_on_failure: true,
        ..Default::default()
    };

    let docker_manager = DockerManager::new(config)?;

    if !docker_manager.is_available().await {
        println!("Docker not available, skipping test");
        return Ok(());
    }

    // Create container that will fail
    let container_id = docker_manager
        .create_container(
            "alpine:latest",
            vec!["sh", "-c", "exit 1"], // Exit with error
            HashMap::new(),
            vec![],
            None,
        )
        .await?;

    docker_manager.start_container(&container_id).await?;

    // Wait for container to exit
    sleep(Duration::from_secs(1)).await;

    // Check exit code
    let exit_code = docker_manager
        .get_container_exit_code(&container_id)
        .await?;
    assert_eq!(exit_code, 1);

    // Trigger cleanup
    docker_manager
        .cleanup_failed_container(&container_id)
        .await?;

    // Verify container was removed
    let state_result = docker_manager.get_container_state(&container_id).await;
    assert!(state_result.is_err());

    Ok(())
}

#[tokio::test]
async fn test_podman_compatibility() -> Result<()> {
    let config = ContainerConfig {
        runtime: RuntimeType::Podman,
        ..Default::default()
    };

    let podman_manager = PodmanManager::new(config)?;

    if !podman_manager.is_available().await {
        println!("Podman not available, skipping test");
        return Ok(());
    }

    // Basic podman operations
    let container_id = podman_manager
        .create_container(
            "alpine:latest",
            vec!["echo", "Hello from Podman"],
            HashMap::new(),
            vec![],
            None,
        )
        .await?;

    podman_manager.start_container(&container_id).await?;

    sleep(Duration::from_secs(1)).await;

    let logs = podman_manager.get_container_logs(&container_id, 10).await?;
    assert!(logs.contains("Hello from Podman"));

    // Cleanup
    podman_manager.remove_container(&container_id).await?;

    Ok(())
}
