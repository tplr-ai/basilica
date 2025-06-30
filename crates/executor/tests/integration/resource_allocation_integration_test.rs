use anyhow::Result;
use executor::config::{ExecutorConfig, ResourceAllocationConfig, ResourceLimits};
use executor::resources::{
    AllocationStrategy, ResourceAllocator, ResourceConstraints, ResourceManager, ResourcePool,
    ResourceRequest, ResourceType,
};
use std::collections::HashMap;
use std::sync::Arc;
use tempfile::TempDir;
use tokio::sync::{Mutex, RwLock};
use tokio::time::{sleep, timeout, Duration};
use uuid::Uuid;

#[tokio::test]
async fn test_resource_pool_management() -> Result<()> {
    let config = ResourceAllocationConfig {
        total_cpu_cores: 16,
        total_memory_gb: 64,
        total_storage_gb: 1000,
        total_gpu_count: 4,
        allocation_strategy: AllocationStrategy::BestFit,
        oversubscription_ratio: 1.2,
        reserved_cpu_percent: 10.0,
        reserved_memory_percent: 10.0,
    };

    let resource_pool = ResourcePool::new(config)?;

    // Check initial capacity
    let capacity = resource_pool.get_total_capacity().await?;
    assert_eq!(capacity.cpu_cores, 16);
    assert_eq!(capacity.memory_gb, 64);
    assert_eq!(capacity.storage_gb, 1000);
    assert_eq!(capacity.gpu_count, 4);

    // Check available resources (accounting for reserved)
    let available = resource_pool.get_available_resources().await?;
    assert!(available.cpu_cores < 16.0); // Less due to reservation
    assert!(available.memory_gb < 64.0);

    // Allocate resources
    let request = ResourceRequest {
        cpu_cores: 4.0,
        memory_gb: 16.0,
        storage_gb: 100.0,
        gpu_count: 1,
        exclusive_gpu: true,
        duration: Duration::from_secs(3600),
    };

    let allocation = resource_pool.allocate(request).await?;
    assert!(!allocation.id.is_empty());
    assert_eq!(allocation.cpu_cores, 4.0);
    assert_eq!(allocation.memory_gb, 16.0);
    assert_eq!(allocation.gpu_count, 1);

    // Verify resources were deducted
    let available_after = resource_pool.get_available_resources().await?;
    assert!(available_after.cpu_cores < available.cpu_cores);
    assert!(available_after.memory_gb < available.memory_gb);
    assert_eq!(available_after.gpu_count, available.gpu_count - 1);

    // Release resources
    resource_pool.release(&allocation.id).await?;

    // Verify resources were returned
    let available_released = resource_pool.get_available_resources().await?;
    assert_eq!(available_released.cpu_cores, available.cpu_cores);
    assert_eq!(available_released.memory_gb, available.memory_gb);
    assert_eq!(available_released.gpu_count, available.gpu_count);

    Ok(())
}

#[tokio::test]
async fn test_resource_oversubscription() -> Result<()> {
    let config = ResourceAllocationConfig {
        total_cpu_cores: 8,
        total_memory_gb: 32,
        total_storage_gb: 500,
        total_gpu_count: 2,
        allocation_strategy: AllocationStrategy::BestFit,
        oversubscription_ratio: 1.5, // Allow 50% oversubscription
        reserved_cpu_percent: 0.0,
        reserved_memory_percent: 0.0,
    };

    let resource_pool = ResourcePool::new(config)?;

    // Allocate up to physical limit
    let request1 = ResourceRequest {
        cpu_cores: 8.0,
        memory_gb: 32.0,
        storage_gb: 100.0,
        gpu_count: 0,
        exclusive_gpu: false,
        duration: Duration::from_secs(3600),
    };

    let alloc1 = resource_pool.allocate(request1).await?;

    // Should still be able to allocate due to oversubscription
    let request2 = ResourceRequest {
        cpu_cores: 4.0, // 50% of physical = 12 total (within 1.5x)
        memory_gb: 16.0,
        storage_gb: 50.0,
        gpu_count: 0,
        exclusive_gpu: false,
        duration: Duration::from_secs(3600),
    };

    let alloc2 = resource_pool.allocate(request2).await?;
    assert!(!alloc2.id.is_empty());

    // This should fail - would exceed oversubscription limit
    let request3 = ResourceRequest {
        cpu_cores: 4.0, // Would be 16 total (2x physical)
        memory_gb: 16.0,
        storage_gb: 50.0,
        gpu_count: 0,
        exclusive_gpu: false,
        duration: Duration::from_secs(3600),
    };

    let result = resource_pool.allocate(request3).await;
    assert!(
        result.is_err(),
        "Should fail when exceeding oversubscription limit"
    );

    // Cleanup
    resource_pool.release(&alloc1.id).await?;
    resource_pool.release(&alloc2.id).await?;

    Ok(())
}

#[tokio::test]
async fn test_gpu_allocation_strategies() -> Result<()> {
    let config = ResourceAllocationConfig {
        total_cpu_cores: 32,
        total_memory_gb: 128,
        total_storage_gb: 2000,
        total_gpu_count: 8,
        allocation_strategy: AllocationStrategy::GpuAffinity,
        ..Default::default()
    };

    let resource_pool = ResourcePool::new(config)?;

    // Test exclusive GPU allocation
    let exclusive_request = ResourceRequest {
        cpu_cores: 8.0,
        memory_gb: 32.0,
        storage_gb: 200.0,
        gpu_count: 2,
        exclusive_gpu: true,
        duration: Duration::from_secs(3600),
    };

    let exclusive_alloc = resource_pool.allocate(exclusive_request).await?;
    assert_eq!(exclusive_alloc.gpu_count, 2);
    assert!(exclusive_alloc.gpu_ids.len() == 2);

    // Test shared GPU allocation
    let shared_request = ResourceRequest {
        cpu_cores: 4.0,
        memory_gb: 16.0,
        storage_gb: 100.0,
        gpu_count: 1,
        exclusive_gpu: false,
        duration: Duration::from_secs(3600),
    };

    let shared_alloc1 = resource_pool.allocate(shared_request.clone()).await?;
    let shared_alloc2 = resource_pool.allocate(shared_request).await?;

    // Both should succeed but might share same GPU
    assert_eq!(shared_alloc1.gpu_count, 1);
    assert_eq!(shared_alloc2.gpu_count, 1);

    // Cleanup
    resource_pool.release(&exclusive_alloc.id).await?;
    resource_pool.release(&shared_alloc1.id).await?;
    resource_pool.release(&shared_alloc2.id).await?;

    Ok(())
}

#[tokio::test]
async fn test_resource_constraints_enforcement() -> Result<()> {
    let constraints = ResourceConstraints {
        min_cpu_cores: 1.0,
        max_cpu_cores: 8.0,
        min_memory_gb: 2.0,
        max_memory_gb: 32.0,
        max_storage_gb: 500.0,
        max_gpu_count: 2,
        require_gpu: false,
        require_exclusive_gpu: false,
    };

    let manager = ResourceManager::new_with_constraints(constraints.clone())?;

    // Test request within constraints
    let valid_request = ResourceRequest {
        cpu_cores: 4.0,
        memory_gb: 16.0,
        storage_gb: 200.0,
        gpu_count: 1,
        exclusive_gpu: false,
        duration: Duration::from_secs(3600),
    };

    assert!(manager.validate_request(&valid_request).await?);

    // Test request exceeding CPU constraint
    let invalid_cpu = ResourceRequest {
        cpu_cores: 16.0, // Exceeds max
        memory_gb: 16.0,
        storage_gb: 200.0,
        gpu_count: 0,
        exclusive_gpu: false,
        duration: Duration::from_secs(3600),
    };

    assert!(!manager.validate_request(&invalid_cpu).await?);

    // Test request below minimum memory
    let invalid_memory = ResourceRequest {
        cpu_cores: 2.0,
        memory_gb: 1.0, // Below minimum
        storage_gb: 100.0,
        gpu_count: 0,
        exclusive_gpu: false,
        duration: Duration::from_secs(3600),
    };

    assert!(!manager.validate_request(&invalid_memory).await?);

    Ok(())
}

#[tokio::test]
async fn test_resource_allocation_timeout() -> Result<()> {
    let config = ResourceAllocationConfig {
        total_cpu_cores: 4,
        total_memory_gb: 16,
        total_storage_gb: 100,
        total_gpu_count: 1,
        allocation_strategy: AllocationStrategy::FirstFit,
        ..Default::default()
    };

    let resource_pool = Arc::new(ResourcePool::new(config)?);

    // Allocate all resources
    let full_request = ResourceRequest {
        cpu_cores: 4.0,
        memory_gb: 16.0,
        storage_gb: 100.0,
        gpu_count: 1,
        exclusive_gpu: true,
        duration: Duration::from_secs(2), // Short duration
    };

    let allocation = resource_pool.allocate(full_request).await?;

    // Try to allocate more (should wait)
    let pool_clone = resource_pool.clone();
    let wait_handle = tokio::spawn(async move {
        let request = ResourceRequest {
            cpu_cores: 2.0,
            memory_gb: 8.0,
            storage_gb: 50.0,
            gpu_count: 0,
            exclusive_gpu: false,
            duration: Duration::from_secs(3600),
        };

        pool_clone
            .allocate_with_timeout(request, Duration::from_secs(5))
            .await
    });

    // Wait for automatic release
    sleep(Duration::from_secs(3)).await;

    // Should succeed after resources are released
    let result = wait_handle.await??;
    assert!(
        result.is_some(),
        "Should allocate after resources are released"
    );

    Ok(())
}

#[tokio::test]
async fn test_resource_allocation_priority() -> Result<()> {
    let config = ResourceAllocationConfig {
        total_cpu_cores: 8,
        total_memory_gb: 32,
        total_storage_gb: 500,
        total_gpu_count: 2,
        allocation_strategy: AllocationStrategy::Priority,
        ..Default::default()
    };

    let allocator = ResourceAllocator::new(config)?;

    // Create high priority request
    let high_priority = ResourceRequest {
        cpu_cores: 4.0,
        memory_gb: 16.0,
        storage_gb: 100.0,
        gpu_count: 1,
        exclusive_gpu: false,
        duration: Duration::from_secs(3600),
    }
    .with_priority(10);

    // Create low priority request
    let low_priority = ResourceRequest {
        cpu_cores: 4.0,
        memory_gb: 16.0,
        storage_gb: 100.0,
        gpu_count: 1,
        exclusive_gpu: false,
        duration: Duration::from_secs(3600),
    }
    .with_priority(1);

    // Allocate resources to fill capacity
    let blocking_request = ResourceRequest {
        cpu_cores: 6.0,
        memory_gb: 24.0,
        storage_gb: 200.0,
        gpu_count: 1,
        exclusive_gpu: false,
        duration: Duration::from_secs(1),
    };

    let blocking_alloc = allocator.allocate(blocking_request).await?;

    // Queue both requests
    let allocator_clone1 = allocator.clone();
    let high_handle =
        tokio::spawn(async move { allocator_clone1.allocate_queued(high_priority).await });

    let allocator_clone2 = allocator.clone();
    let low_handle =
        tokio::spawn(async move { allocator_clone2.allocate_queued(low_priority).await });

    // Wait a bit
    sleep(Duration::from_millis(100)).await;

    // Release blocking allocation
    allocator.release(&blocking_alloc.id).await?;

    // High priority should be allocated first
    let high_result = timeout(Duration::from_secs(1), high_handle).await??;
    assert!(high_result.is_ok(), "High priority should be allocated");

    // Low priority should still be waiting
    let low_result = timeout(Duration::from_millis(100), low_handle).await;
    assert!(low_result.is_err(), "Low priority should still be waiting");

    Ok(())
}

#[tokio::test]
async fn test_resource_usage_tracking() -> Result<()> {
    let temp_dir = TempDir::new()?;

    let config = ExecutorConfig {
        working_dir: temp_dir.path().to_path_buf(),
        resource_allocation: ResourceAllocationConfig {
            total_cpu_cores: 16,
            total_memory_gb: 64,
            total_storage_gb: 1000,
            total_gpu_count: 4,
            track_usage_history: true,
            usage_history_retention: Duration::from_secs(3600),
            ..Default::default()
        },
        ..Default::default()
    };

    let manager = ResourceManager::new(config.resource_allocation)?;

    // Simulate resource usage over time
    for i in 0..5 {
        let request = ResourceRequest {
            cpu_cores: 2.0 * (i + 1) as f64,
            memory_gb: 8.0 * (i + 1) as f64,
            storage_gb: 50.0,
            gpu_count: i % 2,
            exclusive_gpu: false,
            duration: Duration::from_secs(300),
        };

        let alloc = manager.allocate(request).await?;

        // Record usage
        manager
            .record_usage(&alloc.id, 0.8 * (i + 1) as f64, 0.7)
            .await?;

        sleep(Duration::from_millis(100)).await;
    }

    // Get usage statistics
    let stats = manager.get_usage_statistics(Duration::from_secs(1)).await?;

    assert!(stats.avg_cpu_utilization > 0.0);
    assert!(stats.avg_memory_utilization > 0.0);
    assert!(stats.peak_cpu_usage > stats.avg_cpu_utilization);
    assert!(stats.total_allocations >= 5);

    // Get usage history
    let history = manager.get_usage_history(Duration::from_secs(1)).await?;
    assert!(!history.is_empty());

    Ok(())
}

#[tokio::test]
async fn test_resource_allocation_persistence() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let state_file = temp_dir.path().join("resource_state.json");

    let config = ResourceAllocationConfig {
        total_cpu_cores: 16,
        total_memory_gb: 64,
        total_storage_gb: 1000,
        total_gpu_count: 4,
        persist_state: true,
        state_file: Some(state_file.clone()),
        ..Default::default()
    };

    let allocation_id: String;

    // First instance - create allocations
    {
        let manager = ResourceManager::new(config.clone())?;

        let request = ResourceRequest {
            cpu_cores: 8.0,
            memory_gb: 32.0,
            storage_gb: 200.0,
            gpu_count: 2,
            exclusive_gpu: true,
            duration: Duration::from_secs(7200),
        };

        let allocation = manager.allocate(request).await?;
        allocation_id = allocation.id.clone();

        // Force state persistence
        manager.persist_state().await?;
    }

    // Verify state was saved
    assert!(state_file.exists());

    // Second instance - restore state
    {
        let manager = ResourceManager::new(config)?;

        // Should restore previous allocations
        let allocations = manager.list_allocations().await?;
        assert!(!allocations.is_empty());
        assert!(allocations.iter().any(|a| a.id == allocation_id));

        // Verify resources are still allocated
        let available = manager.get_available_resources().await?;
        assert!(available.cpu_cores < 16.0);
        assert!(available.gpu_count < 4);
    }

    Ok(())
}

#[tokio::test]
async fn test_resource_allocation_fairness() -> Result<()> {
    let config = ResourceAllocationConfig {
        total_cpu_cores: 16,
        total_memory_gb: 64,
        total_storage_gb: 1000,
        total_gpu_count: 4,
        allocation_strategy: AllocationStrategy::FairShare,
        max_allocations_per_session: 2,
        ..Default::default()
    };

    let manager = ResourceManager::new(config)?;

    // Create multiple sessions
    let session1 = Uuid::new_v4().to_string();
    let session2 = Uuid::new_v4().to_string();

    // Session 1 allocates resources
    let request1 = ResourceRequest {
        cpu_cores: 4.0,
        memory_gb: 16.0,
        storage_gb: 100.0,
        gpu_count: 1,
        exclusive_gpu: false,
        duration: Duration::from_secs(3600),
    }
    .with_session(&session1);

    let alloc1 = manager.allocate(request1.clone()).await?;
    let alloc2 = manager.allocate(request1.clone()).await?;

    // Third allocation for session 1 should fail
    let result = manager.allocate(request1).await;
    assert!(
        result.is_err(),
        "Should enforce per-session allocation limit"
    );

    // Session 2 should still be able to allocate
    let request2 = ResourceRequest {
        cpu_cores: 4.0,
        memory_gb: 16.0,
        storage_gb: 100.0,
        gpu_count: 1,
        exclusive_gpu: false,
        duration: Duration::from_secs(3600),
    }
    .with_session(&session2);

    let alloc3 = manager.allocate(request2).await?;
    assert!(!alloc3.id.is_empty());

    Ok(())
}
