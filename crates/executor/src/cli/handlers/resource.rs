//! Resource management command handlers

use super::HandlerUtils;
use crate::cli::{commands::ResourceCommands, CliContext};
use anyhow::Result;

pub async fn handle_resource_command(cmd: &ResourceCommands, context: &CliContext) -> Result<()> {
    match cmd {
        ResourceCommands::Show => show_resources(context).await,
        ResourceCommands::SetLimits {
            cpu,
            memory,
            gpu_memory,
        } => set_limits(*cpu, *memory, *gpu_memory, context).await,
        ResourceCommands::Stats { hours } => show_stats(*hours, context).await,
        ResourceCommands::Optimize => optimize_resources(context).await,
    }
}

async fn show_resources(context: &CliContext) -> Result<()> {
    HandlerUtils::print_info("Showing current resource allocation...");

    let config = HandlerUtils::load_config(&context.config_path)?;
    let state = HandlerUtils::init_executor_state(config).await?;

    let system_info = state.system_monitor.get_system_info().await?;

    println!("Resource Allocation:");
    println!("\nCPU:");
    println!("  Cores: {}", system_info.cpu.cores);
    println!("  Usage: {:.1}%", system_info.cpu.usage_percent);
    println!("  Limit: {:.1}%", state.config.system.max_cpu_usage);
    println!(
        "  Available: {:.1}%",
        state.config.system.max_cpu_usage - system_info.cpu.usage_percent
    );

    println!("\nMemory:");
    println!(
        "  Total: {} GB",
        system_info.memory.total_bytes / (1024 * 1024 * 1024)
    );
    println!(
        "  Used: {} GB ({:.1}%)",
        system_info.memory.used_bytes / (1024 * 1024 * 1024),
        system_info.memory.usage_percent
    );
    println!("  Limit: {:.1}%", state.config.system.max_memory_usage);
    println!(
        "  Available: {} GB",
        (system_info.memory.total_bytes as f32
            * (state.config.system.max_memory_usage - system_info.memory.usage_percent)
            / 100.0) as u64
            / (1024 * 1024 * 1024)
    );

    if !system_info.gpu.is_empty() {
        println!("\nGPU:");
        for (i, gpu) in system_info.gpu.iter().enumerate() {
            println!("  GPU {i}:");
            println!("    Name: {}", gpu.name);
            println!(
                "    Memory: {} GB total",
                gpu.memory_total_bytes / (1024 * 1024 * 1024)
            );
            println!(
                "    Used: {} GB ({:.1}%)",
                gpu.memory_used_bytes / (1024 * 1024 * 1024),
                gpu.memory_usage_percent
            );
            println!("    Utilization: {:.1}%", gpu.utilization_percent);
            println!(
                "    Limit: {:.1}%",
                state.config.system.max_gpu_memory_usage
            );
        }
    }

    println!("\nDisk:");
    for disk in &system_info.disk {
        println!("  {}:", disk.mount_point);
        println!("    Total: {} GB", disk.total_bytes / (1024 * 1024 * 1024));
        println!(
            "    Used: {} GB ({:.1}%)",
            disk.used_bytes / (1024 * 1024 * 1024),
            disk.usage_percent
        );
        println!(
            "    Available: {} GB",
            disk.available_bytes / (1024 * 1024 * 1024)
        );
    }

    println!("\nContainer Limits:");
    println!(
        "  Memory: {} GB",
        state.config.docker.resource_limits.memory_bytes / (1024 * 1024 * 1024)
    );
    println!(
        "  CPU: {:.1} cores",
        state.config.docker.resource_limits.cpu_cores
    );
    if let Some(gpu_mem) = state.config.docker.resource_limits.gpu_memory_bytes {
        println!("  GPU Memory: {} GB", gpu_mem / (1024 * 1024 * 1024));
    }

    HandlerUtils::print_success("Resource allocation displayed");
    Ok(())
}

async fn set_limits(
    cpu: Option<f32>,
    memory: Option<f32>,
    gpu_memory: Option<f32>,
    context: &CliContext,
) -> Result<()> {
    HandlerUtils::print_info("Setting resource limits...");

    let config = HandlerUtils::load_config(&context.config_path)?;

    if cpu.is_none() && memory.is_none() && gpu_memory.is_none() {
        HandlerUtils::print_error(
            "No limits specified. Use --cpu, --memory, or --gpu-memory flags",
        );
        return Ok(());
    }

    println!("Current Limits:");
    println!("  CPU: {:.1}%", config.system.max_cpu_usage);
    println!("  Memory: {:.1}%", config.system.max_memory_usage);
    println!("  GPU Memory: {:.1}%", config.system.max_gpu_memory_usage);

    println!("\nProposed Changes:");
    if let Some(cpu_limit) = cpu {
        if cpu_limit <= 0.0 || cpu_limit > 100.0 {
            HandlerUtils::print_error("CPU limit must be between 0 and 100");
            return Ok(());
        }
        println!(
            "  CPU: {:.1}% -> {:.1}%",
            config.system.max_cpu_usage, cpu_limit
        );
    }

    if let Some(mem_limit) = memory {
        if mem_limit <= 0.0 || mem_limit > 100.0 {
            HandlerUtils::print_error("Memory limit must be between 0 and 100");
            return Ok(());
        }
        println!(
            "  Memory: {:.1}% -> {:.1}%",
            config.system.max_memory_usage, mem_limit
        );
    }

    if let Some(gpu_limit) = gpu_memory {
        if gpu_limit <= 0.0 || gpu_limit > 100.0 {
            HandlerUtils::print_error("GPU memory limit must be between 0 and 100");
            return Ok(());
        }
        println!(
            "  GPU Memory: {:.1}% -> {:.1}%",
            config.system.max_gpu_memory_usage, gpu_limit
        );
    }

    HandlerUtils::print_warning("Note: This would update the configuration file. Restart required for changes to take effect.");
    HandlerUtils::print_info("Limits validation passed");

    Ok(())
}

async fn show_stats(hours: u32, context: &CliContext) -> Result<()> {
    HandlerUtils::print_info(&format!(
        "Historical metrics removed - showing current resource statistics (requested {hours} hours)"
    ));

    let config = HandlerUtils::load_config(&context.config_path)?;
    let state = HandlerUtils::init_executor_state(config).await?;

    let system_info = state.system_monitor.get_system_info().await?;

    println!("Current Resource Statistics:");
    println!("\nCPU Usage:");
    println!("  Current: {:.1}%", system_info.cpu.usage_percent);
    println!("  Cores: {}", system_info.cpu.cores);
    println!("  Load Average: {:?}", system_info.system.load_average);

    println!("\nMemory Usage:");
    println!("  Current: {:.1}%", system_info.memory.usage_percent);
    println!(
        "  Used: {} GB",
        system_info.memory.used_bytes / (1024 * 1024 * 1024)
    );
    println!(
        "  Total: {} GB",
        system_info.memory.total_bytes / (1024 * 1024 * 1024)
    );

    if !system_info.gpu.is_empty() {
        println!("\nGPU Usage:");
        for (idx, gpu) in system_info.gpu.iter().enumerate() {
            println!(
                "  GPU {}: {:.1}% utilization, {:.1}% memory",
                idx, gpu.utilization_percent, gpu.memory_usage_percent
            );
        }
    }

    println!("\nDisk Usage:");
    for disk in &system_info.disk {
        let used_gb = (disk.total_bytes - disk.available_bytes) / (1024 * 1024 * 1024);
        let total_gb = disk.total_bytes / (1024 * 1024 * 1024);
        let usage_percent = if disk.total_bytes > 0 {
            ((disk.total_bytes - disk.available_bytes) as f64 / disk.total_bytes as f64) * 100.0
        } else {
            0.0
        };
        println!(
            "  {}: {} GB / {} GB ({:.1}%)",
            disk.mount_point, used_gb, total_gb, usage_percent
        );
    }

    // Container resources
    let containers = state.container_manager.list_containers().await?;
    println!("\nContainer Resources:");
    println!("  Active Containers: {}", containers.len());
    println!(
        "  Max Concurrent: {}",
        state.config.docker.max_concurrent_containers
    );

    HandlerUtils::print_success("Current resource statistics displayed");
    HandlerUtils::print_info("Note: Historical metrics storage has been removed. Use 'system monitor' for real-time tracking.");
    Ok(())
}

async fn optimize_resources(context: &CliContext) -> Result<()> {
    HandlerUtils::print_info("Analyzing system for resource optimization opportunities...");

    let config = HandlerUtils::load_config(&context.config_path)?;
    let state = HandlerUtils::init_executor_state(config).await?;

    let system_info = state.system_monitor.get_system_info().await?;

    let mut recommendations: Vec<String> = Vec::new();

    // CPU optimization
    if system_info.cpu.usage_percent < 20.0 {
        recommendations.push(
            "CPU usage is very low - consider reducing resource allocation or accepting more tasks"
                .to_string(),
        );
    } else if system_info.cpu.usage_percent > 90.0 {
        recommendations.push(
            "CPU usage is very high - consider reducing concurrent tasks or upgrading hardware"
                .to_string(),
        );
    }

    // Memory optimization
    if system_info.memory.usage_percent < 30.0 {
        recommendations
            .push("Memory usage is low - consider increasing container memory limits".to_string());
    } else if system_info.memory.usage_percent > 85.0 {
        recommendations.push(
            "Memory usage is high - consider adding more RAM or reducing memory-intensive tasks"
                .to_string(),
        );
    }

    // GPU optimization
    for (i, gpu) in system_info.gpu.iter().enumerate() {
        if gpu.utilization_percent < 20.0 {
            let rec = format!("GPU {i} utilization is low - consider GPU-accelerated tasks");
            recommendations.push(rec);
        } else if gpu.memory_usage_percent > 90.0 {
            let rec = format!("GPU {i} memory is nearly full - consider reducing GPU memory usage");
            recommendations.push(rec);
        }
    }

    // Container optimization
    let containers = state.container_manager.list_containers().await?;
    if containers.len() < (state.config.docker.max_concurrent_containers / 2) as usize {
        recommendations.push(
            "Container utilization is low - consider increasing concurrent container limit"
                .to_string(),
        );
    }

    println!("Resource Optimization Analysis:");

    if recommendations.is_empty() {
        HandlerUtils::print_success("System is well-optimized - no immediate recommendations");
    } else {
        println!("\nRecommendations:");
        for (i, rec) in recommendations.iter().enumerate() {
            println!("  {}. {}", i + 1, rec);
        }

        println!("\nOptimization Summary:");
        println!("  Total recommendations: {}", recommendations.len());
        println!("  Priority: Review and implement based on workload requirements");
    }

    // Performance baseline
    println!("\nCurrent Performance Baseline:");
    println!("  CPU Cores: {}", system_info.cpu.cores);
    println!(
        "  Memory: {} GB",
        system_info.memory.total_bytes / (1024 * 1024 * 1024)
    );
    println!("  GPU Count: {}", system_info.gpu.len());
    println!("  Active Containers: {}", containers.len());

    HandlerUtils::print_success("Resource optimization analysis completed");
    Ok(())
}
