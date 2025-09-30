//! Table formatting for CLI output

use crate::error::Result;
use basilica_api::country_mapping::get_country_name_from_code;
use basilica_common::LocationProfile;
use basilica_sdk::{
    types::{ApiKeyInfo, ApiRentalListItem, ExecutorDetails, GpuSpec, RentalStatusResponse},
    AvailableExecutor,
};
use basilica_validator::gpu::GpuCategory;
use chrono::{DateTime, Local};
use std::{collections::HashMap, str::FromStr};
use tabled::{settings::Style, Table, Tabled};

/// Format RFC3339 timestamp to YY-MM-DD HH:MM:SS format
fn format_timestamp(timestamp: &str) -> String {
    DateTime::parse_from_rfc3339(timestamp)
        .ok()
        .map(|dt| {
            let local_dt = dt.with_timezone(&Local);
            local_dt.format("%y-%m-%d %H:%M:%S").to_string()
        })
        .unwrap_or_else(|| timestamp.to_string())
}

/// Display executors in table format
pub fn display_executors(executors: &[ExecutorDetails]) -> Result<()> {
    #[derive(Tabled)]
    struct ExecutorRow {
        #[tabled(rename = "ID")]
        id: String,
        // #[tabled(rename = "GPUs")]
        // gpus: String,
        // #[tabled(rename = "CPU")]
        // cpu: String,
        // #[tabled(rename = "Memory")]
        // memory: String,
        #[tabled(rename = "Location")]
        location: String,
    }

    let rows: Vec<ExecutorRow> = executors
        .iter()
        .map(|executor| {
            // let gpu_info = if executor.gpu_specs.is_empty() {
            //     "None".to_string()
            // } else {
            //     format!(
            //         "{} x {} ({}GB)",
            //         executor.gpu_specs.len(),
            //         executor.gpu_specs[0].name,
            //         executor.gpu_specs[0].memory_gb
            //     )
            // };

            ExecutorRow {
                id: executor.id.clone(),
                // gpus: gpu_info,
                // cpu: format!("{} cores", executor.cpu_specs.cores),
                // memory: format!("{}GB", executor.cpu_specs.memory_gb),
                location: executor
                    .location
                    .clone()
                    .unwrap_or_else(|| "Unknown".to_string()),
            }
        })
        .collect();

    let mut table = Table::new(rows);
    table.with(Style::modern());
    println!("{table}");

    Ok(())
}

/// Display active rentals in table format (for RentalStatusResponse - legacy)
pub fn display_rentals(rentals: &[RentalStatusResponse]) -> Result<()> {
    #[derive(Tabled)]
    struct RentalRow {
        #[tabled(rename = "Rental ID")]
        rental_id: String,
        #[tabled(rename = "Status")]
        status: String,
        #[tabled(rename = "Executor")]
        executor: String,
        #[tabled(rename = "Created")]
        created: String,
    }

    let rows: Vec<RentalRow> = rentals
        .iter()
        .map(|rental| RentalRow {
            rental_id: rental.rental_id.clone(),
            status: format!("{:?}", rental.status),
            executor: rental.executor.id.clone(),
            created: rental.created_at.format("%y-%m-%d %H:%M:%S").to_string(),
        })
        .collect();

    let mut table = Table::new(rows);
    table.with(Style::modern());
    println!("{table}");

    Ok(())
}

/// Display rental items in table format
pub fn display_rental_items(
    rentals: &[ApiRentalListItem],
    show_standard: bool,
    show_ids: bool,
) -> Result<()> {
    if show_ids {
        // Detailed view with IDs
        #[derive(Tabled)]
        struct DetailedRentalRowWithIds {
            #[tabled(rename = "RENTAL ID")]
            rental_id: String,
            #[tabled(rename = "EXECUTOR ID")]
            executor_id: String,
            #[tabled(rename = "GPU")]
            gpu: String,
            #[tabled(rename = "State")]
            state: String,
            #[tabled(rename = "SSH")]
            ssh: String,
            #[tabled(rename = "Image")]
            image: String,
            #[tabled(rename = "CPU")]
            cpu: String,
            #[tabled(rename = "RAM")]
            ram: String,
            #[tabled(rename = "Location")]
            location: String,
            #[tabled(rename = "Created")]
            created: String,
        }

        let rows: Vec<DetailedRentalRowWithIds> = rentals
            .iter()
            .map(|rental| {
                // Extract the executor ID (remove miner prefix if present)
                let executor_id = rental
                    .executor_id
                    .split_once("__")
                    .map(|(_, id)| id)
                    .unwrap_or(&rental.executor_id)
                    .to_string();

                // Format GPU info from specs
                let gpu = format_gpu_info(&rental.gpu_specs, true);

                // Format CPU info
                let cpu = rental
                    .cpu_specs
                    .as_ref()
                    .map(|cpu| format!("{} ({} cores)", cpu.model, cpu.cores))
                    .unwrap_or_else(|| "Unknown".to_string());

                // Format RAM info
                let ram = rental
                    .cpu_specs
                    .as_ref()
                    .map(|cpu| format!("{}GB", cpu.memory_gb))
                    .unwrap_or_else(|| "Unknown".to_string());

                // Format location
                let location = rental
                    .location
                    .as_ref()
                    .and_then(|loc| LocationProfile::from_str(loc).ok())
                    .map(|profile| profile.to_string())
                    .unwrap_or_else(|| "Unknown".to_string());

                // Format SSH availability
                let ssh = if rental.has_ssh { "✓" } else { "✗" };

                DetailedRentalRowWithIds {
                    rental_id: rental.rental_id.clone(),
                    executor_id,
                    gpu,
                    state: rental.state.to_string(),
                    ssh: ssh.to_string(),
                    image: rental.container_image.clone(),
                    cpu,
                    ram,
                    location,
                    created: format_timestamp(&rental.created_at),
                }
            })
            .collect();

        let mut table = Table::new(rows);
        table.with(Style::modern());
        println!("{table}");
    } else if show_standard {
        // Standard view with full information (no IDs)
        #[derive(Tabled)]
        struct DetailedRentalRow {
            #[tabled(rename = "GPU")]
            gpu: String,
            #[tabled(rename = "State")]
            state: String,
            #[tabled(rename = "SSH")]
            ssh: String,
            #[tabled(rename = "Image")]
            image: String,
            #[tabled(rename = "CPU")]
            cpu: String,
            #[tabled(rename = "RAM")]
            ram: String,
            #[tabled(rename = "Location")]
            location: String,
            #[tabled(rename = "Created")]
            created: String,
        }

        let rows: Vec<DetailedRentalRow> = rentals
            .iter()
            .map(|rental| {
                // Format GPU info from specs
                let gpu = format_gpu_info(&rental.gpu_specs, true);

                // Format CPU info
                let cpu = rental
                    .cpu_specs
                    .as_ref()
                    .map(|cpu| format!("{} ({} cores)", cpu.model, cpu.cores))
                    .unwrap_or_else(|| "Unknown".to_string());

                // Format RAM info
                let ram = rental
                    .cpu_specs
                    .as_ref()
                    .map(|cpu| format!("{}GB", cpu.memory_gb))
                    .unwrap_or_else(|| "Unknown".to_string());

                // Format location
                let location = rental
                    .location
                    .as_ref()
                    .and_then(|loc| LocationProfile::from_str(loc).ok())
                    .map(|profile| profile.to_string())
                    .unwrap_or_else(|| "Unknown".to_string());

                // Format SSH availability
                let ssh = if rental.has_ssh { "✓" } else { "✗" };

                DetailedRentalRow {
                    gpu,
                    state: rental.state.to_string(),
                    ssh: ssh.to_string(),
                    image: rental.container_image.clone(),
                    cpu,
                    ram,
                    location,
                    created: format_timestamp(&rental.created_at),
                }
            })
            .collect();

        let mut table = Table::new(rows);
        table.with(Style::modern());
        println!("{table}");
    } else {
        // Compact view with essential information
        #[derive(Tabled)]
        struct CompactRentalRow {
            #[tabled(rename = "GPU")]
            gpu: String,
            #[tabled(rename = "State")]
            state: String,
            #[tabled(rename = "SSH")]
            ssh: String,
            #[tabled(rename = "Created")]
            created: String,
        }

        let rows: Vec<CompactRentalRow> = rentals
            .iter()
            .map(|rental| {
                // Format GPU info from specs
                let gpu = format_gpu_info(&rental.gpu_specs, false);

                // Format SSH availability
                let ssh = if rental.has_ssh { "✓" } else { "✗" };

                CompactRentalRow {
                    gpu,
                    state: rental.state.to_string(),
                    ssh: ssh.to_string(),
                    created: format_timestamp(&rental.created_at),
                }
            })
            .collect();

        let mut table = Table::new(rows);
        table.with(Style::modern());
        println!("{table}");
    }

    Ok(())
}

/// Helper function to format GPU info
fn format_gpu_info(gpu_specs: &[GpuSpec], detailed: bool) -> String {
    if gpu_specs.is_empty() {
        return "Unknown".to_string();
    }

    // Check if all GPUs are the same
    let first_gpu = &gpu_specs[0];
    let all_same = gpu_specs
        .iter()
        .all(|g| g.name == first_gpu.name && g.memory_gb == first_gpu.memory_gb);

    if all_same {
        let gpu_display_name = if detailed {
            // Detailed mode: show full GPU name
            first_gpu.name.clone()
        } else {
            // Compact mode: show categorized name
            GpuCategory::from_str(&first_gpu.name).unwrap().to_string()
        };

        if gpu_specs.len() > 1 {
            format!("{}x {}", gpu_specs.len(), gpu_display_name)
        } else {
            format!("1x {}", gpu_display_name)
        }
    } else {
        // List each GPU
        gpu_specs
            .iter()
            .map(|g| {
                if detailed {
                    g.name.clone()
                } else {
                    GpuCategory::from_str(&g.name).unwrap().to_string()
                }
            })
            .collect::<Vec<_>>()
            .join(", ")
    }
}

/// Display configuration in table format
pub fn display_config(config: &HashMap<String, String>) -> Result<()> {
    #[derive(Tabled)]
    struct ConfigRow {
        #[tabled(rename = "Key")]
        key: String,
        #[tabled(rename = "Value")]
        value: String,
    }

    let mut rows: Vec<ConfigRow> = config
        .iter()
        .map(|(key, value)| ConfigRow {
            key: key.clone(),
            value: value.clone(),
        })
        .collect();

    rows.sort_by(|a, b| a.key.cmp(&b.key));

    let mut table = Table::new(rows);
    table.with(Style::modern());
    println!("{table}");

    Ok(())
}

/// Display available executors in compact format (grouped by location and GPU type)
pub fn display_available_executors_compact(executors: &[AvailableExecutor]) -> Result<()> {
    if executors.is_empty() {
        println!("No available executors found matching the specified criteria.");
        return Ok(());
    }

    // Group executors by country (extracted from location) and GPU configuration
    let mut country_groups: HashMap<String, HashMap<String, Vec<&AvailableExecutor>>> =
        HashMap::new();

    for executor in executors {
        // Parse location string using LocationProfile to extract country
        let country = executor
            .executor
            .location
            .as_ref()
            .and_then(|loc| {
                LocationProfile::from_str(loc)
                    .ok()
                    .and_then(|profile| profile.country)
            })
            .unwrap_or_else(|| "Unknown".to_string());

        let gpu_key = if executor.executor.gpu_specs.is_empty() {
            "No GPU".to_string()
        } else {
            let gpu = &executor.executor.gpu_specs[0];
            let category = GpuCategory::from_str(&gpu.name).unwrap();
            let gpu_count = executor.executor.gpu_specs.len();
            format!("{}x {}", gpu_count, category)
        };

        country_groups
            .entry(country)
            .or_default()
            .entry(gpu_key)
            .or_default()
            .push(executor);
    }

    // Sort countries for consistent display
    let mut sorted_countries: Vec<_> = country_groups.keys().cloned().collect();
    sorted_countries.sort();

    println!("Available GPU Instances by Country\n");

    for country in sorted_countries {
        // Print country header with full name
        let country_display = get_country_name_from_code(&country);
        println!("{}", country_display);

        #[derive(Tabled)]
        struct CompactRow {
            #[tabled(rename = "GPU TYPE")]
            gpu_type: String,
            #[tabled(rename = "AVAILABLE")]
            available: String,
        }

        let gpu_groups = country_groups.get(&country).unwrap();
        let mut rows: Vec<CompactRow> = Vec::new();

        // Sort GPU configurations for consistent display
        let mut sorted_gpu_configs: Vec<_> = gpu_groups.keys().cloned().collect();
        sorted_gpu_configs.sort();

        for gpu_config in sorted_gpu_configs {
            let executors_in_group = gpu_groups.get(&gpu_config).unwrap();
            let count = executors_in_group.len();

            rows.push(CompactRow {
                gpu_type: gpu_config.clone(),
                available: count.to_string(),
            });
        }

        let mut table = Table::new(rows);
        table.with(Style::modern());
        println!("{}", table);
        println!();
    }

    let total_count = executors.len();
    println!("Total available executors: {}", total_count);

    Ok(())
}

/// Display API keys in table format
pub fn display_api_keys(keys: &[ApiKeyInfo]) -> Result<()> {
    #[derive(Tabled)]
    struct ApiKeyRow {
        #[tabled(rename = "Name")]
        name: String,
        #[tabled(rename = "Created")]
        created: String,
        #[tabled(rename = "Last Used")]
        last_used: String,
    }

    let rows: Vec<ApiKeyRow> = keys
        .iter()
        .map(|key| ApiKeyRow {
            name: key.name.clone(),
            created: format_timestamp(&key.created_at.to_rfc3339()),
            last_used: key
                .last_used_at
                .map(|dt| format_timestamp(&dt.to_rfc3339()))
                .unwrap_or_else(|| "Never".to_string()),
        })
        .collect();

    let mut table = Table::new(rows);
    table.with(Style::modern());
    println!("{table}");

    Ok(())
}

/// Helper function to format GPU info for an executor
fn format_executor_gpu_info(executor: &AvailableExecutor, show_full_gpu_names: bool) -> String {
    if executor.executor.gpu_specs.is_empty() {
        "No GPU".to_string()
    } else if executor.executor.gpu_specs.len() == 1 {
        // Single GPU
        let gpu = &executor.executor.gpu_specs[0];
        let gpu_display_name = if show_full_gpu_names {
            gpu.name.clone()
        } else {
            GpuCategory::from_str(&gpu.name).unwrap().to_string()
        };
        format!("1x {}", gpu_display_name)
    } else {
        // Multiple GPUs - check if they're all the same model
        let first_gpu = &executor.executor.gpu_specs[0];
        let all_same = executor
            .executor
            .gpu_specs
            .iter()
            .all(|g| g.name == first_gpu.name && g.memory_gb == first_gpu.memory_gb);

        if all_same {
            // All GPUs are identical - use count prefix format
            let gpu_display_name = if show_full_gpu_names {
                first_gpu.name.clone()
            } else {
                GpuCategory::from_str(&first_gpu.name).unwrap().to_string()
            };
            format!(
                "{}x {}",
                executor.executor.gpu_specs.len(),
                gpu_display_name
            )
        } else {
            // Different GPU models - list them individually
            let gpu_names: Vec<String> = executor
                .executor
                .gpu_specs
                .iter()
                .map(|g| {
                    if show_full_gpu_names {
                        g.name.clone()
                    } else {
                        GpuCategory::from_str(&g.name).unwrap().to_string()
                    }
                })
                .collect();
            gpu_names.join(", ")
        }
    }
}

/// Helper function to format location
fn format_executor_location(location: &Option<String>) -> String {
    location
        .as_ref()
        .map(|loc| {
            LocationProfile::from_str(loc)
                .ok()
                .map(|profile| profile.to_string())
                .unwrap_or_else(|| loc.clone())
        })
        .unwrap_or_else(|| "Unknown".to_string())
}

/// Display available executors in detailed format (individual executors)
pub fn display_available_executors_detailed(
    executors: &[AvailableExecutor],
    show_full_gpu_names: bool,
    show_ids: bool,
) -> Result<()> {
    if executors.is_empty() {
        println!("No available executors found matching the specified criteria.");
        return Ok(());
    }

    // Different structs based on whether we show IDs
    if show_ids {
        #[derive(Tabled)]
        struct DetailedExecutorRowWithId {
            #[tabled(rename = "EXECUTOR ID")]
            executor_id: String,
            #[tabled(rename = "GPU")]
            gpu_info: String,
            #[tabled(rename = "CPU")]
            cpu: String,
            #[tabled(rename = "RAM")]
            ram: String,
            #[tabled(rename = "Location")]
            location: String,
        }

        let rows: Vec<DetailedExecutorRowWithId> = executors
            .iter()
            .map(|executor| {
                // Extract the executor ID (remove miner prefix if present)
                let executor_id = executor
                    .executor
                    .id
                    .split_once("__")
                    .map(|(_, id)| id)
                    .unwrap_or(&executor.executor.id)
                    .to_string();

                DetailedExecutorRowWithId {
                    executor_id,
                    gpu_info: format_executor_gpu_info(executor, show_full_gpu_names),
                    cpu: format!(
                        "{} ({} cores)",
                        executor.executor.cpu_specs.model, executor.executor.cpu_specs.cores
                    ),
                    ram: format!("{}GB", executor.executor.cpu_specs.memory_gb),
                    location: format_executor_location(&executor.executor.location),
                }
            })
            .collect();

        let mut table = Table::new(rows);
        table.with(Style::modern());
        println!("{}", table);
    } else {
        #[derive(Tabled)]
        struct DetailedExecutorRow {
            #[tabled(rename = "GPU")]
            gpu_info: String,
            #[tabled(rename = "CPU")]
            cpu: String,
            #[tabled(rename = "RAM")]
            ram: String,
            #[tabled(rename = "Location")]
            location: String,
        }

        let rows: Vec<DetailedExecutorRow> = executors
            .iter()
            .map(|executor| DetailedExecutorRow {
                gpu_info: format_executor_gpu_info(executor, show_full_gpu_names),
                cpu: format!(
                    "{} ({} cores)",
                    executor.executor.cpu_specs.model, executor.executor.cpu_specs.cores
                ),
                ram: format!("{}GB", executor.executor.cpu_specs.memory_gb),
                location: format_executor_location(&executor.executor.location),
            })
            .collect();

        let mut table = Table::new(rows);
        table.with(Style::modern());
        println!("{}", table);
    }

    println!("\nTotal available executors: {}", executors.len());

    Ok(())
}
