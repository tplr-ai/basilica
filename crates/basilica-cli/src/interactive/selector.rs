//! Interactive selection utilities

use crate::error::Result;
use basilica_sdk::types::{ApiRentalListItem, NodeSelection};
use basilica_sdk::GpuRequirements;
use basilica_validator::api::types::AvailableNode;
use basilica_validator::gpu::GpuCategory;
use color_eyre::eyre::eyre;
use console::Term;
use dialoguer::{theme::ColorfulTheme, Confirm, MultiSelect, Select};
use std::collections::HashMap;
use std::str::FromStr;

/// Interactive selector for CLI operations
pub struct InteractiveSelector {
    theme: ColorfulTheme,
}

impl InteractiveSelector {
    /// Create a new interactive selector
    pub fn new() -> Self {
        // Create a customized theme for better display
        let theme = ColorfulTheme::default();
        // The theme already has good defaults, we can customize if needed
        Self { theme }
    }

    /// Let user select an node from available options
    pub fn select_node(
        &self,
        nodes: &[AvailableNode],
        use_detailed: bool,
        show_ids: bool,
        gpu_count_filter: Option<u32>,
    ) -> Result<NodeSelection> {
        if nodes.is_empty() {
            return Err(eyre!("No nodes available").into());
        }

        if use_detailed {
            // Detailed mode: Show all nodes individually
            self.select_node_detailed(nodes, show_ids, gpu_count_filter)
        } else {
            // Grouped mode: Group by GPU configuration
            self.select_node_grouped(nodes, gpu_count_filter)
        }
    }

    /// Select node in detailed mode (show all nodes)
    fn select_node_detailed(
        &self,
        nodes: &[AvailableNode],
        show_ids: bool,
        gpu_count_filter: Option<u32>,
    ) -> Result<NodeSelection> {
        // Filter nodes by exact GPU count if specified
        let filtered_nodes: Vec<&AvailableNode> = if let Some(required_count) = gpu_count_filter {
            nodes
                .iter()
                .filter(|e| e.node.gpu_specs.len() as u32 == required_count)
                .collect()
        } else {
            nodes.iter().collect()
        };

        if filtered_nodes.is_empty() {
            if let Some(required_count) = gpu_count_filter {
                return Err(
                    eyre!("No nodes available with exactly {} GPU(s)", required_count).into(),
                );
            } else {
                return Err(eyre!("No nodes available").into());
            }
        }

        // Collect all display components to calculate proper column widths
        struct DisplayComponents {
            node_id: String,
            gpu_info: String,
            cpu_info: String,
            ram_info: String,
            use_case: String,
        }

        let display_components: Vec<DisplayComponents> = filtered_nodes
            .iter()
            .map(|node| {
                // Extract node ID (remove miner prefix if present)
                let node_id = if show_ids {
                    node.node
                        .id
                        .split_once("__")
                        .map(|(_, id)| id)
                        .unwrap_or(&node.node.id)
                        .to_string()
                } else {
                    String::new()
                };
                // Format GPU info
                let gpu_info = if node.node.gpu_specs.is_empty() {
                    "No GPUs".to_string()
                } else {
                    let gpu = &node.node.gpu_specs[0];
                    let gpu_display_name = gpu.name.clone(); // Full name in detailed mode
                    if node.node.gpu_specs.len() > 1 {
                        format!("{}x {}", node.node.gpu_specs.len(), gpu_display_name)
                    } else {
                        format!("1x {}", gpu_display_name)
                    }
                };

                // Format CPU info - handle "Unknown" case
                let cpu_info =
                    if node.node.cpu_specs.model == "Unknown" || node.node.cpu_specs.cores == 0 {
                        "Unknown CPU".to_string()
                    } else {
                        format!(
                            "{} ({} cores)",
                            node.node.cpu_specs.model, node.node.cpu_specs.cores
                        )
                    };

                // Format RAM info
                let ram_info = if node.node.cpu_specs.memory_gb == 0 {
                    "Unknown".to_string()
                } else {
                    format!("{}GB", node.node.cpu_specs.memory_gb)
                };

                // Get use case description for GPUs
                let use_case = if node.node.gpu_specs.is_empty() {
                    "General compute".to_string()
                } else {
                    let gpu = &node.node.gpu_specs[0];
                    GpuCategory::from_str(&gpu.name)
                        .unwrap()
                        .description()
                        .to_string()
                };

                DisplayComponents {
                    node_id,
                    gpu_info,
                    cpu_info,
                    ram_info,
                    use_case,
                }
            })
            .collect();

        // Calculate maximum widths for each column
        let id_max_width = if show_ids {
            display_components
                .iter()
                .map(|c| c.node_id.len())
                .max()
                .unwrap_or(36) // UUID default length
        } else {
            0
        };

        let gpu_max_width = display_components
            .iter()
            .map(|c| c.gpu_info.len())
            .max()
            .unwrap_or(30);

        let cpu_max_width = display_components
            .iter()
            .map(|c| c.cpu_info.len())
            .max()
            .unwrap_or(40);

        let ram_max_width = display_components
            .iter()
            .map(|c| c.ram_info.len())
            .max()
            .unwrap_or(10);

        // Create formatted items for the selector
        let selector_items: Vec<String> = display_components
            .iter()
            .map(|components| {
                if show_ids {
                    format!(
                        "{:<id_width$} │ {:<gpu_width$} │ {:<cpu_width$} │ {:<ram_width$} │ {}",
                        components.node_id,
                        components.gpu_info,
                        components.cpu_info,
                        components.ram_info,
                        components.use_case,
                        id_width = id_max_width,
                        gpu_width = gpu_max_width,
                        cpu_width = cpu_max_width,
                        ram_width = ram_max_width
                    )
                } else {
                    format!(
                        "{:<gpu_width$} │ {:<cpu_width$} │ {:<ram_width$} │ {}",
                        components.gpu_info,
                        components.cpu_info,
                        components.ram_info,
                        components.use_case,
                        gpu_width = gpu_max_width,
                        cpu_width = cpu_max_width,
                        ram_width = ram_max_width
                    )
                }
            })
            .collect();

        let selection = Select::with_theme(&self.theme)
            .with_prompt("Select node")
            .items(&selector_items)
            .default(0)
            .interact_opt()
            .map_err(|e| eyre!("Selection failed: {}", e))?;

        let selection = match selection {
            Some(s) => s,
            None => return Err(eyre!("Selection cancelled").into()),
        };

        // Get the selected node ID
        let node_id = filtered_nodes[selection].node.id.clone();
        let node_id = match node_id.split_once("__") {
            Some((_, second)) => second.to_string(),
            None => node_id,
        };

        Ok(NodeSelection::NodeId { node_id })
    }

    /// Select node in grouped mode (group by GPU configuration)
    fn select_node_grouped(
        &self,
        nodes: &[AvailableNode],
        gpu_count_filter: Option<u32>,
    ) -> Result<NodeSelection> {
        // Group nodes by GPU configuration
        let mut gpu_groups: HashMap<String, (String, u32, u32)> = HashMap::new();

        for node in nodes {
            let key = if node.node.gpu_specs.is_empty() {
                "no_gpu".to_string()
            } else {
                let gpu = &node.node.gpu_specs[0];
                let category = GpuCategory::from_str(&gpu.name)
                    .unwrap_or(GpuCategory::Other(gpu.name.clone()));
                let gpu_count = node.node.gpu_specs.len() as u32;
                format!("{}_{}_{}", gpu_count, category, gpu.memory_gb)
            };

            gpu_groups.entry(key).or_insert_with(|| {
                if node.node.gpu_specs.is_empty() {
                    ("".to_string(), 0, 0)
                } else {
                    let gpu = &node.node.gpu_specs[0];
                    let category = GpuCategory::from_str(&gpu.name)
                        .unwrap_or(GpuCategory::Other(gpu.name.clone()));
                    let gpu_count = node.node.gpu_specs.len() as u32;
                    (category.to_string(), gpu_count, gpu.memory_gb)
                }
            });
        }

        // Create sorted list of unique GPU configurations
        let mut gpu_configs: Vec<(String, String, u32, u32)> = gpu_groups
            .into_iter()
            .map(|(key, (gpu_type, count, memory))| (key, gpu_type, count, memory))
            .filter(|(_, _, count, _)| {
                // Filter by GPU count if specified
                if let Some(required_count) = gpu_count_filter {
                    *count == required_count
                } else {
                    true
                }
            })
            .collect();
        gpu_configs.sort_by(|a, b| {
            // Sort by GPU type, then count, then memory
            a.1.cmp(&b.1).then(a.2.cmp(&b.2)).then(a.3.cmp(&b.3))
        });

        // Check if any configurations match the filter
        if gpu_configs.is_empty() {
            if let Some(required_count) = gpu_count_filter {
                return Err(eyre!(
                    "No GPU configurations available with exactly {} GPU(s)",
                    required_count
                )
                .into());
            } else {
                return Err(eyre!("No GPU configurations available").into());
            }
        }

        // Create display items with GPU use case descriptions
        let selector_items: Vec<String> = gpu_configs
            .iter()
            .map(|(_, gpu_type, count, _memory)| {
                if gpu_type.is_empty() {
                    format!("{:<30} {}", "No GPUs", "General compute")
                } else {
                    let gpu_info = if *count > 1 {
                        format!("{}x {}", count, gpu_type)
                    } else {
                        format!("1x {}", gpu_type)
                    };
                    // Parse the category string directly to get the enum and its description
                    let category = GpuCategory::from_str(gpu_type)
                        .unwrap_or(GpuCategory::Other(gpu_type.to_string()));
                    let use_case = category.description();
                    format!("{:<30} {}", gpu_info, use_case)
                }
            })
            .collect();

        let selection = Select::with_theme(&self.theme)
            .with_prompt("Select GPU configuration")
            .items(&selector_items)
            .default(0)
            .interact_opt()
            .map_err(|e| eyre!("Selection failed: {}", e))?;

        let selection = match selection {
            Some(s) => s,
            None => return Err(eyre!("Selection cancelled").into()),
        };

        let selected_config = &gpu_configs[selection];

        // Use console crate to clear the previous line properly
        let term = Term::stdout();
        let _ = term.clear_last_lines(1);

        // Confirm selection
        let display_name = &selector_items[selection];
        let confirmed = Confirm::with_theme(&self.theme)
            .with_prompt(format!("Proceed with {}?", display_name))
            .default(true)
            .interact()
            .map_err(|e| eyre!("Confirmation failed: {}", e))?;

        if !confirmed {
            return Err(eyre!("Selection cancelled").into());
        }

        // Return GPU requirements for automatic selection
        if selected_config.1.is_empty() {
            // No GPU case - just pick the first available node
            let node_id = nodes[0].node.id.clone();
            let node_id = match node_id.split_once("__") {
                Some((_, second)) => second.to_string(),
                None => node_id,
            };
            Ok(NodeSelection::NodeId { node_id })
        } else {
            // Use ExactGpuConfiguration for compact mode to ensure exact GPU count matching
            Ok(NodeSelection::ExactGpuConfiguration {
                gpu_requirements: GpuRequirements {
                    gpu_type: Some(selected_config.1.clone()),
                    gpu_count: selected_config.2,
                    min_memory_gb: 0, // We match exact memory from the selection
                },
            })
        }
    }

    /// Let user select a single instance from active instances
    pub fn select_rental(&self, rentals: &[ApiRentalListItem], detailed: bool) -> Result<String> {
        if rentals.is_empty() {
            return Err(eyre!("No active instances").into());
        }

        let items: Vec<String> = rentals
            .iter()
            .map(|rental| {
                // Format GPU info from specs
                let gpu = if rental.gpu_specs.is_empty() {
                    "Unknown GPU".to_string()
                } else {
                    let first_gpu = &rental.gpu_specs[0];
                    let all_same = rental
                        .gpu_specs
                        .iter()
                        .all(|g| g.name == first_gpu.name && g.memory_gb == first_gpu.memory_gb);

                    if all_same {
                        let gpu_display_name = if detailed {
                            first_gpu.name.clone()
                        } else {
                            let category = GpuCategory::from_str(&first_gpu.name)
                                .unwrap_or(GpuCategory::Other(first_gpu.name.clone()));
                            category.to_string()
                        };
                        if detailed {
                            // Detailed mode: show memory
                            if rental.gpu_specs.len() > 1 {
                                format!("{}x {}", rental.gpu_specs.len(), gpu_display_name)
                            } else {
                                format!("1x {}", gpu_display_name)
                            }
                        } else {
                            // Non-detailed mode: no memory
                            if rental.gpu_specs.len() > 1 {
                                format!("{}x {}", rental.gpu_specs.len(), gpu_display_name)
                            } else {
                                format!("1x {}", gpu_display_name)
                            }
                        }
                    } else {
                        rental
                            .gpu_specs
                            .iter()
                            .map(|g| {
                                if detailed {
                                    g.name.clone()
                                } else {
                                    let category = GpuCategory::from_str(&g.name)
                                        .unwrap_or(GpuCategory::Other(g.name.clone()));
                                    category.to_string()
                                }
                            })
                            .collect::<Vec<_>>()
                            .join(", ")
                    }
                };

                // Format: "GPU Type    Container Image"
                format!("{:<30} {:<30}", gpu, rental.container_image)
            })
            .collect();

        let selection = Select::with_theme(&self.theme)
            .with_prompt("Select instance")
            .items(&items)
            .default(0)
            .interact_opt()
            .map_err(|e| eyre!("Selection failed: {}", e))?;

        let selection = match selection {
            Some(s) => s,
            None => return Err(eyre!("Selection cancelled").into()),
        };

        // Clear the selection prompt line
        let term = Term::stdout();
        let _ = term.clear_last_lines(1);

        Ok(rentals[selection].rental_id.clone())
    }

    /// Let user select instance items for termination
    pub fn select_rental_items_for_termination(
        &self,
        rentals: &[ApiRentalListItem],
    ) -> Result<Vec<String>> {
        if rentals.is_empty() {
            return Err(eyre!("No active instances").into());
        }

        let items: Vec<String> = rentals
            .iter()
            .map(|rental| {
                // Format GPU info from specs
                let gpu = if rental.gpu_specs.is_empty() {
                    "Unknown GPU".to_string()
                } else {
                    let first_gpu = &rental.gpu_specs[0];
                    let all_same = rental
                        .gpu_specs
                        .iter()
                        .all(|g| g.name == first_gpu.name && g.memory_gb == first_gpu.memory_gb);

                    if all_same {
                        format!("{}x {}", rental.gpu_specs.len(), first_gpu.name)
                    } else {
                        rental
                            .gpu_specs
                            .iter()
                            .map(|g| g.name.clone())
                            .collect::<Vec<_>>()
                            .join(", ")
                    }
                };

                // Format consistently with select_rental
                format!("{:<30} {:<30}", gpu, rental.container_image)
            })
            .collect();

        let selections = MultiSelect::with_theme(&self.theme)
            .with_prompt("Select instances to terminate (Space to select, Enter to confirm)")
            .items(&items)
            .interact()
            .map_err(|e| eyre!("Selection failed: {}", e))?;

        if selections.is_empty() {
            return Err(eyre!("No instances selected").into());
        }

        let selected_ids: Vec<String> = selections
            .into_iter()
            .map(|i| rentals[i].rental_id.clone())
            .collect();

        Ok(selected_ids)
    }

    /// Confirm an action with yes/no prompt
    pub fn confirm(&self, message: &str) -> Result<bool> {
        let confirmed = dialoguer::Confirm::with_theme(&self.theme)
            .with_prompt(message)
            .default(false)
            .interact()
            .map_err(|e| eyre!("Confirmation failed: {}", e))?;

        Ok(confirmed)
    }
}

impl Default for InteractiveSelector {
    fn default() -> Self {
        Self::new()
    }
}
