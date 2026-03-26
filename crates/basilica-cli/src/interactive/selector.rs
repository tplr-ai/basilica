//! Interactive selection utilities

use crate::error::Result;
use basilica_common::types::GpuCategory;
use basilica_sdk::types::ApiRentalListItem;
use color_eyre::eyre::eyre;
use console::Term;
use dialoguer::{theme::ColorfulTheme, MultiSelect, Select};
use std::str::FromStr;

/// Interactive selector for CLI operations
pub struct InteractiveSelector {
    theme: ColorfulTheme,
}

impl InteractiveSelector {
    /// Create a new interactive selector
    pub fn new() -> Self {
        let theme = ColorfulTheme::default();
        Self { theme }
    }

    /// Let user select a single instance from active instances
    pub fn select_rental(&self, rentals: &[ApiRentalListItem], detailed: bool) -> Result<String> {
        if rentals.is_empty() {
            return Err(eyre!("No active instances").into());
        }

        let items: Vec<String> = rentals
            .iter()
            .map(|rental| {
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
                        if rental.gpu_specs.len() > 1 {
                            format!("{}x {}", rental.gpu_specs.len(), gpu_display_name)
                        } else {
                            format!("1x {}", gpu_display_name)
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
