//! Volume management handlers for the Basilica CLI

use crate::error::CliError;
use crate::output::{json_output, print_success, table_output};
use crate::progress::{complete_spinner_and_clear, complete_spinner_error, create_spinner};
use basilica_sdk::types::{AttachVolumeRequest, CreateVolumeRequest, VolumeResponse, VolumeStatus};
use basilica_sdk::BasilicaClient;
use console::{style, Term};
use dialoguer::{theme::ColorfulTheme, Confirm, Input, Select};

/// Pricing constant: $0.000096774 per GB per hour (~$0.07/GB/month)
const VOLUME_PRICE_PER_GB_HOUR: f64 = 0.000096774;

/// Volume providers and their available regions
const VOLUME_PROVIDERS: &[(&str, &[&str])] = &[("hyperstack", &["US-1", "CANADA-1", "NORWAY-1"])];

/// Calculate hourly and monthly cost for a given size
fn calculate_volume_cost(size_gb: u32) -> (f64, f64) {
    let hourly = VOLUME_PRICE_PER_GB_HOUR * size_gb as f64;
    let monthly = hourly * 24.0 * 30.0;
    (hourly, monthly)
}

/// Core validation for volume name characters
fn validate_name_chars(name: &str) -> Result<(), &'static str> {
    if name.is_empty() {
        return Err("Volume name cannot be empty");
    }
    if name.len() > 100 {
        return Err("Volume name too long (max 100 characters)");
    }
    if !name
        .chars()
        .all(|c| c.is_alphanumeric() || c == '-' || c == '_')
    {
        return Err("Only alphanumeric characters, hyphens, and underscores are allowed");
    }
    Ok(())
}

/// Validate volume name according to the same rules as interactive prompt
fn validate_volume_name(name: &str) -> Result<String, CliError> {
    let trimmed = name.trim();
    validate_name_chars(trimmed).map_err(|e| CliError::Internal(color_eyre::eyre::eyre!(e)))?;
    Ok(trimmed.to_string())
}

/// Prompt for volume name interactively
fn prompt_volume_name() -> Result<String, CliError> {
    let theme = ColorfulTheme::default();
    let name: String = Input::with_theme(&theme)
        .with_prompt("Volume name")
        .validate_with(|input: &String| validate_name_chars(input.trim()))
        .interact_text()
        .map_err(|e| CliError::Internal(e.into()))?;

    validate_volume_name(&name)
}

/// Prompt for volume size interactively
fn prompt_volume_size() -> Result<u32, CliError> {
    let theme = ColorfulTheme::default();
    let size: u32 = Input::with_theme(&theme)
        .with_prompt("Volume size in GB (1-10240)")
        .default(100)
        .validate_with(|input: &u32| {
            if *input == 0 || *input > 10240 {
                Err("Size must be between 1 and 10240 GB")
            } else {
                Ok(())
            }
        })
        .interact_text()
        .map_err(|e| CliError::Internal(e.into()))?;

    Ok(size)
}

/// Prompt for provider selection
fn prompt_provider() -> Result<String, CliError> {
    // If only one provider, auto-select it
    if VOLUME_PROVIDERS.len() == 1 {
        return Ok(VOLUME_PROVIDERS[0].0.to_string());
    }

    let theme = ColorfulTheme::default();
    let providers: Vec<&str> = VOLUME_PROVIDERS.iter().map(|(p, _)| *p).collect();

    let selection = Select::with_theme(&theme)
        .with_prompt("Select provider")
        .items(&providers)
        .default(0)
        .interact()
        .map_err(|e| CliError::Internal(e.into()))?;

    Ok(providers[selection].to_string())
}

/// Prompt for region selection based on provider
fn prompt_region(provider: &str) -> Result<String, CliError> {
    // Find regions for this provider
    let regions = match VOLUME_PROVIDERS
        .iter()
        .find(|(p, _)| *p == provider)
        .map(|(_, r)| *r)
    {
        Some(r) => r,
        None => return Err(CliError::InvalidProvider(provider.to_string())),
    };

    // If only one region, auto-select it
    if regions.len() == 1 {
        return Ok(regions[0].to_string());
    }

    println!(
        "{}",
        style("Note: Volumes can only be attached to rentals in the same provider and region.")
            .dim()
    );

    let theme = ColorfulTheme::default();
    let selection = Select::with_theme(&theme)
        .with_prompt("Select region")
        .items(regions)
        .default(0)
        .interact()
        .map_err(|e| CliError::Internal(e.into()))?;

    Ok(regions[selection].to_string())
}

/// Helper to truncate strings to fit column width (unicode-safe)
fn truncate(s: &str, max_len: usize) -> String {
    let char_count = s.chars().count();
    if char_count <= max_len {
        s.to_string()
    } else {
        let truncated: String = s.chars().take(max_len - 1).collect();
        format!("{}…", truncated)
    }
}

/// Select a volume interactively
///
/// # Arguments
/// * `client` - Basilica client
/// * `prompt` - Prompt text to display
/// * `status_filter` - Optional filter for volume status (None = all statuses)
async fn select_volume(
    client: &BasilicaClient,
    prompt: &str,
    status_filter: Option<VolumeStatus>,
) -> Result<VolumeResponse, CliError> {
    let response = client.list_volumes().await.map_err(CliError::Api)?;

    // Filter volumes by status if specified
    let filtered_volumes: Vec<&VolumeResponse> = response
        .volumes
        .iter()
        .filter(|v| {
            if let Some(status) = status_filter {
                v.status == status
            } else {
                true
            }
        })
        .collect();

    if filtered_volumes.is_empty() {
        let msg = match status_filter {
            Some(VolumeStatus::Available) => {
                "No available volumes found.\nCreate one with: basilica volumes create"
            }
            Some(VolumeStatus::Attached) => "No attached volumes found.",
            _ => "No volumes found.\nCreate one with: basilica volumes create",
        };
        return Err(CliError::Internal(color_eyre::eyre::eyre!(msg)));
    }

    // Format items for selection
    // Header: Name | Size | Status | Provider | Region
    let header = "  Name                 │   Size │     Status │   Provider │   Region";
    let full_prompt = format!("{}\n{}", prompt, style(header).dim());

    let items: Vec<String> = filtered_volumes
        .iter()
        .map(|v| {
            let status_str = match v.status {
                VolumeStatus::Available => style("Available").green().to_string(),
                VolumeStatus::Attached => {
                    let rental_suffix = v
                        .rental_id
                        .as_ref()
                        .map(|r| format!(" -> {}", truncate(r, 10)))
                        .unwrap_or_default();
                    format!("{}{}", style("Attached").yellow(), rental_suffix)
                }
                VolumeStatus::Pending => style("Pending").cyan().to_string(),
                VolumeStatus::Deleting => style("Deleting").red().to_string(),
                VolumeStatus::Error => style("Error").red().to_string(),
            };

            format!(
                "{:<20} │ {:>5}GB │ {:>10} │ {:>10} │ {:>8}",
                truncate(&v.name, 20),
                v.size_gb,
                status_str,
                v.provider,
                v.region
            )
        })
        .collect();

    let theme = ColorfulTheme::default();
    let selection = Select::with_theme(&theme)
        .with_prompt(&full_prompt)
        .items(&items)
        .default(0)
        .interact_opt()
        .map_err(|e| CliError::Internal(e.into()))?;

    let selection = match selection {
        Some(s) => s,
        None => {
            return Err(CliError::Internal(color_eyre::eyre::eyre!(
                "Selection cancelled"
            )))
        }
    };

    // Clear the selection prompt lines (prompt + header + items become single block)
    let term = Term::stdout();
    let _ = term.clear_last_lines(2);

    Ok(filtered_volumes[selection].clone())
}

/// Select a rental compatible with a volume (same provider and region)
///
/// # Arguments
/// * `client` - Basilica client
/// * `volume` - Volume to find compatible rentals for
async fn select_rental_for_volume(
    client: &BasilicaClient,
    volume: &VolumeResponse,
) -> Result<String, CliError> {
    // Fetch both GPU and CPU secure cloud rentals in parallel
    let (gpu_rentals, cpu_rentals) = tokio::try_join!(
        client.list_secure_cloud_rentals(),
        client.list_cpu_rentals()
    )
    .map_err(CliError::Api)?;

    // Merge all rentals for filtering
    let all_rentals: Vec<_> = gpu_rentals
        .rentals
        .iter()
        .chain(cpu_rentals.rentals.iter())
        .collect();

    // Filter for active rentals in the same provider and region
    let compatible_rentals: Vec<_> = all_rentals
        .into_iter()
        .filter(|r| {
            // Only active rentals (not stopped)
            if r.stopped_at.is_some() {
                return false;
            }

            // Check provider match
            if r.provider.to_lowercase() != volume.provider.to_lowercase() {
                return false;
            }

            // Check region match (location_code contains region info)
            if let Some(ref loc) = r.location_code {
                if loc.to_uppercase() != volume.region.to_uppercase() {
                    return false;
                }
            } else {
                // No location info, can't verify compatibility
                return false;
            }

            true
        })
        .collect();

    if compatible_rentals.is_empty() {
        return Err(CliError::Internal(color_eyre::eyre::eyre!(
            "No compatible rentals found.\n\n\
             Volume '{}' is in provider '{}' region '{}'.\n\
             You need an active rental in the same provider and region to attach this volume.\n\n\
             Start a new rental with: basilica up --compute secure-cloud",
            volume.name,
            volume.provider,
            volume.region
        )));
    }

    // Format items for selection
    // Header: Compute | Status | Provider | Rental ID
    let header = "  Compute              │       Status │   Provider │ Rental ID";
    let prompt = "Select rental to attach volume to";
    let full_prompt = format!("{}\n{}", prompt, style(header).dim());

    let items: Vec<String> = compatible_rentals
        .iter()
        .map(|r| {
            let compute_str = if r.gpu_count > 1 {
                format!("{}x {}", r.gpu_count, r.gpu_type.to_uppercase())
            } else if r.gpu_count == 1 {
                r.gpu_type.to_uppercase()
            } else {
                // CPU-only rental
                match (r.vcpu_count, r.system_memory_gb) {
                    (Some(vcpu), Some(mem)) => format!("{} vCPU / {}GB", vcpu, mem),
                    (Some(vcpu), None) => format!("{} vCPU", vcpu),
                    _ => "CPU-only".to_string(),
                }
            };

            format!(
                "{:<20} │ {:>12} │ {:>10} │ {}",
                truncate(&compute_str, 20),
                r.status,
                r.provider,
                truncate(&r.rental_id, 12)
            )
        })
        .collect();

    let theme = ColorfulTheme::default();
    let selection = Select::with_theme(&theme)
        .with_prompt(&full_prompt)
        .items(&items)
        .default(0)
        .interact_opt()
        .map_err(|e| CliError::Internal(e.into()))?;

    let selection = match selection {
        Some(s) => s,
        None => {
            return Err(CliError::Internal(color_eyre::eyre::eyre!(
                "Selection cancelled"
            )))
        }
    };

    // Clear the selection prompt lines (prompt + header + items become single block)
    let term = Term::stdout();
    let _ = term.clear_last_lines(2);

    Ok(compatible_rentals[selection].rental_id.clone())
}

/// Find a volume by ID or name from the user's volumes
async fn find_volume_by_id_or_name(
    client: &BasilicaClient,
    volume_identifier: &str,
) -> Result<VolumeResponse, CliError> {
    let response = client.list_volumes().await.map_err(CliError::Api)?;

    // First try exact ID match
    if let Some(volume) = response
        .volumes
        .iter()
        .find(|v| v.volume_id == volume_identifier)
    {
        return Ok(volume.clone());
    }

    // Then try case-insensitive name match
    let lower_identifier = volume_identifier.to_lowercase();
    if let Some(volume) = response
        .volumes
        .iter()
        .find(|v| v.name.to_lowercase() == lower_identifier)
    {
        return Ok(volume.clone());
    }

    Err(CliError::Internal(color_eyre::eyre::eyre!(
        "Volume '{}' not found. Use 'basilica volumes list' to see your volumes.",
        volume_identifier
    )))
}

/// Handle creating a new volume
pub async fn handle_create_volume(
    client: &BasilicaClient,
    name: Option<String>,
    size: Option<u32>,
    provider: Option<String>,
    region: Option<String>,
    description: Option<String>,
    json: bool,
) -> Result<(), CliError> {
    // Get provider first (needed for region selection)
    let provider = match provider {
        Some(p) => p,
        None => prompt_provider()?,
    };

    // Get region (depends on provider)
    let region = match region {
        Some(r) => r,
        None => prompt_region(&provider)?,
    };

    // Get name
    let name = match name {
        Some(n) => validate_volume_name(&n)?,
        None => prompt_volume_name()?,
    };

    // Get size
    let size = match size {
        Some(s) => {
            // Validate provided size
            if s == 0 || s > 10240 {
                return Err(CliError::Internal(color_eyre::eyre::eyre!(
                    "Volume size must be between 1 and 10240 GB"
                )));
            }
            s
        }
        None => prompt_volume_size()?,
    };

    // Calculate and display pricing
    let (hourly, monthly) = calculate_volume_cost(size);

    // Compact summary before confirmation
    println!();
    println!(
        "Volume: {} ({}GB) on {}/{}",
        style(&name).cyan(),
        size,
        provider,
        region
    );
    println!("Cost: ~${:.2}/hr (${:.2}/mo estimated)", hourly, monthly);
    println!();

    // Confirm creation
    let name_for_confirm = name.clone();
    let confirmed = tokio::task::spawn_blocking(move || {
        let theme = ColorfulTheme::default();
        Confirm::with_theme(&theme)
            .with_prompt(format!("Create volume '{}'?", name_for_confirm))
            .default(true)
            .interact()
    })
    .await
    .map_err(|e| CliError::Internal(color_eyre::eyre::eyre!("Task join error: {}", e)))?
    .map_err(|e| CliError::Internal(e.into()))?;

    if !confirmed {
        println!("Volume creation cancelled.");
        return Ok(());
    }

    // Create the volume
    let request = CreateVolumeRequest {
        name: name.clone(),
        description,
        size_gb: size,
        provider,
        region,
    };

    let spinner = create_spinner(&format!("Creating volume \"{}\"...", name));
    let volume = client.create_volume(request).await.map_err(|e| {
        complete_spinner_error(spinner.clone(), "Failed to create volume");
        CliError::Api(e)
    })?;
    complete_spinner_and_clear(spinner);

    if json {
        json_output(&volume)?;
        return Ok(());
    }

    print_success(&format!("Volume \"{}\" created", volume.name));

    Ok(())
}

/// Handle listing volumes
pub async fn handle_list_volumes(client: &BasilicaClient, json: bool) -> Result<(), CliError> {
    let response = client.list_volumes().await.map_err(CliError::Api)?;

    if json {
        json_output(&response)?;
        return Ok(());
    }

    if response.volumes.is_empty() {
        println!("No volumes found.");
        println!();
        println!(
            "Create one with: {}",
            style("basilica volumes create").cyan()
        );
        return Ok(());
    }

    table_output::display_volumes(&response.volumes)?;

    println!();
    println!("Total volumes: {}", response.total_count);

    Ok(())
}

/// Handle deleting a volume
pub async fn handle_delete_volume(
    client: &BasilicaClient,
    volume_identifier: Option<String>,
    skip_confirm: bool,
    json: bool,
) -> Result<(), CliError> {
    // Get volume - either from argument or interactive selection
    let volume = match volume_identifier {
        Some(id) => find_volume_by_id_or_name(client, &id).await?,
        None => {
            // Interactive selection - only show non-attached volumes
            select_volume(
                client,
                "Select volume to delete",
                Some(VolumeStatus::Available),
            )
            .await?
        }
    };

    // Check if volume is in available status
    if volume.status != VolumeStatus::Available {
        let message = match volume.status {
            VolumeStatus::Attached => {
                format!(
                    "Cannot delete volume '{}' because it is attached to rental '{}'.\nDetach it first with: basilica volumes detach {}",
                    volume.name,
                    volume.rental_id.as_deref().unwrap_or("unknown"),
                    volume.name
                )
            }
            VolumeStatus::Pending => {
                format!(
                    "Cannot delete volume '{}' because it is still pending.\nPlease wait for the volume to become Available before deletion.",
                    volume.name
                )
            }
            VolumeStatus::Deleting => {
                format!(
                    "Cannot delete volume '{}' because it is already being deleted.\nPlease wait for the deletion to complete.",
                    volume.name
                )
            }
            VolumeStatus::Error => {
                format!(
                    "Cannot delete volume '{}' because it is in an error state.\nPlease inspect the volume status before attempting deletion.",
                    volume.name
                )
            }
            VolumeStatus::Available => unreachable!(),
        };
        return Err(CliError::Internal(color_eyre::eyre::eyre!(message)));
    }

    // Confirm deletion if not skipped
    if !skip_confirm {
        let volume_name = volume.name.clone();
        let size_gb = volume.size_gb;

        println!(
            "{}",
            style(format!(
                "Warning: This will permanently delete volume \"{}\" ({}GB)",
                volume_name, size_gb
            ))
            .yellow()
        );

        let confirmed = tokio::task::spawn_blocking(move || {
            let theme = ColorfulTheme::default();
            Confirm::with_theme(&theme)
                .with_prompt(format!("Delete volume '{}'?", volume_name))
                .default(false)
                .interact()
        })
        .await
        .map_err(|e| CliError::Internal(color_eyre::eyre::eyre!("Task join error: {}", e)))?
        .map_err(|e| CliError::Internal(e.into()))?;

        if !confirmed {
            println!("Deletion cancelled.");
            return Ok(());
        }
    }

    // Delete the volume
    let spinner = create_spinner(&format!("Deleting volume \"{}\"...", volume.name));
    client.delete_volume(&volume.volume_id).await.map_err(|e| {
        complete_spinner_error(spinner.clone(), "Failed to delete volume");
        CliError::Api(e)
    })?;
    complete_spinner_and_clear(spinner);

    if json {
        json_output(&serde_json::json!({
            "success": true,
            "volume_id": volume.volume_id,
            "name": volume.name,
            "message": format!("Volume \"{}\" deleted", volume.name)
        }))?;
        return Ok(());
    }

    print_success(&format!("Volume \"{}\" deleted", volume.name));

    Ok(())
}

/// Handle attaching a volume to a rental
pub async fn handle_attach_volume(
    client: &BasilicaClient,
    volume_identifier: Option<String>,
    rental_id: Option<String>,
    json: bool,
) -> Result<(), CliError> {
    // Get volume - either from argument or interactive selection
    let volume = match volume_identifier {
        Some(id) => find_volume_by_id_or_name(client, &id).await?,
        None => {
            // Interactive selection - only show available (not attached) volumes
            select_volume(
                client,
                "Select volume to attach",
                Some(VolumeStatus::Available),
            )
            .await?
        }
    };

    // Check if volume is already attached
    if volume.status == VolumeStatus::Attached {
        return Err(CliError::Internal(color_eyre::eyre::eyre!(
            "Volume '{}' is already attached to rental '{}'.\nDetach it first with: basilica volumes detach {}",
            volume.name,
            volume.rental_id.as_deref().unwrap_or("unknown"),
            volume.name
        )));
    }

    // Check if volume is in a valid state to attach
    if volume.status != VolumeStatus::Available {
        return Err(CliError::Internal(color_eyre::eyre::eyre!(
            "Volume '{}' is not available for attachment (status: {}).",
            volume.name,
            volume.status
        )));
    }

    // Get rental ID - either from argument or interactive selection
    let rental_id = match rental_id {
        Some(id) => id,
        None => select_rental_for_volume(client, &volume).await?,
    };

    // Attach the volume with spinner
    let request = AttachVolumeRequest {
        rental_id: rental_id.clone(),
    };

    let rental_id_short = &rental_id[..8.min(rental_id.len())];
    let spinner = create_spinner(&format!(
        "Attaching volume \"{}\" to rental {}...",
        volume.name, rental_id_short
    ));

    let response = client
        .attach_volume(&volume.volume_id, request)
        .await
        .map_err(|e| {
            complete_spinner_error(spinner.clone(), "Failed to attach volume");
            CliError::Api(e)
        })?;
    complete_spinner_and_clear(spinner);

    if json {
        json_output(&response)?;
        return Ok(());
    }

    print_success(&format!("Volume \"{}\" attached", volume.name));

    Ok(())
}

/// Handle detaching a volume from a rental
pub async fn handle_detach_volume(
    client: &BasilicaClient,
    volume_identifier: Option<String>,
    skip_confirm: bool,
    json: bool,
) -> Result<(), CliError> {
    // Get volume - either from argument or interactive selection
    let volume = match volume_identifier {
        Some(id) => find_volume_by_id_or_name(client, &id).await?,
        None => {
            // Interactive selection - only show attached volumes
            select_volume(
                client,
                "Select volume to detach",
                Some(VolumeStatus::Attached),
            )
            .await?
        }
    };

    // Check if volume is attached
    if volume.status != VolumeStatus::Attached {
        return Err(CliError::Internal(color_eyre::eyre::eyre!(
            "Volume '{}' is not currently attached to any rental.",
            volume.name
        )));
    }

    let rental_id = volume.rental_id.as_deref().unwrap_or("unknown");
    let rental_id_short = &rental_id[..8.min(rental_id.len())];

    // Confirm detachment if not skipped
    if !skip_confirm {
        let volume_name = volume.name.clone();

        println!(
            "{}",
            style(format!(
                "Warning: Detaching volume \"{}\" from rental {}...",
                volume_name, rental_id_short
            ))
            .yellow()
        );
        println!(
            "{}",
            style("Make sure no processes are using the volume before detaching.").yellow()
        );

        let confirmed = tokio::task::spawn_blocking(move || {
            let theme = ColorfulTheme::default();
            Confirm::with_theme(&theme)
                .with_prompt(format!("Detach volume '{}'?", volume_name))
                .default(false)
                .interact()
        })
        .await
        .map_err(|e| CliError::Internal(color_eyre::eyre::eyre!("Task join error: {}", e)))?
        .map_err(|e| CliError::Internal(e.into()))?;

        if !confirmed {
            println!("Detachment cancelled.");
            return Ok(());
        }
    }

    // Detach the volume with spinner
    let spinner = create_spinner(&format!("Detaching volume \"{}\"...", volume.name));
    let response = client.detach_volume(&volume.volume_id).await.map_err(|e| {
        complete_spinner_error(spinner.clone(), "Failed to detach volume");
        CliError::Api(e)
    })?;
    complete_spinner_and_clear(spinner);

    if json {
        json_output(&response)?;
        return Ok(());
    }

    print_success(&format!("Volume \"{}\" detached", volume.name));

    Ok(())
}
