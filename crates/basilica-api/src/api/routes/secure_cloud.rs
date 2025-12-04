//! Secure cloud (GPU aggregator) route handlers
//! These routes proxy requests to the aggregator service

use crate::api::extractors::ownership::archive_secure_cloud_rental;
use crate::api::middleware::AuthContext;
use crate::api::query::GpuPriceQuery;
use crate::error::ApiError;
use crate::server::AppState;
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::IntoResponse,
    Extension, Json,
};
use basilica_sdk::types::{
    ListSecureCloudRentalsResponse, SecureCloudRentalListItem, SecureCloudRentalResponse,
    StartSecureCloudRentalRequest, StopSecureCloudRentalResponse,
};
use rust_decimal::prelude::{FromPrimitive, ToPrimitive};
use rust_decimal::Decimal;
use serde_json::json;

/// Type alias for secure cloud rental query result from database
type RentalQueryRow = (
    String,                                // id
    String,                                // provider
    Option<String>,                        // provider_instance_id
    String,                                // offering_id
    String,                                // instance_type
    Option<String>,                        // location_code
    String,                                // status
    Option<String>,                        // ip_address
    chrono::DateTime<chrono::Utc>,         // created_at
    Option<chrono::DateTime<chrono::Utc>>, // stopped_at
    Option<i32>,                           // vcpu_count (from gpu_offerings)
    Option<i32>,                           // system_memory_gb (from gpu_offerings)
    Option<String>,                        // region (from gpu_offerings)
);

/// Get SSH public key for user and validate ownership
async fn get_ssh_key_for_user(
    pool: &sqlx::PgPool,
    ssh_key_id: &str,
    user_id: &str,
) -> Result<String, anyhow::Error> {
    let row: (String,) =
        sqlx::query_as("SELECT public_key FROM ssh_keys WHERE id = $1 AND user_id = $2")
            .bind(ssh_key_id)
            .bind(user_id)
            .fetch_optional(pool)
            .await?
            .ok_or_else(|| anyhow::anyhow!("SSH key not found or unauthorized"))?;

    Ok(row.0)
}

/// List GPU prices from aggregator service
/// This is a thin proxy to the aggregator's get_gpu_prices handler
pub async fn list_gpu_prices(
    State(state): State<AppState>,
    Query(query): Query<GpuPriceQuery>,
) -> impl IntoResponse {
    match state.aggregator_service.get_offerings().await {
        Ok(mut offerings) => {
            // Apply filters (same logic as aggregator handler)
            if let Some(gpu_type) = query.gpu_type() {
                offerings.retain(|o| o.gpu_type == gpu_type);
            }

            if let Some(region) = &query.region {
                let region_lower = region.to_lowercase();
                offerings.retain(|o| o.region.to_lowercase().contains(&region_lower));
            }

            if let Some(provider) = query.provider() {
                offerings.retain(|o| o.provider == provider);
            }

            if let Some(min_price) = query.min_price() {
                offerings.retain(|o| o.hourly_rate_per_gpu >= min_price);
            }

            if let Some(max_price) = query.max_price() {
                offerings.retain(|o| o.hourly_rate_per_gpu <= max_price);
            }

            if query.available_only.unwrap_or(false) {
                offerings.retain(|o| o.availability);
            }

            // Sort results
            match query.sort_by.as_deref() {
                Some("price") => offerings.sort_by_key(|o| o.hourly_rate_per_gpu),
                Some("gpu_type") => offerings.sort_by_key(|o| o.gpu_type.as_str().to_string()),
                Some("region") => offerings.sort_by(|a, b| a.region.cmp(&b.region)),
                _ => {}
            }

            // Apply secure cloud markup to prices
            let markup_multiplier = rust_decimal::Decimal::from_f64(
                1.0 + (state.pricing_config.secure_cloud_markup_percent / 100.0),
            )
            .unwrap_or(rust_decimal::Decimal::ONE);
            for offering in &mut offerings {
                offering.hourly_rate_per_gpu *= markup_multiplier;
            }

            // raw_metadata is automatically excluded via #[serde(skip)]
            let total_count = offerings.len();

            (
                StatusCode::OK,
                Json(json!({
                    "nodes": offerings,
                    "count": total_count,
                })),
            )
        }
        Err(e) => {
            tracing::error!("Failed to get offerings: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({
                    "error": "Failed to fetch GPU prices",
                    "message": e.to_string()
                })),
            )
        }
    }
}

/// List secure cloud rentals for the authenticated user
pub async fn list_secure_cloud_rentals(
    State(state): State<AppState>,
    Extension(auth): Extension<AuthContext>,
) -> Result<impl IntoResponse, ApiError> {
    // 1. Query secure_cloud_rentals table for this user, JOIN with gpu_offerings for resource specs
    let rentals: Vec<RentalQueryRow> = sqlx::query_as(
        "SELECT r.id, r.provider, r.provider_instance_id, r.offering_id, r.instance_type, \
         r.location_code, r.status, r.ip_address, r.created_at, r.stopped_at, \
         o.vcpu_count, o.system_memory_gb, o.region \
         FROM secure_cloud_rentals r \
         LEFT JOIN gpu_offerings o ON r.offering_id = o.id \
         WHERE r.user_id = $1 \
         ORDER BY r.created_at DESC",
    )
    .bind(&auth.user_id)
    .fetch_all(&state.db)
    .await
    .map_err(|e| {
        tracing::error!("Failed to query secure_cloud_rentals: {}", e);
        ApiError::Internal {
            message: "Failed to fetch rentals".to_string(),
        }
    })?;

    // 2. Get offerings from aggregator to enrich with GPU details
    let offerings = state
        .aggregator_service
        .get_offerings()
        .await
        .unwrap_or_else(|e| {
            tracing::warn!("Failed to fetch offerings, using empty list: {}", e);
            vec![]
        });

    // Create a map of offering_id -> GpuOffering for fast lookup
    let offerings_map: std::collections::HashMap<_, _> =
        offerings.iter().map(|o| (o.id.as_str(), o)).collect();

    // 3. Get accumulated costs from billing service
    let cost_map: std::collections::HashMap<String, String> =
        if let Some(ref billing_client) = state.billing_client {
            match billing_client
                .get_active_rentals_for_user(&auth.user_id, None, None)
                .await
            {
                Ok(response) => response
                    .rentals
                    .into_iter()
                    .map(|r| (r.rental_id, r.current_cost))
                    .collect(),
                Err(e) => {
                    tracing::warn!("Failed to fetch billing costs, will use None: {}", e);
                    std::collections::HashMap::new()
                }
            }
        } else {
            std::collections::HashMap::new()
        };

    // 4. Build response items by joining rental data with offering data
    let rental_items: Vec<SecureCloudRentalListItem> = rentals
        .into_iter()
        .map(
            |(
                rental_id,
                provider,
                provider_instance_id,
                offering_id,
                instance_type,
                location_code,
                status,
                ip_address,
                created_at,
                stopped_at,
                db_vcpu_count,
                db_system_memory_gb,
                db_region,
            )| {
                let markup_multiplier =
                    1.0 + (state.pricing_config.secure_cloud_markup_percent / 100.0);

                // Try to find the offering to get GPU details and pricing
                let (gpu_type, gpu_count, hourly_cost) = if let Some(offering) =
                    offerings_map.get(offering_id.as_str())
                {
                    let base_rate = offering.hourly_rate_per_gpu.to_f64().unwrap_or(0.0);
                    let gpu_count = offering.gpu_count;
                    // Total hourly cost = per-GPU price × markup × number of GPUs
                    let hourly_cost = base_rate * markup_multiplier * f64::from(gpu_count.max(1));

                    (offering.gpu_type.to_string(), gpu_count, hourly_cost)
                } else {
                    // Fallback if offering not found (e.g., offering expired)
                    tracing::warn!(
                        "Offering {} not found for rental {}, using defaults",
                        offering_id,
                        rental_id
                    );
                    ("unknown".to_string(), 0, 0.0)
                };

                // Use resource specs from database JOIN (already available)
                let vcpu_count = db_vcpu_count.map(|v| v as u32);
                let system_memory_gb = db_system_memory_gb.map(|v| v as u32);

                // Prefer location_code from rental table, fallback to region from offering
                let final_location_code = location_code.or(db_region);

                // Generate SSH command if IP available
                let ssh_command = ip_address.as_ref().map(|ip| format!("ssh ubuntu@{}", ip));

                // Get accumulated cost from billing service
                let accumulated_cost = cost_map.get(&rental_id).cloned();

                SecureCloudRentalListItem {
                    rental_id,
                    provider,
                    provider_instance_id,
                    gpu_type,
                    gpu_count,
                    instance_type,
                    location_code: final_location_code,
                    status,
                    ip_address,
                    hourly_cost,
                    created_at,
                    stopped_at,
                    ssh_command,
                    vcpu_count,
                    system_memory_gb,
                    accumulated_cost,
                }
            },
        )
        .collect();

    let total_count = rental_items.len();

    Ok((
        StatusCode::OK,
        Json(ListSecureCloudRentalsResponse {
            rentals: rental_items,
            total_count,
        }),
    ))
}

/// Start a secure cloud rental (direct datacenter provisioning)
pub async fn start_secure_cloud_rental(
    State(state): State<AppState>,
    Extension(auth): Extension<AuthContext>,
    Json(request): Json<StartSecureCloudRentalRequest>,
) -> Result<impl IntoResponse, ApiError> {
    // 1. Validate SSH key ownership
    let _public_key = get_ssh_key_for_user(&state.db, &request.ssh_public_key_id, &auth.user_id)
        .await
        .map_err(|e| {
            tracing::error!("SSH key lookup failed: {}", e);
            ApiError::BadRequest {
                message: "Invalid SSH key".to_string(),
            }
        })?;

    // 2. Get offering to extract pricing
    let offerings = state
        .aggregator_service
        .get_offerings()
        .await
        .map_err(|e| {
            tracing::error!("Failed to get offerings: {}", e);
            ApiError::Internal {
                message: "Failed to fetch GPU offerings".to_string(),
            }
        })?;

    let offering = offerings
        .iter()
        .find(|o| o.id == request.offering_id)
        .ok_or_else(|| ApiError::NotFound {
            message: format!("Offering {} not found", request.offering_id),
        })?;

    // 2.5. Validate user has sufficient balance before creating rental
    if let Some(billing_client) = &state.billing_client {
        let hourly_cost = offering.hourly_rate_per_gpu * Decimal::from(offering.gpu_count.max(1));
        crate::api::middleware::validate_balance_for_rental(
            billing_client,
            &auth.user_id,
            hourly_cost,
        )
        .await?;
    }

    // 3. Deploy via aggregator (which creates the rental/deployment record)
    let deployment = state
        .aggregator_service
        .deploy_instance(
            request.offering_id.clone(),
            request.ssh_public_key_id.clone(),
            Some(offering.region.clone()), // Pass region from offering
        )
        .await
        .map_err(|e| {
            tracing::error!("Deployment failed: {}", e);
            ApiError::Internal {
                message: "Failed to deploy instance".to_string(),
            }
        })?;

    let rental_id = deployment.id.clone();
    let provider_instance_id = deployment.provider_instance_id.clone().unwrap_or_default();

    // 4. Register with billing service
    use basilica_protocol::billing::{
        track_rental_request::CloudType, SecureCloudData, TrackRentalRequest,
    };

    let resource_spec = Some(basilica_protocol::billing::ResourceSpec {
        cpu_cores: offering.vcpu_count,
        memory_mb: offering.system_memory_gb as u64 * 1024,
        gpus: vec![basilica_protocol::billing::GpuSpec {
            model: offering.gpu_type.to_string(),
            memory_mb: 0, // Not provided by offerings
            count: offering.gpu_count,
        }],
        disk_gb: 0, // Storage not provided in standardized format
        network_bandwidth_mbps: 0,
    });

    let timestamp = prost_types::Timestamp::from(std::time::SystemTime::now());

    // Apply markup to the per-GPU price before sending to billing
    // Note: offering.hourly_rate_per_gpu is already normalized to per-GPU rate by the aggregator
    let markup_multiplier =
        Decimal::from_f64(1.0 + (state.pricing_config.secure_cloud_markup_percent / 100.0))
            .unwrap_or(Decimal::ONE);
    let base_price_per_gpu = (offering.hourly_rate_per_gpu * markup_multiplier)
        .to_f64()
        .unwrap_or(0.0);

    let track_request = TrackRentalRequest {
        rental_id: rental_id.clone(),
        user_id: auth.user_id.clone(),
        resource_spec,
        start_time: Some(timestamp),
        metadata: std::collections::HashMap::new(),
        cloud_type: Some(CloudType::Secure(SecureCloudData {
            provider_instance_id,
            provider: offering.provider.to_string(),
            offering_id: request.offering_id.clone(),
            base_price_per_gpu,
            gpu_count: offering.gpu_count,
        })),
    };

    if let Some(ref billing_client) = state.billing_client {
        if let Err(e) = billing_client.track_rental(track_request).await {
            tracing::error!("Failed to register with billing: {}", e);

            // Rollback: delete the deployed instance since billing registration failed
            // This prevents orphaned instances running without billing tracking
            if let Err(rollback_err) = state.aggregator_service.delete_deployment(&rental_id).await
            {
                tracing::error!(
                    "CRITICAL: Failed to rollback deployment {} after billing failure: {}. Manual cleanup required.",
                    rental_id, rollback_err
                );
            } else {
                tracing::info!(
                    "Successfully rolled back deployment {} after billing registration failure",
                    rental_id
                );
            }

            return Err(ApiError::Internal {
                message: "Failed to register rental with billing service".to_string(),
            });
        }
    }

    // 5. Calculate hourly cost (total per-instance price with markup)
    // hourly_cost = base_price_per_gpu * gpu_count
    let hourly_cost = base_price_per_gpu * f64::from(offering.gpu_count.max(1));

    // 6. Return response
    Ok((
        StatusCode::CREATED,
        Json(SecureCloudRentalResponse {
            rental_id: rental_id.clone(),
            deployment_id: rental_id, // Same as rental_id now (consolidated)
            provider: offering.provider.to_string(),
            status: deployment.status.to_string(),
            ip_address: deployment.ip_address.clone(),
            ssh_command: deployment.ip_address.map(|ip| format!("ssh ubuntu@{}", ip)),
            hourly_cost,
        }),
    ))
}

/// Stop a secure cloud rental and calculate final cost
pub async fn stop_secure_cloud_rental(
    State(state): State<AppState>,
    Extension(auth): Extension<AuthContext>,
    Path(rental_id): Path<String>,
) -> Result<impl IntoResponse, ApiError> {
    use chrono::Utc;

    // 1. Get rental and verify ownership
    let rental: (String, chrono::DateTime<Utc>) = sqlx::query_as(
        "SELECT user_id, created_at
         FROM secure_cloud_rentals
         WHERE id = $1",
    )
    .bind(&rental_id)
    .fetch_one(&state.db)
    .await
    .map_err(|e| match e {
        sqlx::Error::RowNotFound => ApiError::NotFound {
            message: format!("Rental {} not found", rental_id),
        },
        _ => ApiError::Internal {
            message: "Database error".to_string(),
        },
    })?;

    if rental.0 != auth.user_id {
        return Err(ApiError::Authorization {
            message: "Not authorized to stop this rental".to_string(),
        });
    }

    let stop_time = Utc::now();
    let duration = stop_time.signed_duration_since(rental.1);
    let duration_hours = duration.num_seconds() as f64 / 3600.0;

    // 2. Delete deployment via aggregator FIRST (rental_id IS the deployment_id)
    // This must happen before billing finalization to ensure consistent state.
    // If the provider returns NotFound (VM deleted externally), we proceed anyway
    // since the desired outcome (VM gone) is achieved.
    let was_deleted_externally = match state.aggregator_service.delete_deployment(&rental_id).await
    {
        Ok(()) => {
            tracing::info!("Deployment {} deleted successfully", rental_id);
            false
        }
        Err(basilica_aggregator::AggregatorError::NotFound(msg)) => {
            tracing::warn!(
                "Deployment {} not found at provider (deleted externally): {}. Proceeding with billing finalization.",
                rental_id, msg
            );
            true
        }
        Err(e) => {
            tracing::error!("Failed to delete deployment {}: {}", rental_id, e);
            return Err(ApiError::Internal {
                message: "Failed to stop instance".to_string(),
            });
        }
    };

    // Determine termination reason and final status based on how the VM was stopped
    let (termination_reason, final_status, stop_reason) = if was_deleted_externally {
        (
            "vm_deleted_externally",
            "deleted",
            "VM not found at provider (already deleted externally)",
        )
    } else {
        ("user_requested", "stopped", "User requested stop")
    };

    // 3. Finalize rental in billing service (get accumulated cost)
    let total_cost = if let Some(billing_client) = &state.billing_client {
        use basilica_protocol::billing::FinalizeRentalRequest;
        use prost_types::Timestamp;

        let end_timestamp = Timestamp {
            seconds: stop_time.timestamp(),
            nanos: stop_time.timestamp_subsec_nanos() as i32,
        };

        let finalize_request = FinalizeRentalRequest {
            rental_id: rental_id.clone(),
            end_time: Some(end_timestamp),
            termination_reason: termination_reason.to_string(),
        };

        match billing_client.finalize_rental(finalize_request).await {
            Ok(response) => {
                // Parse total_cost from decimal string
                response.total_cost.parse::<f64>().unwrap_or(0.0)
            }
            Err(e) => {
                tracing::warn!("Failed to finalize rental in billing service: {}", e);
                0.0 // Fallback if billing fails
            }
        }
    } else {
        tracing::warn!("Billing client not available, cannot get final cost");
        0.0
    };

    // 4. Archive rental to terminated_secure_cloud_rentals table
    if let Err(e) =
        archive_secure_cloud_rental(&state.db, &rental_id, Some(stop_reason), Some(final_status))
            .await
    {
        tracing::error!("Failed to archive secure cloud rental: {}", e);
    }

    tracing::info!(
        "Rental {} stopped. Duration: {} hours, Total cost: ${}",
        rental_id,
        duration_hours,
        total_cost
    );

    Ok((
        StatusCode::OK,
        Json(StopSecureCloudRentalResponse {
            rental_id,
            status: "stopped".to_string(),
            duration_hours,
            total_cost,
        }),
    ))
}
