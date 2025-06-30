//! Validator information route handlers

use crate::{
    api::types::{ListValidatorsResponse, ValidatorDetails},
    error::{Error, Result},
    server::AppState,
};
use axum::{
    extract::{Path, State},
    Json,
};

/// List validators
#[utoipa::path(
    get,
    path = "/api/v1/validators",
    responses(
        (status = 200, description = "List of validators", body = ListValidatorsResponse),
    ),
    tag = "validators",
)]
pub async fn list_validators(
    State(state): State<AppState>,
) -> Result<Json<ListValidatorsResponse>> {
    let validators = state.discovery.get_all_validators();

    let validator_details: Vec<ValidatorDetails> = validators
        .into_iter()
        .map(|v| ValidatorDetails {
            uid: v.uid,
            hotkey: v.hotkey,
            endpoint: v.endpoint,
            score: v.score,
            is_healthy: v.is_healthy,
            last_health_check: v.last_health_check,
        })
        .collect();

    let total_count = validator_details.len();

    Ok(Json(ListValidatorsResponse {
        validators: validator_details,
        total_count,
    }))
}

/// Get validator by ID
#[utoipa::path(
    get,
    path = "/api/v1/validators/{validator_id}",
    params(
        ("validator_id" = u16, Path, description = "Validator UID"),
    ),
    responses(
        (status = 200, description = "Validator details", body = ValidatorDetails),
        (status = 404, description = "Validator not found", body = crate::error::ErrorResponse),
    ),
    tag = "validators",
)]
pub async fn get_validator(
    State(state): State<AppState>,
    Path(validator_id): Path<u16>,
) -> Result<Json<ValidatorDetails>> {
    match state.discovery.get_validator(validator_id) {
        Some(validator) => Ok(Json(ValidatorDetails {
            uid: validator.uid,
            hotkey: validator.hotkey,
            endpoint: validator.endpoint,
            score: validator.score,
            is_healthy: validator.is_healthy,
            last_health_check: validator.last_health_check,
        })),
        None => Err(Error::NotFound {
            resource: format!("Validator {validator_id}"),
        }),
    }
}
