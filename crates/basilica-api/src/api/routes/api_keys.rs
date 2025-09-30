//! API Key management route handlers

use crate::{
    api::{
        auth::api_keys::{self},
        middleware::AuthContext,
    },
    error::{ApiError, Result},
    server::AppState,
};
use axum::{extract::State, Json};
use basilica_common::{ApiKeyName, ApiKeyNameError};
use serde::{Deserialize, Serialize};
use tracing::{debug, info, instrument, warn};

/// Request to create a new API key
#[derive(Debug, Deserialize)]
pub struct CreateKeyRequest {
    /// Name for the API key
    pub name: ApiKeyName,

    /// Optional scopes for the API key
    pub scopes: Option<Vec<String>>,
}

/// Response after creating a new API key
#[derive(Debug, Serialize)]
pub struct CreateKeyResponse {
    /// Name of the key
    pub name: String,

    /// Creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,

    /// The full API key token (only returned once at creation)
    pub token: String,
}

/// List item for API keys
#[derive(Debug, Serialize)]
pub struct ListKeyItem {
    /// Key identifier (kid)
    pub kid: String,

    /// Name of the key
    pub name: String,

    /// Creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,

    /// Last usage timestamp
    pub last_used_at: Option<chrono::DateTime<chrono::Utc>>,
}

/// Create a new API key
///
/// This endpoint requires JWT authentication (human users only).
/// Only one active key per user is allowed.
#[instrument(skip(state, auth_context, request), fields(user_id = %auth_context.user_id, key_name = %request.name))]
pub async fn create_key(
    State(state): State<AppState>,
    axum::Extension(auth_context): axum::Extension<AuthContext>,
    Json(request): Json<CreateKeyRequest>,
) -> Result<Json<CreateKeyResponse>> {
    // Require JWT authentication for key management
    if !auth_context.is_jwt() {
        warn!(
            "User {} attempted to create API key with non-JWT auth",
            auth_context.user_id
        );
        return Err(ApiError::Authentication {
            message: "API key management requires human authentication (JWT)".to_string(),
        });
    }

    // ApiKeyName is already validated by its type

    info!("Creating API key");

    // Optional: Check for maximum number of keys per user (e.g., 10)
    const MAX_KEYS_PER_USER: usize = 10;
    let existing_keys = api_keys::list_user_api_keys(&state.db, &auth_context.user_id)
        .await
        .map_err(|e| ApiError::Internal {
            message: format!("Failed to check existing keys: {}", e),
        })?;

    if existing_keys.len() >= MAX_KEYS_PER_USER {
        return Err(ApiError::Conflict {
            message: format!(
                "Maximum number of API keys ({}) reached. Please delete an existing key first.",
                MAX_KEYS_PER_USER
            ),
        });
    }

    // Generate new API key
    let generated_key = api_keys::GeneratedApiKey::generate();
    let kid = generated_key.kid_hex();
    let hash = generated_key
        .hash_string()
        .map_err(|e| ApiError::Internal {
            message: format!("Failed to generate API key hash: {}", e),
        })?;
    let token = generated_key.into_token();
    let display_token = token.to_string();

    // Use provided scopes or default to user's current scopes
    let scopes = request
        .scopes
        .unwrap_or_else(|| auth_context.scopes.clone());

    // Store in database
    let key = api_keys::insert_api_key(
        &state.db,
        &auth_context.user_id,
        request.name.as_str(),
        &kid,
        &hash,
        &scopes,
    )
    .await
    .map_err(|e| match e {
        api_keys::ApiKeyError::Database(ref sqlx_err) => {
            // Check if it's a unique constraint violation (PostgreSQL error code 23505)
            if let sqlx::Error::Database(ref db_err) = sqlx_err {
                if db_err.code().as_deref() == Some("23505") {
                    return ApiError::Conflict {
                        message: format!("API key with name '{}' already exists", request.name),
                    };
                }
            }
            ApiError::Internal {
                message: format!("Failed to store API key: {}", e),
            }
        }
        _ => ApiError::Internal {
            message: format!("Failed to store API key: {}", e),
        },
    })?;

    debug!("Successfully created API key");

    Ok(Json(CreateKeyResponse {
        name: key.name,
        created_at: key.created_at,
        token: display_token,
    }))
}

/// List all API keys for the authenticated user
///
/// This endpoint requires JWT authentication (human users only).
#[instrument(skip(state, auth_context), fields(user_id = %auth_context.user_id))]
pub async fn list_keys(
    State(state): State<AppState>,
    axum::Extension(auth_context): axum::Extension<AuthContext>,
) -> Result<Json<Vec<ListKeyItem>>> {
    // Require JWT authentication for key management
    if !auth_context.is_jwt() {
        warn!(
            "User {} attempted to list API keys with non-JWT auth",
            auth_context.user_id
        );
        return Err(ApiError::Authentication {
            message: "API key management requires human authentication (JWT)".to_string(),
        });
    }

    debug!("Listing API keys");

    let keys = api_keys::list_user_api_keys(&state.db, &auth_context.user_id)
        .await
        .map_err(|e| ApiError::Internal {
            message: format!("Failed to list API keys: {}", e),
        })?;

    let items: Vec<ListKeyItem> = keys
        .into_iter()
        .map(|key| ListKeyItem {
            kid: key.kid.clone(),
            name: key.name,
            created_at: key.created_at,
            last_used_at: key.last_used_at,
        })
        .collect();

    debug!("Found {} API keys", items.len());

    Ok(Json(items))
}

/// Delete an API key by name
///
/// This endpoint requires JWT authentication (human users only).
#[instrument(skip(state, auth_context), fields(user_id = %auth_context.user_id, key_name = %name))]
pub async fn revoke_key(
    State(state): State<AppState>,
    axum::Extension(auth_context): axum::Extension<AuthContext>,
    axum::extract::Path(name): axum::extract::Path<String>,
) -> Result<()> {
    // Require JWT authentication for key management
    if !auth_context.is_jwt() {
        warn!(
            "User {} attempted to delete API key with non-JWT auth",
            auth_context.user_id
        );
        return Err(ApiError::Authentication {
            message: "API key management requires human authentication (JWT)".to_string(),
        });
    }

    // Validate API key name
    let api_key_name = ApiKeyName::new(name.clone()).map_err(|e| match e {
        ApiKeyNameError::Empty => ApiError::BadRequest {
            message: "API key name cannot be empty".to_string(),
        },
        ApiKeyNameError::TooLong => ApiError::BadRequest {
            message: "API key name too long (max 100 characters)".to_string(),
        },
        ApiKeyNameError::InvalidCharacters => ApiError::BadRequest {
            message: "Invalid API key name. Only alphanumeric characters, hyphens, and underscores are allowed".to_string(),
        },
    })?;

    info!("Deleting API key");

    let deleted =
        api_keys::delete_api_key_by_name(&state.db, &auth_context.user_id, api_key_name.as_str())
            .await
            .map_err(|e| ApiError::Internal {
                message: format!("Failed to delete API key: {}", e),
            })?;

    if !deleted {
        return Err(ApiError::NotFound {
            message: format!("API key with name '{}' not found", name),
        });
    }

    debug!("Successfully deleted API key");

    Ok(())
}
