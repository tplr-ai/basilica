//! SSH Key management route handlers

use crate::{
    api::middleware::AuthContext,
    error::{ApiError, Result},
    server::AppState,
};
use axum::{extract::State, http::StatusCode, Json};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::Row;
use ssh_key::PublicKey;
use tracing::{debug, info, instrument, warn};
use uuid::Uuid;

/// User SSH key (full record with public_key)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SshKey {
    pub id: String,
    pub user_id: String,
    pub name: String,
    pub public_key: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// SSH key response (includes public_key for local key matching)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SshKeyResponse {
    pub id: String,
    pub user_id: String,
    pub name: String,
    pub public_key: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl From<SshKey> for SshKeyResponse {
    fn from(key: SshKey) -> Self {
        Self {
            id: key.id,
            user_id: key.user_id,
            name: key.name,
            public_key: key.public_key,
            created_at: key.created_at,
            updated_at: key.updated_at,
        }
    }
}

/// Request to register SSH key
#[derive(Debug, Deserialize)]
pub struct RegisterSshKeyRequest {
    pub name: String,
    pub public_key: String,
}

/// Register a new SSH key for the authenticated user
///
/// This endpoint requires JWT authentication.
/// Only one SSH key per user is allowed (enforced by database UNIQUE constraint on user_id).
#[instrument(skip(state, auth_context, request), fields(user_id = %auth_context.user_id, key_name = %request.name))]
pub async fn register_ssh_key(
    State(state): State<AppState>,
    axum::Extension(auth_context): axum::Extension<AuthContext>,
    Json(request): Json<RegisterSshKeyRequest>,
) -> Result<(StatusCode, Json<SshKeyResponse>)> {
    // Require JWT authentication for SSH key management
    if !auth_context.is_jwt() {
        warn!(
            "User {} attempted to register SSH key with non-JWT auth",
            auth_context.user_id
        );
        return Err(ApiError::Authentication {
            message: "SSH key management requires human authentication (JWT)".to_string(),
        });
    }

    info!("Registering SSH key");

    // Validate SSH key name (basic validation)
    if request.name.trim().is_empty() {
        return Err(ApiError::BadRequest {
            message: "SSH key name cannot be empty".to_string(),
        });
    }

    if request.name.len() > 100 {
        return Err(ApiError::BadRequest {
            message: "SSH key name must be 100 characters or less".to_string(),
        });
    }

    // Validate SSH public key format (basic validation)
    if request.public_key.trim().is_empty() {
        return Err(ApiError::BadRequest {
            message: "SSH public key cannot be empty".to_string(),
        });
    }

    // Basic SSH public key format check
    let trimmed_key = request.public_key.trim();
    if PublicKey::from_openssh(trimmed_key).is_err() {
        return Err(ApiError::BadRequest {
            message: "Invalid SSH public key format".to_string(),
        });
    }

    // Check if user already has an SSH key
    let existing = sqlx::query("SELECT id FROM ssh_keys WHERE user_id = $1")
        .bind(&auth_context.user_id)
        .fetch_optional(&state.db)
        .await
        .map_err(|e| ApiError::Internal {
            message: format!("Failed to check existing SSH key: {}", e),
        })?;

    if existing.is_some() {
        return Err(ApiError::Conflict {
            message:
                "User already has an SSH key registered. Please delete the existing key first."
                    .to_string(),
        });
    }

    // Create SSH key record
    let now = Utc::now();
    let id = Uuid::new_v4().to_string();

    sqlx::query(
        r#"
        INSERT INTO ssh_keys (id, user_id, name, public_key, created_at, updated_at)
        VALUES ($1, $2, $3, $4, $5, $6)
        "#,
    )
    .bind(&id)
    .bind(&auth_context.user_id)
    .bind(&request.name)
    .bind(trimmed_key)
    .bind(now)
    .bind(now)
    .execute(&state.db)
    .await
    .map_err(|e| {
        if e.to_string().contains("UNIQUE constraint") {
            ApiError::Conflict {
                message: "User already has an SSH key registered".to_string(),
            }
        } else {
            ApiError::Internal {
                message: format!("Failed to register SSH key: {}", e),
            }
        }
    })?;

    debug!("SSH key registered successfully");

    let ssh_key = SshKey {
        id: id.clone(),
        user_id: auth_context.user_id.clone(),
        name: request.name,
        public_key: trimmed_key.to_string(),
        created_at: now,
        updated_at: now,
    };

    Ok((StatusCode::CREATED, Json(ssh_key.into())))
}

/// Get the authenticated user's SSH key
///
/// This endpoint requires JWT authentication.
#[instrument(skip(state, auth_context), fields(user_id = %auth_context.user_id))]
pub async fn get_ssh_key(
    State(state): State<AppState>,
    axum::Extension(auth_context): axum::Extension<AuthContext>,
) -> Result<Json<Option<SshKeyResponse>>> {
    // Require JWT authentication for SSH key management
    if !auth_context.is_jwt() {
        warn!(
            "User {} attempted to get SSH key with non-JWT auth",
            auth_context.user_id
        );
        return Err(ApiError::Authentication {
            message: "SSH key management requires human authentication (JWT)".to_string(),
        });
    }

    debug!("Getting SSH key for user");

    let row = sqlx::query(
        r#"
        SELECT id, user_id, name, public_key, created_at, updated_at
        FROM ssh_keys
        WHERE user_id = $1
        "#,
    )
    .bind(&auth_context.user_id)
    .fetch_optional(&state.db)
    .await
    .map_err(|e| ApiError::Internal {
        message: format!("Failed to get SSH key: {}", e),
    })?;

    let ssh_key_response = row.map(|r| SshKeyResponse {
        id: r.get("id"),
        user_id: r.get("user_id"),
        name: r.get("name"),
        public_key: r.get("public_key"),
        created_at: r.get("created_at"),
        updated_at: r.get("updated_at"),
    });

    Ok(Json(ssh_key_response))
}

/// Delete the authenticated user's SSH key
///
/// This endpoint requires JWT authentication.
/// Also removes all provider-specific SSH key registrations.
#[instrument(skip(state, auth_context), fields(user_id = %auth_context.user_id))]
pub async fn delete_ssh_key(
    State(state): State<AppState>,
    axum::Extension(auth_context): axum::Extension<AuthContext>,
) -> Result<StatusCode> {
    // Require JWT authentication for SSH key management
    if !auth_context.is_jwt() {
        warn!(
            "User {} attempted to delete SSH key with non-JWT auth",
            auth_context.user_id
        );
        return Err(ApiError::Authentication {
            message: "SSH key management requires human authentication (JWT)".to_string(),
        });
    }

    info!("Deleting SSH key for user");

    // Check if SSH key exists
    let row = sqlx::query("SELECT id FROM ssh_keys WHERE user_id = $1")
        .bind(&auth_context.user_id)
        .fetch_optional(&state.db)
        .await
        .map_err(|e| ApiError::Internal {
            message: format!("Failed to check SSH key: {}", e),
        })?;

    let Some(row) = row else {
        return Err(ApiError::NotFound {
            message: "SSH key not found".to_string(),
        });
    };

    let ssh_key_id: String = row.get("id");

    // Start transaction to delete SSH key and provider keys
    let mut tx = state.db.begin().await.map_err(|e| ApiError::Internal {
        message: format!("Failed to start transaction: {}", e),
    })?;

    // Delete provider-specific SSH keys (cascade cleanup)
    sqlx::query("DELETE FROM provider_ssh_keys WHERE ssh_key_id = $1")
        .bind(&ssh_key_id)
        .execute(&mut *tx)
        .await
        .map_err(|e| ApiError::Internal {
            message: format!("Failed to delete provider SSH keys: {}", e),
        })?;

    // Delete the SSH key
    sqlx::query("DELETE FROM ssh_keys WHERE id = $1")
        .bind(&ssh_key_id)
        .execute(&mut *tx)
        .await
        .map_err(|e| ApiError::Internal {
            message: format!("Failed to delete SSH key: {}", e),
        })?;

    tx.commit().await.map_err(|e| ApiError::Internal {
        message: format!("Failed to commit transaction: {}", e),
    })?;

    debug!("SSH key deleted successfully");

    Ok(StatusCode::NO_CONTENT)
}
