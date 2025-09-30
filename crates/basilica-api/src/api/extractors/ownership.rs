//! Ownership validation extractor for rental resources
//!
//! This extractor validates that the authenticated user owns the requested rental
//! before allowing access to rental-specific endpoints.

use axum::{
    async_trait,
    extract::{FromRequestParts, Path},
    http::{request::Parts, StatusCode},
};
use sqlx::{FromRow, PgPool};
use tracing::{debug, warn};

use crate::{api::middleware::AuthContext, server::AppState};

/// Database row structure for user_rentals table
#[derive(Debug, FromRow)]
struct UserRentalRow {
    rental_id: String,
    user_id: String,
    ssh_credentials: Option<String>,
    #[allow(dead_code)]
    created_at: chrono::DateTime<chrono::Utc>,
}

/// Extractor that validates rental ownership
///
/// This extractor ensures that the authenticated user owns the requested rental.
/// If the user doesn't own the rental, it returns 404 Not Found to avoid leaking
/// information about the existence of rentals owned by other users.
pub struct OwnedRental {
    pub rental_id: String,
    pub user_id: String,
    pub ssh_credentials: Option<String>,
}

#[async_trait]
impl FromRequestParts<AppState> for OwnedRental {
    type Rejection = StatusCode;

    async fn from_request_parts(
        parts: &mut Parts,
        state: &AppState,
    ) -> Result<Self, Self::Rejection> {
        // Extract the rental ID from the path
        let Path(rental_id): Path<String> = Path::from_request_parts(parts, state)
            .await
            .map_err(|_| StatusCode::BAD_REQUEST)?;

        // Get the authenticated user's context
        let auth_context = get_auth_context_from_parts(parts).ok_or(StatusCode::UNAUTHORIZED)?;

        let user_id = auth_context.user_id.clone();

        // Get rental ownership details from the database
        let rental_row = get_rental_ownership(&state.db, &rental_id, &user_id)
            .await
            .map_err(|e| {
                warn!("Database error checking rental ownership: {}", e);
                StatusCode::INTERNAL_SERVER_ERROR
            })?;

        match rental_row {
            Some(row) => {
                debug!(
                    "User {} authorized to access their rental {}",
                    user_id, rental_id
                );
                Ok(OwnedRental {
                    rental_id: row.rental_id,
                    user_id: row.user_id,
                    ssh_credentials: row.ssh_credentials,
                })
            }
            None => {
                warn!(
                    "User {} attempted to access rental {} which they don't own",
                    user_id, rental_id
                );
                // Return 404 to avoid leaking information about rental existence
                Err(StatusCode::NOT_FOUND)
            }
        }
    }
}

/// Get rental ownership details if user owns the rental
async fn get_rental_ownership(
    db: &PgPool,
    rental_id: &str,
    user_id: &str,
) -> Result<Option<UserRentalRow>, sqlx::Error> {
    let row = sqlx::query_as::<_, UserRentalRow>(
        r#"
        SELECT rental_id, user_id, ssh_credentials, created_at
        FROM user_rentals 
        WHERE rental_id = $1 AND user_id = $2
        "#,
    )
    .bind(rental_id)
    .bind(user_id)
    .fetch_optional(db)
    .await?;

    Ok(row)
}

/// Helper function to extract authentication context from request parts
fn get_auth_context_from_parts(parts: &Parts) -> Option<&AuthContext> {
    parts.extensions.get::<AuthContext>()
}

/// Store a new rental ownership record with optional SSH credentials
pub async fn store_rental_ownership(
    db: &PgPool,
    rental_id: &str,
    user_id: &str,
    ssh_credentials: Option<&str>,
) -> Result<(), sqlx::Error> {
    sqlx::query(
        r#"
        INSERT INTO user_rentals (rental_id, user_id, ssh_credentials)
        VALUES ($1, $2, $3)
        "#,
    )
    .bind(rental_id)
    .bind(user_id)
    .bind(ssh_credentials)
    .execute(db)
    .await?;

    debug!(
        "Stored ownership record for rental {} owned by user {} (SSH: {})",
        rental_id,
        user_id,
        ssh_credentials.is_some()
    );

    Ok(())
}

/// Get all rentals owned by a specific user
pub async fn get_user_rental_ids(db: &PgPool, user_id: &str) -> Result<Vec<String>, sqlx::Error> {
    let records: Vec<(String,)> = sqlx::query_as(
        r#"
        SELECT rental_id
        FROM user_rentals
        WHERE user_id = $1
        ORDER BY created_at DESC
        "#,
    )
    .bind(user_id)
    .fetch_all(db)
    .await?;

    Ok(records.into_iter().map(|(rental_id,)| rental_id).collect())
}

/// Struct to hold rental ID with SSH availability status
#[derive(Debug)]
pub struct RentalWithSshStatus {
    pub rental_id: String,
    pub has_ssh: bool,
}

/// Get all rentals owned by a specific user with SSH availability status
pub async fn get_user_rentals_with_ssh(
    db: &PgPool,
    user_id: &str,
) -> Result<Vec<RentalWithSshStatus>, sqlx::Error> {
    let records: Vec<(String, Option<String>)> = sqlx::query_as(
        r#"
        SELECT rental_id, ssh_credentials
        FROM user_rentals
        WHERE user_id = $1
        ORDER BY created_at DESC
        "#,
    )
    .bind(user_id)
    .fetch_all(db)
    .await?;

    Ok(records
        .into_iter()
        .map(|(rental_id, ssh_credentials)| RentalWithSshStatus {
            rental_id,
            has_ssh: ssh_credentials.is_some(),
        })
        .collect())
}

/// Structure for historical rental records
#[derive(Debug, FromRow)]
pub struct TerminatedUserRentalRow {
    pub rental_id: String,
    pub user_id: String,
    pub ssh_credentials: Option<String>,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub stopped_at: chrono::DateTime<chrono::Utc>,
    pub stop_reason: Option<String>,
}

/// Archive a rental ownership record to terminated_user_rentals table (when rental is stopped)
/// This preserves rental history instead of deleting it
pub async fn archive_rental_ownership(
    db: &PgPool,
    rental_id: &str,
    stop_reason: Option<&str>,
) -> Result<(), sqlx::Error> {
    // Use a transaction to ensure atomicity
    let mut tx = db.begin().await?;

    // First, copy the rental to terminated_user_rentals table
    sqlx::query(
        r#"
        INSERT INTO terminated_user_rentals (rental_id, user_id, ssh_credentials, created_at, stopped_at, stop_reason)
        SELECT rental_id, user_id, ssh_credentials, created_at, NOW(), $2
        FROM user_rentals
        WHERE rental_id = $1
        "#,
    )
    .bind(rental_id)
    .bind(stop_reason)
    .execute(&mut *tx)
    .await?;

    // Then delete from active rentals table
    sqlx::query(
        r#"
        DELETE FROM user_rentals
        WHERE rental_id = $1
        "#,
    )
    .bind(rental_id)
    .execute(&mut *tx)
    .await?;

    // Commit the transaction
    tx.commit().await?;

    debug!(
        "Archived ownership record for rental {} to terminated_user_rentals",
        rental_id
    );

    Ok(())
}

/// Get historical rentals for a specific user
#[cfg(test)]
pub async fn get_user_rental_history(
    db: &PgPool,
    user_id: &str,
    limit: Option<i64>,
) -> Result<Vec<TerminatedUserRentalRow>, sqlx::Error> {
    let records = if let Some(limit) = limit {
        sqlx::query_as::<_, TerminatedUserRentalRow>(
            r#"
            SELECT rental_id, user_id, ssh_credentials, created_at, stopped_at, stop_reason
            FROM terminated_user_rentals
            WHERE user_id = $1
            ORDER BY stopped_at DESC
            LIMIT $2
            "#,
        )
        .bind(user_id)
        .bind(limit)
        .fetch_all(db)
        .await?
    } else {
        sqlx::query_as::<_, TerminatedUserRentalRow>(
            r#"
            SELECT rental_id, user_id, ssh_credentials, created_at, stopped_at, stop_reason
            FROM terminated_user_rentals
            WHERE user_id = $1
            ORDER BY stopped_at DESC
            "#,
        )
        .bind(user_id)
        .fetch_all(db)
        .await?
    };

    Ok(records)
}

/// Get a specific historical rental by ID
#[cfg(test)]
pub async fn get_rental_history_by_id(
    db: &PgPool,
    rental_id: &str,
) -> Result<Option<TerminatedUserRentalRow>, sqlx::Error> {
    let record = sqlx::query_as::<_, TerminatedUserRentalRow>(
        r#"
        SELECT rental_id, user_id, ssh_credentials, created_at, stopped_at, stop_reason
        FROM terminated_user_rentals
        WHERE rental_id = $1
        "#,
    )
    .bind(rental_id)
    .fetch_optional(db)
    .await?;

    Ok(record)
}

/// Get all rental IDs (both active and historical) for a user
#[cfg(test)]
pub async fn get_all_user_rental_ids(
    db: &PgPool,
    user_id: &str,
) -> Result<Vec<String>, sqlx::Error> {
    let records: Vec<(String,)> = sqlx::query_as(
        r#"
        SELECT rental_id FROM (
            SELECT rental_id, created_at FROM user_rentals WHERE user_id = $1
            UNION ALL
            SELECT rental_id, created_at FROM terminated_user_rentals WHERE user_id = $1
        ) combined
        ORDER BY created_at DESC
        "#,
    )
    .bind(user_id)
    .fetch_all(db)
    .await?;

    Ok(records.into_iter().map(|(rental_id,)| rental_id).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore] // Requires PostgreSQL to be running
    async fn test_rental_ownership_archiving() {
        // Connect to test PostgreSQL database
        // This test requires DATABASE_URL to be set or PostgreSQL running locally
        let database_url = std::env::var("DATABASE_URL")
            .unwrap_or_else(|_| "postgres://basilica:dev@localhost:5432/basilica_test".to_string());

        let db = PgPool::connect(&database_url)
            .await
            .expect("Failed to connect to test database");

        // Run migration to create tables
        sqlx::migrate!("./migrations")
            .run(&db)
            .await
            .expect("Failed to run migrations");

        let rental_id = "test-rental-archive-123";
        let user_id = "user-archive-456";
        let ssh_creds = Some("ssh user@host -p 22");
        let stop_reason = Some("Test completion");

        // Initially, user should not own the rental
        assert!(get_rental_ownership(&db, rental_id, user_id)
            .await
            .expect("Failed to check ownership")
            .is_none());

        // Store ownership with SSH credentials
        store_rental_ownership(&db, rental_id, user_id, ssh_creds)
            .await
            .expect("Failed to store ownership");

        // Now user should own the rental and have SSH credentials
        let ownership = get_rental_ownership(&db, rental_id, user_id)
            .await
            .expect("Failed to check ownership");
        assert!(ownership.is_some());
        let row = ownership.unwrap();
        assert_eq!(row.ssh_credentials, ssh_creds.map(String::from));

        // Get user's active rentals
        let rentals = get_user_rental_ids(&db, user_id)
            .await
            .expect("Failed to get user rentals");
        assert_eq!(rentals, vec![rental_id]);

        // Archive ownership (instead of deleting)
        archive_rental_ownership(&db, rental_id, stop_reason)
            .await
            .expect("Failed to archive ownership");

        // User should no longer own the rental in active table
        assert!(get_rental_ownership(&db, rental_id, user_id)
            .await
            .expect("Failed to check ownership")
            .is_none());

        // User's active rentals should be empty
        let active_rentals = get_user_rental_ids(&db, user_id)
            .await
            .expect("Failed to get user rentals");
        assert!(active_rentals.is_empty());

        // Check rental exists in history
        let history = get_user_rental_history(&db, user_id, None)
            .await
            .expect("Failed to get rental history");
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].rental_id, rental_id);
        assert_eq!(history[0].user_id, user_id);
        assert_eq!(history[0].ssh_credentials, ssh_creds.map(String::from));
        assert_eq!(history[0].stop_reason, stop_reason.map(String::from));

        // Check get by ID in history
        let history_by_id = get_rental_history_by_id(&db, rental_id)
            .await
            .expect("Failed to get rental history by ID");
        assert!(history_by_id.is_some());
        let history_row = history_by_id.unwrap();
        assert_eq!(history_row.rental_id, rental_id);
        assert_eq!(history_row.stop_reason, stop_reason.map(String::from));

        // Check all rental IDs (active + historical)
        let all_rentals = get_all_user_rental_ids(&db, user_id)
            .await
            .expect("Failed to get all user rentals");
        assert_eq!(all_rentals, vec![rental_id]);
    }
}
