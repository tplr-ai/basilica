use basilica_validator::persistence::SimplePersistence;
use sqlx::Row;

#[tokio::test]
async fn test_network_profile_table_exists() -> Result<(), anyhow::Error> {
    // Create an in-memory database
    let persistence = SimplePersistence::new(":memory:", "test_validator".to_string()).await?;
    persistence.run_migrations().await?;

    // Check if the table exists
    let row = sqlx::query(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='node_network_profile'",
    )
    .fetch_optional(persistence.pool())
    .await?;

    assert!(row.is_some(), "node_network_profile table should exist");

    // Check the columns exist
    let columns = sqlx::query("PRAGMA table_info(node_network_profile)")
        .fetch_all(persistence.pool())
        .await?;

    let column_names: Vec<String> = columns.iter().map(|col| col.get("name")).collect();

    // Verify all expected columns exist
    assert!(column_names.contains(&"miner_uid".to_string()));
    assert!(column_names.contains(&"node_id".to_string()));
    assert!(column_names.contains(&"ip_address".to_string()));
    assert!(column_names.contains(&"hostname".to_string()));
    assert!(column_names.contains(&"city".to_string()));
    assert!(column_names.contains(&"region".to_string()));
    assert!(column_names.contains(&"country".to_string()));
    assert!(column_names.contains(&"location".to_string()));
    assert!(column_names.contains(&"organization".to_string()));
    assert!(column_names.contains(&"postal_code".to_string()));
    assert!(column_names.contains(&"timezone".to_string()));
    assert!(column_names.contains(&"test_timestamp".to_string()));
    assert!(column_names.contains(&"full_result_json".to_string()));
    assert!(column_names.contains(&"created_at".to_string()));
    assert!(column_names.contains(&"updated_at".to_string()));

    Ok(())
}
