//! Unit tests for CLI functionality

use common::config::DatabaseConfig;
use miner::cli::{
    display_executor_details, display_executors_table, handle_command, AddExecutorArgs, Command,
    GenerateConfigArgs, ListExecutorsArgs, MinerArgs, RemoveExecutorArgs, StatusArgs,
    UpdateExecutorArgs, ValidatorCommand,
};
use miner::config::{AppConfig, ExecutorConfig};
use miner::persistence::RegistrationDb;
use std::path::PathBuf;
use std::time::Duration;
use tempfile::NamedTempFile;

#[test]
fn test_miner_args_default() {
    let args = MinerArgs {
        config: Some(PathBuf::from("config.toml")),
        command: None,
    };

    assert_eq!(args.config, Some(PathBuf::from("config.toml")));
    assert!(args.command.is_none());
}

#[test]
fn test_command_variants() {
    // Test ListExecutors command
    let cmd = Command::ListExecutors(ListExecutorsArgs {
        active_only: true,
        format: Some("json".to_string()),
    });

    match cmd {
        Command::ListExecutors(args) => {
            assert!(args.active_only);
            assert_eq!(args.format, Some("json".to_string()));
        }
        _ => panic!("Wrong command type"),
    }

    // Test AddExecutor command
    let cmd = Command::AddExecutor(AddExecutorArgs {
        id: "exec1".to_string(),
        grpc_address: "127.0.0.1:50051".to_string(),
        name: Some("Test Executor".to_string()),
        metadata: None,
    });

    match cmd {
        Command::AddExecutor(args) => {
            assert_eq!(args.id, "exec1");
            assert_eq!(args.grpc_address, "127.0.0.1:50051");
            assert_eq!(args.name, Some("Test Executor".to_string()));
        }
        _ => panic!("Wrong command type"),
    }
}

#[tokio::test]
async fn test_handle_list_executors_empty() {
    let db = create_test_db().await;
    let config = create_test_config();

    let args = ListExecutorsArgs {
        active_only: false,
        format: None,
    };

    // Should not panic even with empty executor list
    let result = handle_command(Command::ListExecutors(args), &config, &db).await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_handle_add_executor() {
    let db = create_test_db().await;
    let config = create_test_config();

    let args = AddExecutorArgs {
        id: "exec1".to_string(),
        grpc_address: "127.0.0.1:50051".to_string(),
        name: Some("Test Executor".to_string()),
        metadata: Some("{\"gpu\": \"RTX 4090\"}".to_string()),
    };

    let result = handle_command(Command::AddExecutor(args), &config, &db).await;
    assert!(result.is_ok());

    // Verify executor was added
    let executor = db.get_executor("exec1").await.unwrap();
    assert!(executor.is_some());
}

#[tokio::test]
async fn test_handle_remove_executor() {
    let db = create_test_db().await;
    let config = create_test_config();

    // First add an executor
    db.register_executor("exec1", "127.0.0.1:50051", serde_json::json!({}))
        .await
        .unwrap();

    let args = RemoveExecutorArgs {
        id: "exec1".to_string(),
        force: false,
    };

    let result = handle_command(Command::RemoveExecutor(args), &config, &db).await;
    assert!(result.is_ok());

    // Verify executor is inactive
    let executor = db.get_executor("exec1").await.unwrap();
    assert!(executor.is_some());
    assert!(!executor.unwrap().is_active);
}

#[tokio::test]
async fn test_handle_update_executor() {
    let db = create_test_db().await;
    let config = create_test_config();

    // First add an executor
    db.register_executor("exec1", "127.0.0.1:50051", serde_json::json!({}))
        .await
        .unwrap();

    let args = UpdateExecutorArgs {
        id: "exec1".to_string(),
        grpc_address: Some("127.0.0.1:50052".to_string()),
        name: Some("Updated Executor".to_string()),
        metadata: None,
    };

    let result = handle_command(Command::UpdateExecutor(args), &config, &db).await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_handle_status() {
    let db = create_test_db().await;
    let config = create_test_config();

    let args = StatusArgs {
        detailed: false,
        json: false,
    };

    let result = handle_command(Command::Status(args), &config, &db).await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_handle_generate_config() {
    let temp_file = NamedTempFile::new().unwrap();
    let output_path = temp_file.path().to_path_buf();

    let db = create_test_db().await;
    let config = create_test_config();

    let args = GenerateConfigArgs {
        output: output_path.clone(),
        force: true,
    };

    let result = handle_command(Command::GenerateConfig(args), &config, &db).await;
    assert!(result.is_ok());

    // Verify file was created
    assert!(output_path.exists());
}

#[tokio::test]
async fn test_handle_validator_list() {
    let db = create_test_db().await;
    let config = create_test_config();

    let cmd = ValidatorCommand::List;

    let result = handle_command(Command::Validator(cmd), &config, &db).await;
    assert!(result.is_ok());
}

#[test]
fn test_display_executors_table() {
    let executors = vec![
        miner::persistence::ExecutorRecord {
            id: "exec1".to_string(),
            grpc_address: "127.0.0.1:50051".to_string(),
            is_active: true,
            is_healthy: true,
            last_seen: chrono::Utc::now(),
            metadata: serde_json::json!({"name": "Executor 1"}),
        },
        miner::persistence::ExecutorRecord {
            id: "exec2".to_string(),
            grpc_address: "127.0.0.1:50052".to_string(),
            is_active: true,
            is_healthy: false,
            last_seen: chrono::Utc::now() - chrono::Duration::minutes(5),
            metadata: serde_json::json!({"name": "Executor 2"}),
        },
    ];

    // Should not panic
    display_executors_table(&executors);
}

#[test]
fn test_display_executor_details() {
    let executor = miner::persistence::ExecutorRecord {
        id: "exec1".to_string(),
        grpc_address: "127.0.0.1:50051".to_string(),
        is_active: true,
        is_healthy: true,
        last_seen: chrono::Utc::now(),
        metadata: serde_json::json!({
            "name": "Test Executor",
            "gpu": "RTX 4090",
            "location": "US-East"
        }),
    };

    // Should not panic
    display_executor_details(&executor);
}

// Helper functions

async fn create_test_db() -> RegistrationDb {
    let db_config = DatabaseConfig {
        url: "sqlite::memory:".to_string(),
        max_connections: 5,
        min_connections: 1,
        connection_timeout: Duration::from_secs(10),
        idle_timeout: Duration::from_secs(300),
        max_lifetime: Duration::from_secs(3600),
    };

    RegistrationDb::new(&db_config).await.unwrap()
}

fn create_test_config() -> AppConfig {
    AppConfig::default()
}
