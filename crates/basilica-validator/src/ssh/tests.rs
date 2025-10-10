//! Unit tests for SSH validation workflow
//!
//! Tests SCP file transfer, remote command execution, retry logic,
//! session management, and connection pooling functionality.

#[cfg(test)]
mod ssh_tests {
    use crate::ssh::{NodeSshDetails, RetryConfig, SshSessionStats, ValidatorSshClient};
    use basilica_common::identity::NodeId;
    use basilica_common::ssh::{SshConnectionConfig, SshConnectionDetails};
    use std::time::Duration;
    use tempfile::tempdir;

    /// Create test SSH connection details
    fn create_test_ssh_details() -> SshConnectionDetails {
        let temp_dir = tempdir().unwrap();
        let key_path = temp_dir.path().join("test_key");
        std::fs::write(&key_path, "dummy_key_content").unwrap();

        SshConnectionDetails {
            host: "test.example.com".to_string(),
            username: "testuser".to_string(),
            port: 2222,
            private_key_path: key_path,
            timeout: Duration::from_secs(30),
        }
    }

    /// Create test node SSH details
    fn create_test_node_details() -> NodeSshDetails {
        let connection = create_test_ssh_details();
        NodeSshDetails {
            node_id: NodeId::new(&connection.host.clone()).unwrap(),
            connection,
        }
    }

    #[test]
    fn test_retry_config_creation() {
        let config = RetryConfig::default();
        assert_eq!(config.max_attempts, 3);
        assert_eq!(config.initial_delay, Duration::from_millis(500));
        assert_eq!(config.max_delay, Duration::from_secs(30));
        assert_eq!(config.backoff_multiplier, 2.0);
        assert!(config.retry_on_timeout);
        assert!(config.retry_on_connection_error);
    }

    #[test]
    fn test_ssh_session_stats_creation() {
        let stats = SshSessionStats::default();
        assert_eq!(stats.total_connections, 0);
        assert_eq!(stats.successful_connections, 0);
        assert_eq!(stats.failed_connections, 0);
        assert_eq!(stats.total_transfers, 0);
        assert_eq!(stats.successful_transfers, 0);
        assert_eq!(stats.failed_transfers, 0);
        assert_eq!(stats.total_commands, 0);
        assert_eq!(stats.successful_commands, 0);
        assert_eq!(stats.failed_commands, 0);
        assert_eq!(stats.average_response_time_ms, 0.0);
    }

    #[test]
    fn test_validator_ssh_client_creation() {
        let client = ValidatorSshClient::new();
        let stats = client.get_session_stats();
        assert_eq!(stats.total_connections, 0);

        let (pool_size, _) = client.get_pool_info();
        assert_eq!(pool_size, 0);
    }

    #[test]
    fn test_validator_ssh_client_with_config() {
        let ssh_config = SshConnectionConfig {
            connection_timeout: Duration::from_secs(60),
            execution_timeout: Duration::from_secs(300),
            max_transfer_size: 50 * 1024 * 1024, // 50MB
            retry_attempts: 5,
            cleanup_remote_files: false,
            strict_host_key_checking: false,
            known_hosts_file: None,
        };

        let client = ValidatorSshClient::with_config(ssh_config);
        let stats = client.get_session_stats();
        assert_eq!(stats.total_connections, 0);
    }

    #[test]
    fn test_validator_ssh_client_with_retry_config() {
        let ssh_config = SshConnectionConfig::default();
        let retry_config = RetryConfig {
            max_attempts: 5,
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(10),
            backoff_multiplier: 1.5,
            retry_on_timeout: false,
            retry_on_connection_error: false,
        };

        let client = ValidatorSshClient::with_retry_config(ssh_config, retry_config);
        let stats = client.get_session_stats();
        assert_eq!(stats.total_connections, 0);
    }

    #[test]
    fn test_pool_key_generation() {
        let client = ValidatorSshClient::new();
        let details = create_test_ssh_details();
        let key = client.get_pool_key(&details);
        assert_eq!(key, "testuser@test.example.com:2222");
    }

    #[test]
    fn test_connection_pool_management() {
        let client = ValidatorSshClient::new();
        let details = create_test_ssh_details();

        // Initially empty pool
        let (pool_size, _) = client.get_pool_info();
        assert_eq!(pool_size, 0);

        // Add connection to pool
        client.update_connection_pool(&details, true);
        let (pool_size, keys) = client.get_pool_info();
        assert_eq!(pool_size, 1);
        assert!(keys.contains(&"testuser@test.example.com:2222".to_string()));

        // Update same connection
        client.update_connection_pool(&details, false);
        let (pool_size, _) = client.get_pool_info();
        assert_eq!(pool_size, 1); // Should still be 1

        // Clear pool
        client.clear_pool();
        let (pool_size, _) = client.get_pool_info();
        assert_eq!(pool_size, 0);
    }

    #[test]
    fn test_session_stats_updates() {
        let client = ValidatorSshClient::new();

        // Update connection stats
        client.update_stats(|stats| {
            stats.total_connections += 1;
            stats.successful_connections += 1;
        });

        let stats = client.get_session_stats();
        assert_eq!(stats.total_connections, 1);
        assert_eq!(stats.successful_connections, 1);

        // Update transfer stats
        client.update_stats(|stats| {
            stats.total_transfers += 1;
            stats.successful_transfers += 1;
        });

        let stats = client.get_session_stats();
        assert_eq!(stats.total_transfers, 1);
        assert_eq!(stats.successful_transfers, 1);
    }

    #[test]
    fn test_should_retry_error_logic() {
        let client = ValidatorSshClient::new();

        // Test authentication errors (should not retry)
        let auth_error = anyhow::anyhow!("Permission denied (publickey)");
        assert!(!client.should_retry_error(&auth_error));

        let invalid_key_error = anyhow::anyhow!("Invalid private key format");
        assert!(!client.should_retry_error(&invalid_key_error));

        // Test connection errors (should retry by default)
        let conn_error = anyhow::anyhow!("Connection refused");
        assert!(client.should_retry_error(&conn_error));

        let unreachable_error = anyhow::anyhow!("Network unreachable");
        assert!(client.should_retry_error(&unreachable_error));

        // Test timeout errors (should retry by default)
        let timeout_error = anyhow::anyhow!("Operation timed out");
        assert!(client.should_retry_error(&timeout_error));

        // Test file not found errors (should not retry)
        let not_found_error = anyhow::anyhow!("No such file or directory");
        assert!(!client.should_retry_error(&not_found_error));

        // Test temporary failures (should retry)
        let temp_error = anyhow::anyhow!("Resource temporarily unavailable");
        assert!(client.should_retry_error(&temp_error));

        // Test unknown errors (should not retry by default)
        let unknown_error = anyhow::anyhow!("Unknown error occurred");
        assert!(!client.should_retry_error(&unknown_error));
    }

    #[test]
    fn test_create_node_connection() {
        let temp_dir = tempdir().unwrap();
        let key_path = temp_dir.path().join("test_key");
        std::fs::write(&key_path, "dummy_key_content").unwrap();

        let node_id = NodeId::new("test_node").unwrap();
        let connection = ValidatorSshClient::create_node_connection(
            node_id,
            "test.example.com".to_string(),
            "testuser".to_string(),
            2222,
            key_path.clone(),
            Some(Duration::from_secs(45)),
        );

        assert_eq!(connection.host, "test.example.com");
        assert_eq!(connection.username, "testuser");
        assert_eq!(connection.port, 2222);
        assert_eq!(connection.private_key_path, key_path);
        assert_eq!(connection.timeout, Duration::from_secs(45));
    }

    #[test]
    fn test_create_node_connection_defaults() {
        let temp_dir = tempdir().unwrap();
        let key_path = temp_dir.path().join("test_key");
        std::fs::write(&key_path, "dummy_key_content").unwrap();

        let node_id = NodeId::new("test_node").unwrap();
        let connection = ValidatorSshClient::create_node_connection(
            node_id,
            "test.example.com".to_string(),
            "testuser".to_string(),
            22,
            key_path.clone(),
            None,
        );

        assert_eq!(connection.port, 22);
        assert_eq!(connection.timeout, Duration::from_secs(30));
    }

    #[test]
    fn test_node_ssh_details_creation() {
        let node_details = create_test_node_details();

        assert_eq!(node_details.connection().host, "test.example.com");
        assert_eq!(node_details.connection().username, "testuser");
        assert_eq!(node_details.connection().port, 2222);
        assert_eq!(node_details.connection().timeout, Duration::from_secs(30));
    }

    #[test]
    fn test_node_ssh_details_getters() {
        let node_details = create_test_node_details();
        let original_id = node_details.node_id.clone();

        assert_eq!(node_details.node_id(), &original_id);
        assert_eq!(node_details.connection().host, "test.example.com");
    }

    /// Test error types that should be retried with custom retry config
    #[test]
    fn test_custom_retry_config_error_handling() {
        let retry_config = RetryConfig {
            max_attempts: 3,
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(5),
            backoff_multiplier: 2.0,
            retry_on_timeout: false,          // Disable timeout retries
            retry_on_connection_error: false, // Disable connection retries
        };

        let ssh_config = SshConnectionConfig::default();
        let client = ValidatorSshClient::with_retry_config(ssh_config, retry_config);

        // With custom config, timeouts should not be retried
        let timeout_error = anyhow::anyhow!("Operation timed out");
        assert!(!client.should_retry_error(&timeout_error));

        // With custom config, connection errors should not be retried
        let conn_error = anyhow::anyhow!("Connection refused");
        assert!(!client.should_retry_error(&conn_error));

        // Temporary failures should still be retried
        let temp_error = anyhow::anyhow!("Resource temporarily unavailable");
        assert!(client.should_retry_error(&temp_error));
    }
}
