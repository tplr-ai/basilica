//! Unit tests for NodeManager

use basilica_miner::config::NodeSshConfig;
use basilica_miner::node_manager::{NodeConfig, NodeManager};

#[tokio::test]
async fn test_node_registration() {
    let manager = NodeManager::new(NodeSshConfig::default());

    let config = NodeConfig {
        host: "192.168.1.100".to_string(),
        port: 22,
        username: "basilica".to_string(),
        hourly_rate_per_gpu: 2.5,
        additional_opts: None,
    };

    let node_id = "test-node-1".to_string();

    // Register node
    let result = manager.register_node(node_id.clone(), config.clone()).await;
    assert!(result.is_ok());

    // Get node
    let node = manager.get_node(&node_id).await.unwrap();
    assert!(node.is_some());

    let node = node.unwrap();
    assert_eq!(node.host, "192.168.1.100");
}

#[tokio::test]
async fn test_list_nodes() {
    let manager = NodeManager::new(NodeSshConfig::default());

    // Register multiple nodes
    manager
        .register_node(
            "node-1".to_string(),
            NodeConfig {
                host: "192.168.1.100".to_string(),
                port: 22,
                username: "basilica".to_string(),
                hourly_rate_per_gpu: 2.5,
                additional_opts: None,
            },
        )
        .await
        .unwrap();

    manager
        .register_node(
            "node-2".to_string(),
            NodeConfig {
                host: "192.168.1.101".to_string(),
                port: 22,
                username: "basilica".to_string(),
                hourly_rate_per_gpu: 2.5,
                additional_opts: None,
            },
        )
        .await
        .unwrap();

    manager
        .register_node(
            "node-3".to_string(),
            NodeConfig {
                host: "192.168.1.102".to_string(),
                port: 22,
                username: "basilica".to_string(),
                hourly_rate_per_gpu: 2.5,
                additional_opts: None,
            },
        )
        .await
        .unwrap();

    // List nodes - all configured nodes are returned
    let nodes = manager.list_nodes().await.unwrap();
    assert_eq!(nodes.len(), 3);
    assert!(nodes.iter().any(|n| n.node_id == "node-1"));
    assert!(nodes.iter().any(|n| n.node_id == "node-2"));
    assert!(nodes.iter().any(|n| n.node_id == "node-3"));
}

#[tokio::test]
async fn test_unregister_node() {
    let manager = NodeManager::new(NodeSshConfig::default());

    let node_id = "test-node".to_string();

    // Register a node
    manager
        .register_node(
            node_id.clone(),
            NodeConfig {
                host: "192.168.1.100".to_string(),
                port: 22,
                username: "basilica".to_string(),
                hourly_rate_per_gpu: 2.5,
                additional_opts: None,
            },
        )
        .await
        .unwrap();

    // Verify it exists
    let node = manager.get_node(&node_id).await.unwrap();
    assert!(node.is_some());

    // Unregister it
    manager.unregister_node(&node_id).await.unwrap();

    // Verify it's gone
    let node = manager.get_node(&node_id).await.unwrap();
    assert!(node.is_none());
}

#[tokio::test]
async fn test_validator_assignment_tracking() {
    let manager = NodeManager::new(NodeSshConfig::default());

    let validator_hotkey = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY";

    // Initially not authorized
    assert!(!manager.is_validator_authorized(validator_hotkey).await);

    // Set the assigned validator
    manager.set_assigned_validator(validator_hotkey).await;

    // Now should be authorized
    assert!(manager.is_validator_authorized(validator_hotkey).await);
}

#[tokio::test]
async fn test_validator_assignment_overwrites_previous() {
    let manager = NodeManager::new(NodeSshConfig::default());

    let validator_a = "validator-a";
    let validator_b = "validator-b";

    // Assign to validator A
    manager.set_assigned_validator(validator_a).await;

    assert!(manager.is_validator_authorized(validator_a).await);
    assert!(!manager.is_validator_authorized(validator_b).await);

    // Reassign to validator B
    manager.set_assigned_validator(validator_b).await;

    assert!(!manager.is_validator_authorized(validator_a).await);
    assert!(manager.is_validator_authorized(validator_b).await);
}

#[tokio::test]
async fn test_revoke_clears_current_assignment() {
    let manager = NodeManager::new(NodeSshConfig::default());

    let validator_hotkey = "validator-revoke";

    // Assign validator
    manager.set_assigned_validator(validator_hotkey).await;

    assert!(manager.is_validator_authorized(validator_hotkey).await);

    // Revoke (without actual SSH operations since there are no nodes)
    manager
        .revoke_validator(validator_hotkey)
        .await
        .expect("revocation should succeed");

    assert!(!manager.is_validator_authorized(validator_hotkey).await);

    // Second revoke should be a no-op
    manager
        .revoke_validator(validator_hotkey)
        .await
        .expect("repeated revoke should not fail");
}

#[tokio::test]
async fn test_deploy_validator_keys_without_nodes() {
    let manager = NodeManager::new(NodeSshConfig::default());

    let validator_hotkey = "validator-no-nodes";
    // Valid ed25519 test key
    let ssh_public_key = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOMqqnkVzrm0SdG6UOoqKLsabgH5C9okWi0dh2l9GKJl validator@example.com";

    // Deploy keys without any registered nodes should succeed (no-op)
    let result = manager
        .deploy_validator_keys(validator_hotkey, ssh_public_key)
        .await;

    assert!(result.is_ok());
}

#[test]
fn test_normalize_ssh_key() {
    use basilica_miner::node_manager::NodeManager;

    let validator_hotkey = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY";

    // Test with key that has existing comment
    let key_with_comment = "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQC user@hostname";
    let normalized = NodeManager::normalize_ssh_key(key_with_comment, validator_hotkey);
    assert_eq!(
        normalized,
        format!(
            "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQC validator-{}",
            validator_hotkey
        )
    );

    // Test with key without comment
    let key_without_comment = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAI";
    let normalized = NodeManager::normalize_ssh_key(key_without_comment, validator_hotkey);
    assert_eq!(
        normalized,
        format!(
            "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAI validator-{}",
            validator_hotkey
        )
    );

    // Test with key with multiple spaces/comments
    let key_complex = "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQC user@host extra comment";
    let normalized = NodeManager::normalize_ssh_key(key_complex, validator_hotkey);
    assert_eq!(
        normalized,
        format!(
            "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQC validator-{}",
            validator_hotkey
        )
    );
}

#[test]
fn test_extract_key_core() {
    use basilica_miner::node_manager::NodeManager;

    // Test with key that has comment
    let key_with_comment = "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQC user@hostname";
    let core = NodeManager::extract_key_core(key_with_comment);
    assert_eq!(core, "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQC");

    // Test with key without comment
    let key_without_comment = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAI";
    let core = NodeManager::extract_key_core(key_without_comment);
    assert_eq!(core, "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAI");

    // Test that different comments result in same core
    let key1 = "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQC comment1";
    let key2 = "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQC comment2";
    assert_eq!(
        NodeManager::extract_key_core(key1),
        NodeManager::extract_key_core(key2)
    );
}
