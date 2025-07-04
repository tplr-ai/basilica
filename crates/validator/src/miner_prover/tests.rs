//! Integration tests for dynamic discovery

#[cfg(test)]
use crate::miner_prover::miner_client::{MinerClient, MinerClientConfig};
#[cfg(test)]
use common::identity::Hotkey;
#[cfg(test)]
use std::time::Duration;

#[test]
fn test_axon_to_grpc_endpoint_conversion() {
    let config = MinerClientConfig {
        timeout: Duration::from_secs(30),
        max_retries: 3,
        grpc_port_offset: None,
        use_tls: false,
    };

    let hotkey =
        Hotkey::new("5HGjWAeFDfFCWPsjFQdVV2Msvz2XtMktvgocEZcCj68kUMaw".to_string()).unwrap();
    let client = MinerClient::new(config.clone(), hotkey);

    // Test default port mapping (axon port -> same gRPC port when no offset)
    let test_cases = vec![
        ("http://192.168.1.100:8091", "http://192.168.1.100:8091"),
        ("http://10.0.0.1:9091", "http://10.0.0.1:9091"),
        ("http://example.com:8091", "http://example.com:8091"),
        ("http://[2001:db8::1]:8091", "http://[2001:db8::1]:8091"),
    ];

    for (axon, expected) in test_cases {
        let result = client.axon_to_grpc_endpoint(axon).unwrap();
        assert_eq!(result, expected, "Failed for input: {axon}");
    }

    // Test with custom offset
    let config_with_offset = MinerClientConfig {
        grpc_port_offset: Some(1000),
        ..config.clone()
    };
    let client_with_offset = MinerClient::new(
        config_with_offset,
        Hotkey::new("5HGjWAeFDfFCWPsjFQdVV2Msvz2XtMktvgocEZcCj68kUMaw".to_string()).unwrap(),
    );
    let result = client_with_offset
        .axon_to_grpc_endpoint("http://10.0.0.1:8091")
        .unwrap();
    assert_eq!(result, "http://10.0.0.1:9091"); // 8091 + 1000 = 9091

    // Test with TLS
    let config_with_tls = MinerClientConfig {
        use_tls: true,
        ..config
    };
    let client_with_tls = MinerClient::new(
        config_with_tls,
        Hotkey::new("5HGjWAeFDfFCWPsjFQdVV2Msvz2XtMktvgocEZcCj68kUMaw".to_string()).unwrap(),
    );
    let result = client_with_tls
        .axon_to_grpc_endpoint("http://example.com:8091")
        .unwrap();
    assert_eq!(result, "https://example.com:8091");
}

#[test]
fn test_ip_conversion_logic() {
    // Test IPv4 address conversion from u128
    let ipv4_u128: u128 = 0x00000000000000000000ffffc0a80101; // 192.168.1.1
    let ipv4_bytes = ipv4_u128.to_be_bytes();
    let ipv4_str = format!(
        "{}.{}.{}.{}",
        ipv4_bytes[12], ipv4_bytes[13], ipv4_bytes[14], ipv4_bytes[15]
    );
    assert_eq!(ipv4_str, "192.168.1.1");

    // Test IPv6 address conversion from u128
    let ipv6_u128: u128 = 0x20010db8000000000000000000000001; // 2001:db8::1
    let ipv6_bytes = ipv6_u128.to_be_bytes();
    let segments: Vec<u16> = (0..8)
        .map(|i| u16::from_be_bytes([ipv6_bytes[i * 2], ipv6_bytes[i * 2 + 1]]))
        .collect();
    let ipv6_str = segments
        .iter()
        .map(|&s| format!("{s:x}"))
        .collect::<Vec<_>>()
        .join(":");
    assert_eq!(ipv6_str, "2001:db8:0:0:0:0:0:1");
}

#[test]
fn test_ssh_credential_parsing() {
    use crate::config::VerificationConfig;
    use crate::miner_prover::verification::VerificationEngine;

    let config = VerificationConfig {
        verification_interval: Duration::from_secs(3600),
        max_concurrent_verifications: 10,
        challenge_timeout: Duration::from_secs(60),
        min_score_threshold: 0.0,
        max_miners_per_round: 10,
        min_verification_interval: Duration::from_secs(3600),
        netuid: 1,
        use_dynamic_discovery: false,
        discovery_timeout: Duration::from_secs(30),
        fallback_to_static: true,
        cache_miner_info_ttl: Duration::from_secs(3600),
        grpc_port_offset: None,
    };
    let engine = VerificationEngine::new(config);

    // Test standard format with port
    let creds = "validator@192.168.1.100:2222";
    let details = engine.parse_ssh_credentials(creds).unwrap();
    assert_eq!(details.username, "validator");
    assert_eq!(details.host, "192.168.1.100");
    assert_eq!(details.port, 2222);

    // Test without port (should default to 22)
    let creds_no_port = "validator@192.168.1.100";
    let details_no_port = engine.parse_ssh_credentials(creds_no_port).unwrap();
    assert_eq!(details_no_port.username, "validator");
    assert_eq!(details_no_port.host, "192.168.1.100");
    assert_eq!(details_no_port.port, 22);

    // Test with IPv6 address
    let creds_ipv6 = "validator@[2001:db8::1]:2222";
    let details_ipv6 = engine.parse_ssh_credentials(creds_ipv6).unwrap();
    assert_eq!(details_ipv6.username, "validator");
    assert_eq!(details_ipv6.host, "[2001:db8::1]");
    assert_eq!(details_ipv6.port, 2222);

    // Test invalid format
    let invalid_creds = "invalid-format";
    assert!(engine.parse_ssh_credentials(invalid_creds).is_err());
}

#[tokio::test]
async fn test_dynamic_discovery_config() {
    use crate::config::VerificationConfig;
    use std::time::Duration;

    let config = VerificationConfig {
        verification_interval: Duration::from_secs(3600),
        max_concurrent_verifications: 10,
        challenge_timeout: Duration::from_secs(60),
        min_score_threshold: 0.0,
        max_miners_per_round: 10,
        min_verification_interval: Duration::from_secs(3600),
        netuid: 1,
        use_dynamic_discovery: true,
        discovery_timeout: Duration::from_secs(30),
        fallback_to_static: true,
        cache_miner_info_ttl: Duration::from_secs(300),
        grpc_port_offset: Some(42000),
    };

    // Verify configuration
    assert!(config.use_dynamic_discovery);
    assert_eq!(config.discovery_timeout, Duration::from_secs(30));
    assert!(config.fallback_to_static);
    assert_eq!(config.cache_miner_info_ttl, Duration::from_secs(300));
    assert_eq!(config.grpc_port_offset, Some(42000));
}
