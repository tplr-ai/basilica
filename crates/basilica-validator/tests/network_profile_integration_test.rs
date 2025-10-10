//! Integration test for network profile collection

use basilica_validator::miner_prover::validation_network::NetworkProfile;
use basilica_validator::persistence::SimplePersistence;

#[tokio::test]
async fn test_network_profile_database_integration() -> Result<(), anyhow::Error> {
    // Create in-memory database for testing
    let persistence = SimplePersistence::new(":memory:", "test_validator".to_string()).await?;
    persistence.run_migrations().await?;

    // Test data
    let miner_uid = 1u16;
    let node_id = "test_node";
    let test_timestamp = chrono::Utc::now();

    // Store a network profile
    persistence
        .store_node_network_profile(
            miner_uid,
            node_id,
            Some("192.168.1.1".to_string()),
            Some("test.example.com".to_string()),
            Some("San Francisco".to_string()),
            Some("California".to_string()),
            Some("US".to_string()),
            Some("37.7749,-122.4194".to_string()),
            Some("AS12345 Example ISP".to_string()),
            Some("94102".to_string()),
            Some("America/Los_Angeles".to_string()),
            &test_timestamp.to_rfc3339(),
            r#"{"ip":"192.168.1.1","city":"San Francisco"}"#,
        )
        .await?;

    // Retrieve the profile
    let result = persistence
        .get_node_network_profile(miner_uid, node_id)
        .await?;

    assert!(result.is_some());
    let (
        full_json,
        ip,
        hostname,
        city,
        region,
        country,
        location,
        org,
        postal,
        timezone,
        timestamp,
    ) = result.unwrap();

    assert_eq!(ip, Some("192.168.1.1".to_string()));
    assert_eq!(hostname, Some("test.example.com".to_string()));
    assert_eq!(city, Some("San Francisco".to_string()));
    assert_eq!(region, Some("California".to_string()));
    assert_eq!(country, Some("US".to_string()));
    assert_eq!(location, Some("37.7749,-122.4194".to_string()));
    assert_eq!(org, Some("AS12345 Example ISP".to_string()));
    assert_eq!(postal, Some("94102".to_string()));
    assert_eq!(timezone, Some("America/Los_Angeles".to_string()));
    assert_eq!(timestamp, test_timestamp.to_rfc3339());
    assert!(full_json.contains("192.168.1.1"));

    Ok(())
}

#[test]
fn test_ipinfo_parsing_with_real_data() {
    let real_response = r#"{
        "ip": "198.51.100.25",
        "hostname": "example-server.hosting.net",
        "city": "Amsterdam",
        "region": "North Holland",
        "country": "NL",
        "loc": "52.3740,4.8897",
        "org": "AS64496 Example Hosting B.V.",
        "postal": "1017",
        "timezone": "Europe/Amsterdam",
        "readme": "https://ipinfo.io/missingauth"
    }"#;

    let profile = NetworkProfile::from_ipinfo_json(real_response).unwrap();

    // Verify all fields are parsed correctly
    assert_eq!(profile.ip_address.as_deref(), Some("198.51.100.25"));
    assert_eq!(
        profile.hostname.as_deref(),
        Some("example-server.hosting.net")
    );
    assert_eq!(profile.city.as_deref(), Some("Amsterdam"));
    assert_eq!(profile.region.as_deref(), Some("North Holland"));
    assert_eq!(profile.country.as_deref(), Some("NL"));
    assert_eq!(profile.location.as_deref(), Some("52.3740,4.8897"));
    assert_eq!(
        profile.organization.as_deref(),
        Some("AS64496 Example Hosting B.V.")
    );
    assert_eq!(profile.postal_code.as_deref(), Some("1017"));
    assert_eq!(profile.timezone.as_deref(), Some("Europe/Amsterdam"));

    // Verify full JSON is preserved
    assert!(profile.full_json.contains("198.51.100.25"));
    assert!(profile.full_json.contains("Example Hosting"));
}
