//! Tests for error handling

use basilica_federation::error::{FederationError, Result};

#[test]
fn test_federation_error_display() {
    let error = FederationError::Config("test error".to_string());
    assert!(format!("{}", error).contains("test error"));
}

#[test]
fn test_federation_error_cluster_not_found() {
    let error = FederationError::ClusterNotFound("cluster-1".to_string());
    assert!(format!("{}", error).contains("cluster-1"));
}

#[test]
fn test_federation_error_discovery() {
    let error = FederationError::Discovery("discovery failed".to_string());
    assert!(format!("{}", error).contains("discovery failed"));
}

#[test]
fn test_federation_error_health() {
    let error = FederationError::Health("health check failed".to_string());
    assert!(format!("{}", error).contains("health check failed"));
}

#[test]
fn test_federation_error_load_balancing() {
    let error = FederationError::LoadBalancing("no clusters available".to_string());
    assert!(format!("{}", error).contains("no clusters available"));
}

#[test]
fn test_federation_error_resource_management() {
    let error = FederationError::ResourceManagement("resource sync failed".to_string());
    assert!(format!("{}", error).contains("resource sync failed"));
}

#[test]
fn test_federation_error_timeout() {
    let error = FederationError::Timeout("operation timed out".to_string());
    assert!(format!("{}", error).contains("operation timed out"));
}

#[test]
fn test_federation_error_invalid_state() {
    let error = FederationError::InvalidState("invalid cluster state".to_string());
    assert!(format!("{}", error).contains("invalid cluster state"));
}

#[test]
fn test_result_type() {
    fn test_function() -> Result<()> {
        Ok(())
    }
    
    assert!(test_function().is_ok());
}

#[test]
fn test_result_type_with_error() {
    fn test_function() -> Result<()> {
        Err(FederationError::Config("error".to_string()))
    }
    
    assert!(test_function().is_err());
}

