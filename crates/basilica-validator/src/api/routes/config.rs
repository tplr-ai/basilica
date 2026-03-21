use axum::{extract::State, Json};
use serde_json::Value;

use crate::api::ApiState;
use crate::config::collateral::CollateralConfig;
use crate::config::ValidatorConfig;

pub async fn get_config(State(state): State<ApiState>) -> Json<Value> {
    Json(serialize_public_config(&state.validator_config))
}

pub async fn get_verification_config(State(state): State<ApiState>) -> Json<Value> {
    let verification_config = &state.validator_config.verification;

    // Use serde to serialize the config directly
    let response = serde_json::to_value(verification_config).unwrap_or_else(
        |_| serde_json::json!({"error": "Failed to serialize verification configuration"}),
    );

    Json(response)
}

pub async fn get_emission_config(State(state): State<ApiState>) -> Json<Value> {
    let emission_config = &state.validator_config.emission;

    // Use serde to serialize the emission config directly
    let response = serde_json::to_value(emission_config).unwrap_or_else(
        |_| serde_json::json!({"error": "Failed to serialize emission configuration"}),
    );

    Json(response)
}

fn serialize_public_config(config: &ValidatorConfig) -> Value {
    serde_json::json!({
        "bittensor": &config.bittensor,
        "verification": &config.verification,
        "automatic_verification": &config.automatic_verification,
        "api": {
            "bind_address": &config.api.bind_address,
            "max_body_size": config.api.max_body_size,
            "auth_enabled": config.api.api_key.is_some(),
            "allow_unauthenticated_routes": config.api.allow_unauthenticated_routes,
        },
        "bid_grpc": &config.bid_grpc,
        "emission": &config.emission,
        "bidding": &config.bidding,
        "pricing": &config.pricing,
        "cleanup": &config.cleanup,
        "api_endpoint": &config.api_endpoint,
        "billing": &config.billing,
        "collateral": config.collateral.as_ref().map(serialize_public_collateral_config),
    })
}

fn serialize_public_collateral_config(config: &CollateralConfig) -> Value {
    serde_json::json!({
        "shadow_mode": config.shadow_mode,
        "warning_threshold_multiplier": config.warning_threshold_multiplier,
        "grace_period_hours": config.grace_period_hours,
        "exclude_on_prolonged_price_failure": config.exclude_on_prolonged_price_failure,
        "minimum_usd_per_gpu": &config.minimum_usd_per_gpu,
        "contract_address": &config.contract_address,
        "network": &config.network,
        "slash_fraction": config.slash_fraction,
        "slash_cooldown_secs": config.slash_cooldown_secs,
        "slash_max_per_window": config.slash_max_per_window,
        "slash_window_secs": config.slash_window_secs,
        "slash_circuit_breaker_threshold": config.slash_circuit_breaker_threshold,
        "slash_circuit_breaker_window_secs": config.slash_circuit_breaker_window_secs,
        "slash_circuit_breaker_cooldown_secs": config.slash_circuit_breaker_cooldown_secs,
        "trustee_key_source": &config.trustee_key_source,
        "aws_region": &config.aws_region,
        "evidence_base_url": &config.evidence_base_url,
        "evidence_storage_path": &config.evidence_storage_path,
        "evidence_r2_account_id": &config.evidence_r2_account_id,
        "evidence_r2_access_key_id": &config.evidence_r2_access_key_id,
        "evidence_r2_bucket": &config.evidence_r2_bucket,
        "evidence_public_url_base": &config.evidence_public_url_base,
        "secrets_redacted": true,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serialize_public_config_omits_secret_fields() {
        let mut config = ValidatorConfig::default();
        config.api.api_key = Some("secret-api-key".to_string());
        config.collateral = Some(CollateralConfig {
            aws_secret_name: Some("prod/collateral".to_string()),
            trustee_private_key_file: Some("/tmp/private-key.pem".into()),
            evidence_r2_secret_access_key: Some("super-secret".to_string()),
            ..CollateralConfig::default()
        });

        let response = serialize_public_config(&config);
        let encoded = response.to_string();

        assert_eq!(response["api"]["auth_enabled"], true);
        assert_eq!(response["api"]["allow_unauthenticated_routes"], false);
        assert!(!encoded.contains("secret-api-key"));
        assert!(!encoded.contains("prod/collateral"));
        assert!(!encoded.contains("private-key.pem"));
        assert!(!encoded.contains("super-secret"));
        assert_eq!(response["collateral"]["secrets_redacted"], true);
    }

    #[test]
    fn secret_bearing_config_fields_do_not_serialize_by_default() {
        let mut config = ValidatorConfig::default();
        config.api.api_key = Some("secret-api-key".to_string());
        config.collateral = Some(CollateralConfig {
            aws_secret_name: Some("prod/collateral".to_string()),
            trustee_private_key_file: Some("/tmp/private-key.pem".into()),
            evidence_r2_secret_access_key: Some("super-secret".to_string()),
            ..CollateralConfig::default()
        });

        let serialized = serde_json::to_value(&config).unwrap();
        let encoded = serialized.to_string();

        assert!(serialized["api"].get("api_key").is_none());
        assert!(!encoded.contains("secret-api-key"));
        assert!(!encoded.contains("prod/collateral"));
        assert!(!encoded.contains("private-key.pem"));
        assert!(!encoded.contains("super-secret"));
    }
}
