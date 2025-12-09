use crate::bdd::TestContext;
use basilica_protocol::billing::{
    track_rental_request::CloudType, CommunityCloudData, RentalStatus, ResourceUsage,
    TelemetryData, TrackRentalRequest, UpdateRentalStatusRequest, UsageReportRequest,
};
use uuid::Uuid;

#[tokio::test]
async fn test_get_usage_report_for_rental() {
    let mut context = TestContext::new().await;
    let user_id = "test_usage_report";

    context.create_test_user(user_id, "1000.0").await;

    let rental_id = Uuid::new_v4().to_string();
    let track_request = TrackRentalRequest {
        rental_id: rental_id.clone(),
        user_id: user_id.to_string(),
        start_time: None,
        metadata: std::collections::HashMap::new(),
        resource_spec: None,
        cloud_type: Some(CloudType::Community(CommunityCloudData {
            node_id: "node_usage".to_string(),
            validator_id: "validator_usage".to_string(),
            base_price_per_gpu: 2.5,
            gpu_count: 1,
            miner_uid: 1,
            miner_hotkey: "test_hotkey".to_string(),
        })),
    };

    let track_response = context
        .client
        .track_rental(track_request)
        .await
        .expect("Failed to track rental")
        .into_inner();

    let activate_request = UpdateRentalStatusRequest {
        rental_id: track_response.tracking_id.clone(),
        status: RentalStatus::Active.into(),
        timestamp: None,
        reason: String::new(),
    };

    context
        .client
        .update_rental_status(activate_request)
        .await
        .expect("Failed to activate rental");

    let rental_uuid =
        Uuid::parse_str(&track_response.tracking_id).expect("Failed to parse rental UUID");

    sqlx::query(
        "INSERT INTO billing.usage_events (event_id, rental_id, user_id, node_id, validator_id, event_type, event_data, timestamp)
         VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())"
    )
    .bind(Uuid::new_v4())
    .bind(rental_uuid)
    .bind(user_id)
    .bind("node_usage")
    .bind("validator_usage")
    .bind("telemetry")
    .bind(serde_json::json!({
        "cpu_percent": 0.75,
        "memory_gb": 16.0,
        "network_gb": 0.5,
        "gpu_hours": 0.9,
        "disk_io_gb": 0.2
    }))
    .execute(&context.pool)
    .await
    .expect("Failed to insert usage event");

    let request = UsageReportRequest {
        rental_id: track_response.tracking_id.clone(),
        start_time: None,
        end_time: None,
        aggregation: 0,
    };

    let response = context
        .client
        .get_usage_report(request)
        .await
        .expect("Failed to get usage report")
        .into_inner();

    assert_eq!(response.rental_id, track_response.tracking_id);
    assert!(
        !response.data_points.is_empty(),
        "Should have usage data points"
    );
    assert!(response.summary.is_some(), "Should have usage summary");

    if let Some(summary) = response.summary {
        assert!(summary.avg_cpu_percent > 0.0, "Should have CPU usage");
        assert!(summary.avg_memory_mb > 0, "Should have memory usage");
        assert!(summary.duration.is_some(), "Should have duration");
    }

    context.cleanup().await;
}

#[tokio::test]
async fn test_usage_report_empty_for_new_rental() {
    let mut context = TestContext::new().await;
    let user_id = "test_empty_usage";

    context.create_test_user(user_id, "1000.0").await;

    let rental_id = Uuid::new_v4().to_string();
    let track_request = TrackRentalRequest {
        rental_id: rental_id.clone(),
        user_id: user_id.to_string(),
        start_time: None,
        metadata: std::collections::HashMap::new(),
        resource_spec: None,
        cloud_type: Some(CloudType::Community(CommunityCloudData {
            node_id: "node_empty".to_string(),
            validator_id: "validator_empty".to_string(),
            base_price_per_gpu: 2.5,
            gpu_count: 1,
            miner_uid: 1,
            miner_hotkey: "test_hotkey".to_string(),
        })),
    };

    let track_response = context
        .client
        .track_rental(track_request)
        .await
        .expect("Failed to track rental")
        .into_inner();

    let request = UsageReportRequest {
        rental_id: track_response.tracking_id.clone(),
        start_time: None,
        end_time: None,
        aggregation: 0,
    };

    let response = context
        .client
        .get_usage_report(request)
        .await
        .expect("Failed to get usage report")
        .into_inner();

    assert_eq!(response.rental_id, track_response.tracking_id);
    assert!(
        response.data_points.len() <= 1,
        "Should have minimal data points"
    );
    assert!(response.summary.is_some(), "Should still have summary");

    context.cleanup().await;
}

#[tokio::test]
async fn test_ingest_telemetry_stream() {
    let mut context = TestContext::new().await;
    let user_id = "test_telemetry_ingest";

    context.create_test_user(user_id, "1000.0").await;

    let rental_id = Uuid::new_v4().to_string();
    let track_request = TrackRentalRequest {
        rental_id: rental_id.clone(),
        user_id: user_id.to_string(),
        start_time: None,
        metadata: std::collections::HashMap::new(),
        resource_spec: None,
        cloud_type: Some(CloudType::Community(CommunityCloudData {
            node_id: "node_telemetry".to_string(),
            validator_id: "validator_telemetry".to_string(),
            base_price_per_gpu: 2.5,
            gpu_count: 1,
            miner_uid: 1,
            miner_hotkey: "test_hotkey".to_string(),
        })),
    };

    let track_response = context
        .client
        .track_rental(track_request)
        .await
        .expect("Failed to track rental")
        .into_inner();

    let (tx, rx) = tokio::sync::mpsc::channel(10);

    for i in 0..3 {
        let telemetry = TelemetryData {
            rental_id: track_response.tracking_id.clone(),
            node_id: "node_telemetry".to_string(),
            timestamp: Some(prost_types::Timestamp::from(std::time::SystemTime::now())),
            resource_usage: Some(ResourceUsage {
                cpu_percent: 50.0 + (i as f64 * 10.0),
                memory_mb: 8192 + (i as u64 * 1024),
                network_rx_bytes: i as u64 * 1000000,
                network_tx_bytes: i as u64 * 500000,
                disk_read_bytes: i as u64 * 10000000,
                disk_write_bytes: i as u64 * 5000000,
                gpu_usage: vec![],
            }),
            custom_metrics: std::collections::HashMap::new(),
        };

        tx.send(telemetry).await.expect("Failed to send telemetry");
    }

    drop(tx);

    let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
    let response = context
        .client
        .ingest_telemetry(stream)
        .await
        .expect("Failed to ingest telemetry")
        .into_inner();

    assert_eq!(response.events_received, 3, "Should receive 3 events");
    assert_eq!(response.events_processed, 3, "Should process 3 events");
    assert_eq!(response.events_failed, 0, "Should have no failures");
    assert!(
        response.last_processed.is_some(),
        "Should have processing timestamp"
    );

    context.cleanup().await;
}

#[tokio::test]
async fn test_usage_aggregation_in_report() {
    let mut context = TestContext::new().await;
    let user_id = "test_usage_aggregation";

    context.create_test_user(user_id, "1000.0").await;

    let rental_id = Uuid::new_v4().to_string();
    let track_request = TrackRentalRequest {
        rental_id: rental_id.clone(),
        user_id: user_id.to_string(),
        start_time: None,
        metadata: std::collections::HashMap::new(),
        resource_spec: None,
        cloud_type: Some(CloudType::Community(CommunityCloudData {
            node_id: "node_agg".to_string(),
            validator_id: "validator_agg".to_string(),
            base_price_per_gpu: 2.5,
            gpu_count: 1,
            miner_uid: 1,
            miner_hotkey: "test_hotkey".to_string(),
        })),
    };

    let track_response = context
        .client
        .track_rental(track_request)
        .await
        .expect("Failed to track rental")
        .into_inner();

    let rental_uuid =
        Uuid::parse_str(&track_response.tracking_id).expect("Failed to parse rental UUID");

    let cpu_values = [25.0, 50.0, 75.0, 100.0];
    let memory_values = [8.0, 12.0, 16.0, 20.0];

    for (i, (cpu, memory)) in cpu_values.iter().zip(memory_values.iter()).enumerate() {
        sqlx::query(
            "INSERT INTO billing.usage_events (event_id, rental_id, user_id, node_id, validator_id, event_type, event_data, timestamp)
             VALUES ($1, $2, $3, $4, $5, $6, $7, NOW() - INTERVAL '1 minute' * $8)"
        )
        .bind(Uuid::new_v4())
        .bind(rental_uuid)
        .bind(user_id)
        .bind("node_agg")
        .bind("validator_agg")
        .bind("telemetry")
        .bind(serde_json::json!({
            "cpu_percent": cpu / 100.0,
            "memory_gb": memory,
            "network_gb": 0.1 * (i + 1) as f64,
            "gpu_hours": 0.8,
            "disk_io_gb": 0.05 * (i + 1) as f64
        }))
        .bind(i as i32)
        .execute(&context.pool)
        .await
        .expect("Failed to insert usage event");
    }

    let request = UsageReportRequest {
        rental_id: track_response.tracking_id.clone(),
        start_time: None,
        end_time: None,
        aggregation: 0,
    };

    let response = context
        .client
        .get_usage_report(request)
        .await
        .expect("Failed to get usage report")
        .into_inner();

    assert_eq!(response.data_points.len(), 4, "Should have 4 data points");

    if let Some(summary) = response.summary {
        let expected_avg_cpu = (25.0 + 50.0 + 75.0 + 100.0) / 4.0;
        assert!(
            (summary.avg_cpu_percent - expected_avg_cpu).abs() < 0.1,
            "Average CPU should be correct"
        );

        let expected_avg_memory = ((8.0 + 12.0 + 16.0 + 20.0) / 4.0 * 1024.0) as u64;
        assert_eq!(
            summary.avg_memory_mb, expected_avg_memory,
            "Average memory should be correct"
        );

        assert!(summary.total_network_bytes > 0, "Should have network usage");
        assert!(summary.total_disk_bytes > 0, "Should have disk usage");
    } else {
        panic!("Summary should be present");
    }

    context.cleanup().await;
}

#[tokio::test]
async fn test_usage_report_calculates_cost() {
    let mut context = TestContext::new().await;
    let user_id = "test_usage_cost";

    context.create_test_user(user_id, "1000.0").await;

    let rental_id = Uuid::new_v4().to_string();
    let track_request = TrackRentalRequest {
        rental_id: rental_id.clone(),
        user_id: user_id.to_string(),
        start_time: None,
        metadata: std::collections::HashMap::new(),
        resource_spec: None,
        cloud_type: Some(CloudType::Community(CommunityCloudData {
            node_id: "node_cost".to_string(),
            validator_id: "validator_cost".to_string(),
            base_price_per_gpu: 2.5,
            gpu_count: 1,
            miner_uid: 1,
            miner_hotkey: "test_hotkey".to_string(),
        })),
    };

    let track_response = context
        .client
        .track_rental(track_request)
        .await
        .expect("Failed to track rental")
        .into_inner();

    let activate_request = UpdateRentalStatusRequest {
        rental_id: track_response.tracking_id.clone(),
        status: RentalStatus::Active.into(),
        timestamp: None,
        reason: String::new(),
    };

    context
        .client
        .update_rental_status(activate_request)
        .await
        .expect("Failed to activate rental");

    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    let request = UsageReportRequest {
        rental_id: track_response.tracking_id.clone(),
        start_time: None,
        end_time: None,
        aggregation: 0,
    };

    let response = context
        .client
        .get_usage_report(request)
        .await
        .expect("Failed to get usage report")
        .into_inner();

    assert!(!response.total_cost.is_empty(), "Should have total cost");

    let total_cost: f64 = response
        .total_cost
        .parse()
        .expect("Cost should be valid number");
    assert!(total_cost >= 0.0, "Cost should be non-negative");

    context.cleanup().await;
}

#[tokio::test]
async fn test_usage_report_for_nonexistent_rental() {
    let mut context = TestContext::new().await;

    let fake_rental_id = Uuid::new_v4().to_string();
    let request = UsageReportRequest {
        rental_id: fake_rental_id,
        start_time: None,
        end_time: None,
        aggregation: 0,
    };

    let result = context.client.get_usage_report(request).await;

    assert!(result.is_err(), "Should fail for nonexistent rental");
    let error = result.unwrap_err();
    assert!(error.message().contains("not found") || error.message().contains("Rental"));

    context.cleanup().await;
}

#[tokio::test]
async fn test_single_gpu_rental_cost() {
    let mut context = TestContext::new().await;
    let user_id = "test_single_gpu";

    context.create_test_user(user_id, "1000.0").await;

    let rental_id = Uuid::new_v4().to_string();
    let track_request = TrackRentalRequest {
        rental_id: rental_id.clone(),
        user_id: user_id.to_string(),
        start_time: None,
        metadata: std::collections::HashMap::new(),
        resource_spec: None,
        cloud_type: Some(CloudType::Community(CommunityCloudData {
            node_id: "node_single_gpu".to_string(),
            validator_id: "validator_single_gpu".to_string(),
            base_price_per_gpu: 2.5,
            gpu_count: 1,
            miner_uid: 1,
            miner_hotkey: "test_hotkey".to_string(),
        })),
    };

    let track_response = context
        .client
        .track_rental(track_request)
        .await
        .expect("Failed to track rental")
        .into_inner();

    let rental_uuid =
        Uuid::parse_str(&track_response.tracking_id).expect("Failed to parse rental UUID");

    sqlx::query(
        "INSERT INTO billing.usage_events (event_id, rental_id, user_id, node_id, validator_id, event_type, event_data, timestamp)
         VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())"
    )
    .bind(Uuid::new_v4())
    .bind(rental_uuid)
    .bind(user_id)
    .bind("node_single_gpu")
    .bind("validator_single_gpu")
    .bind("telemetry")
    .bind(serde_json::json!({
        "gpu_hours": 1.0,
        "gpu_metrics": {
            "gpu_count": 1
        }
    }))
    .execute(&context.pool)
    .await
    .expect("Failed to insert usage event");

    let usage_metrics = context
        .get_usage_for_rental(&track_response.tracking_id)
        .await;

    assert_eq!(usage_metrics.gpu_count, 1, "Should have 1 GPU");

    context.cleanup().await;
}

#[tokio::test]
async fn test_multi_gpu_rental_with_volume_discount() {
    let mut context = TestContext::new().await;
    let user_id = "test_multi_gpu";

    context.create_test_user(user_id, "1000.0").await;

    let rental_id = Uuid::new_v4().to_string();
    let track_request = TrackRentalRequest {
        rental_id: rental_id.clone(),
        user_id: user_id.to_string(),
        start_time: None,
        metadata: std::collections::HashMap::new(),
        resource_spec: None,
        cloud_type: Some(CloudType::Community(CommunityCloudData {
            node_id: "node_multi_gpu".to_string(),
            validator_id: "validator_multi_gpu".to_string(),
            base_price_per_gpu: 2.5,
            gpu_count: 1,
            miner_uid: 1,
            miner_hotkey: "test_hotkey".to_string(),
        })),
    };

    let track_response = context
        .client
        .track_rental(track_request)
        .await
        .expect("Failed to track rental")
        .into_inner();

    let rental_uuid =
        Uuid::parse_str(&track_response.tracking_id).expect("Failed to parse rental UUID");

    sqlx::query(
        "INSERT INTO billing.usage_events (event_id, rental_id, user_id, node_id, validator_id, event_type, event_data, timestamp)
         VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())"
    )
    .bind(Uuid::new_v4())
    .bind(rental_uuid)
    .bind(user_id)
    .bind("node_multi_gpu")
    .bind("validator_multi_gpu")
    .bind("telemetry")
    .bind(serde_json::json!({
        "gpu_hours": 1.0,
        "gpu_metrics": {
            "gpu_count": 4
        }
    }))
    .execute(&context.pool)
    .await
    .expect("Failed to insert usage event");

    let usage_metrics = context
        .get_usage_for_rental(&track_response.tracking_id)
        .await;

    assert_eq!(usage_metrics.gpu_count, 4, "Should have 4 GPUs");

    context.cleanup().await;
}

#[tokio::test]
async fn test_gpu_count_extracted_from_telemetry() {
    let mut context = TestContext::new().await;
    let user_id = "test_gpu_count_telemetry";

    context.create_test_user(user_id, "1000.0").await;

    let rental_id = Uuid::new_v4().to_string();
    let track_request = TrackRentalRequest {
        rental_id: rental_id.clone(),
        user_id: user_id.to_string(),
        start_time: None,
        metadata: std::collections::HashMap::new(),
        resource_spec: None,
        cloud_type: Some(CloudType::Community(CommunityCloudData {
            node_id: "node_gpu_count".to_string(),
            validator_id: "validator_gpu_count".to_string(),
            base_price_per_gpu: 2.5,
            gpu_count: 1,
            miner_uid: 1,
            miner_hotkey: "test_hotkey".to_string(),
        })),
    };

    let track_response = context
        .client
        .track_rental(track_request)
        .await
        .expect("Failed to track rental")
        .into_inner();

    let rental_uuid =
        Uuid::parse_str(&track_response.tracking_id).expect("Failed to parse rental UUID");

    sqlx::query(
        "INSERT INTO billing.usage_events (event_id, rental_id, user_id, node_id, validator_id, event_type, event_data, timestamp)
         VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())"
    )
    .bind(Uuid::new_v4())
    .bind(rental_uuid)
    .bind(user_id)
    .bind("node_gpu_count")
    .bind("validator_gpu_count")
    .bind("telemetry")
    .bind(serde_json::json!({
        "cpu_percent": 0.8,
        "memory_gb": 32.0,
        "network_gb": 1.0,
        "gpu_hours": 1.0,
        "gpu_metrics": {
            "gpu_count": 8
        },
        "disk_io_gb": 0.5
    }))
    .execute(&context.pool)
    .await
    .expect("Failed to insert usage event");

    let usage_metrics = context
        .get_usage_for_rental(&track_response.tracking_id)
        .await;

    assert_eq!(
        usage_metrics.gpu_count, 8,
        "GPU count should be extracted from telemetry"
    );
    assert!(
        usage_metrics.gpu_hours.is_sign_positive(),
        "GPU hours should be positive"
    );

    context.cleanup().await;
}

#[tokio::test]
async fn test_cost_breakdown_includes_volume_discount() {
    let mut context = TestContext::new().await;
    let user_id = "test_volume_discount_breakdown";

    context.create_test_user(user_id, "1000.0").await;

    let rental_id = Uuid::new_v4().to_string();
    let track_request = TrackRentalRequest {
        rental_id: rental_id.clone(),
        user_id: user_id.to_string(),
        start_time: None,
        metadata: std::collections::HashMap::new(),
        resource_spec: None,
        cloud_type: Some(CloudType::Community(CommunityCloudData {
            node_id: "node_discount".to_string(),
            validator_id: "validator_discount".to_string(),
            base_price_per_gpu: 2.5,
            gpu_count: 1,
            miner_uid: 1,
            miner_hotkey: "test_hotkey".to_string(),
        })),
    };

    let track_response = context
        .client
        .track_rental(track_request)
        .await
        .expect("Failed to track rental")
        .into_inner();

    let activate_request = UpdateRentalStatusRequest {
        rental_id: track_response.tracking_id.clone(),
        status: RentalStatus::Active.into(),
        timestamp: None,
        reason: String::new(),
    };

    context
        .client
        .update_rental_status(activate_request)
        .await
        .expect("Failed to activate rental");

    let rental_uuid =
        Uuid::parse_str(&track_response.tracking_id).expect("Failed to parse rental UUID");

    sqlx::query(
        "INSERT INTO billing.usage_events (event_id, rental_id, user_id, node_id, validator_id, event_type, event_data, timestamp)
         VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())"
    )
    .bind(Uuid::new_v4())
    .bind(rental_uuid)
    .bind(user_id)
    .bind("node_discount")
    .bind("validator_discount")
    .bind("telemetry")
    .bind(serde_json::json!({
        "gpu_hours": 1.0,
        "gpu_metrics": {
            "gpu_count": 2
        }
    }))
    .execute(&context.pool)
    .await
    .expect("Failed to insert usage event");

    let usage_metrics = context
        .get_usage_for_rental(&track_response.tracking_id)
        .await;

    assert_eq!(
        usage_metrics.gpu_count, 2,
        "Should have 2 GPUs for volume discount"
    );

    context.cleanup().await;
}
