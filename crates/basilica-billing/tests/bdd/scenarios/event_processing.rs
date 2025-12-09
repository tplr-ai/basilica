use crate::bdd::TestContext;
use basilica_protocol::billing::{
    track_rental_request::CloudType, CommunityCloudData, FinalizeRentalRequest, RentalStatus,
    TrackRentalRequest, UpdateRentalStatusRequest,
};
use uuid::Uuid;

#[tokio::test]
async fn test_rental_start_event_created() {
    let mut context = TestContext::new().await;
    let user_id = "test_rental_start_event";

    context.create_test_user(user_id, "1000.0").await;

    let rental_id = Uuid::new_v4().to_string();
    let request = TrackRentalRequest {
        rental_id: rental_id.clone(),
        user_id: user_id.to_string(),
        start_time: None,
        metadata: std::collections::HashMap::new(),
        resource_spec: None,
        cloud_type: Some(CloudType::Community(CommunityCloudData {
            node_id: "node_event_001".to_string(),
            validator_id: "validator_event_001".to_string(),
            base_price_per_gpu: 2.5,
            gpu_count: 1,
            miner_uid: 1,
            miner_hotkey: "test_hotkey".to_string(),
        })),
    };

    let response = context
        .client
        .track_rental(request)
        .await
        .expect("Failed to track rental")
        .into_inner();

    let event_count = context.count_usage_events(&response.tracking_id).await;
    assert!(event_count > 0, "Should create rental start event");

    let event_exists = sqlx::query_scalar::<_, bool>(
        "SELECT EXISTS(
            SELECT 1 FROM billing.usage_events
            WHERE rental_id = $1::uuid
            AND event_type = 'rental_start'
        )",
    )
    .bind(&response.tracking_id)
    .fetch_one(&context.pool)
    .await
    .expect("Failed to check event");

    assert!(event_exists, "Rental start event should exist");

    let event_data = sqlx::query_scalar::<_, serde_json::Value>(
        "SELECT event_data FROM billing.usage_events
         WHERE rental_id = $1::uuid
         AND event_type = 'rental_start'",
    )
    .bind(&response.tracking_id)
    .fetch_one(&context.pool)
    .await
    .expect("Failed to get event data");

    assert!(
        event_data.get("hourly_rate").is_some(),
        "Event should contain hourly rate"
    );

    context.cleanup().await;
}

#[tokio::test]
async fn test_status_change_events_tracked() {
    let mut context = TestContext::new().await;
    let user_id = "test_status_events";

    context.create_test_user(user_id, "1000.0").await;

    let rental_id = Uuid::new_v4().to_string();
    let track_request = TrackRentalRequest {
        rental_id: rental_id.clone(),
        user_id: user_id.to_string(),
        start_time: None,
        metadata: std::collections::HashMap::new(),
        resource_spec: None,
        cloud_type: Some(CloudType::Community(CommunityCloudData {
            node_id: "node_status".to_string(),
            validator_id: "validator_status".to_string(),
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

    let statuses = vec![
        (RentalStatus::Active, "Container started"),
        (RentalStatus::Stopping, "User requested stop"),
        (RentalStatus::Stopped, "Container terminated"),
    ];

    for (status, reason) in statuses {
        let update_request = UpdateRentalStatusRequest {
            rental_id: track_response.tracking_id.clone(),
            status: status.into(),
            timestamp: None,
            reason: reason.to_string(),
        };

        context
            .client
            .update_rental_status(update_request)
            .await
            .expect("Failed to update status");
    }

    let status_events = sqlx::query_scalar::<_, i64>(
        "SELECT COUNT(*) FROM billing.usage_events
         WHERE rental_id = $1::uuid
         AND event_type = 'status_change'",
    )
    .bind(&track_response.tracking_id)
    .fetch_one(&context.pool)
    .await
    .expect("Failed to count status events");

    assert_eq!(status_events, 3, "Should have 3 status change events");

    let last_event = sqlx::query_as::<_, (serde_json::Value,)>(
        "SELECT event_data FROM billing.usage_events
         WHERE rental_id = $1::uuid
         AND event_type = 'status_change'
         ORDER BY timestamp DESC LIMIT 1",
    )
    .bind(&track_response.tracking_id)
    .fetch_one(&context.pool)
    .await
    .expect("Failed to get last event");

    assert_eq!(
        last_event.0.get("new_status").and_then(|v| v.as_str()),
        Some("completed"),
        "Last status should be completed"
    );

    context.cleanup().await;
}

#[tokio::test]
async fn test_event_timestamps_ordered() {
    let mut context = TestContext::new().await;
    let user_id = "test_event_order";

    context.create_test_user(user_id, "1000.0").await;

    let rental_id = Uuid::new_v4().to_string();
    let track_request = TrackRentalRequest {
        rental_id: rental_id.clone(),
        user_id: user_id.to_string(),
        start_time: None,
        metadata: std::collections::HashMap::new(),
        resource_spec: None,
        cloud_type: Some(CloudType::Community(CommunityCloudData {
            node_id: "node_order".to_string(),
            validator_id: "validator_order".to_string(),
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

    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    let update_request = UpdateRentalStatusRequest {
        rental_id: track_response.tracking_id.clone(),
        status: RentalStatus::Active.into(),
        timestamp: None,
        reason: String::new(),
    };

    context
        .client
        .update_rental_status(update_request)
        .await
        .expect("Failed to update status");

    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    let finalize_request = FinalizeRentalRequest {
        rental_id: track_response.tracking_id.clone(),
        end_time: None,
        termination_reason: String::new(),
        target_status: basilica_protocol::billing::RentalStatus::Stopped.into(),
    };

    context
        .client
        .finalize_rental(finalize_request)
        .await
        .expect("Failed to finalize rental");

    let timestamps = sqlx::query_as::<_, (chrono::DateTime<chrono::Utc>,)>(
        "SELECT timestamp FROM billing.usage_events
         WHERE rental_id = $1::uuid
         ORDER BY timestamp ASC",
    )
    .bind(&track_response.tracking_id)
    .fetch_all(&context.pool)
    .await
    .expect("Failed to get timestamps");

    assert!(timestamps.len() >= 2, "Should have multiple events");

    for i in 1..timestamps.len() {
        assert!(
            timestamps[i].0 >= timestamps[i - 1].0,
            "Events should be chronologically ordered"
        );
    }

    context.cleanup().await;
}

#[tokio::test]
async fn test_event_processing_flags() {
    let mut context = TestContext::new().await;
    let user_id = "test_processing_flags";

    context.create_test_user(user_id, "1000.0").await;

    let rental_id = Uuid::new_v4().to_string();
    let track_request = TrackRentalRequest {
        rental_id: rental_id.clone(),
        user_id: user_id.to_string(),
        start_time: None,
        metadata: std::collections::HashMap::new(),
        resource_spec: None,
        cloud_type: Some(CloudType::Community(CommunityCloudData {
            node_id: "node_flags".to_string(),
            validator_id: "validator_flags".to_string(),
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

    let unprocessed_events = sqlx::query_scalar::<_, i64>(
        "SELECT COUNT(*) FROM billing.usage_events
         WHERE rental_id = $1::uuid
         AND processed = false",
    )
    .bind(&track_response.tracking_id)
    .fetch_one(&context.pool)
    .await
    .expect("Failed to count unprocessed events");

    assert!(unprocessed_events > 0, "New events should be unprocessed");

    let null_batch_events = sqlx::query_scalar::<_, i64>(
        "SELECT COUNT(*) FROM billing.usage_events
         WHERE rental_id = $1::uuid
         AND batch_id IS NULL",
    )
    .bind(&track_response.tracking_id)
    .fetch_one(&context.pool)
    .await
    .expect("Failed to count unbatched events");

    assert_eq!(
        null_batch_events, unprocessed_events,
        "Unprocessed events should not have batch_id"
    );

    context.cleanup().await;
}

#[tokio::test]
async fn test_event_metadata_preserved() {
    let mut context = TestContext::new().await;
    let user_id = "test_event_metadata";

    context.create_test_user(user_id, "1000.0").await;

    let rental_id = Uuid::new_v4().to_string();
    let node_id = "node_metadata_001";
    let validator_id = "validator_metadata_001";

    let track_request = TrackRentalRequest {
        rental_id: rental_id.clone(),
        user_id: user_id.to_string(),
        start_time: None,
        metadata: std::collections::HashMap::new(),
        resource_spec: None,
        cloud_type: Some(CloudType::Community(CommunityCloudData {
            node_id: node_id.to_string(),
            validator_id: validator_id.to_string(),
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

    let event = sqlx::query_as::<_, (String, String, String, serde_json::Value)>(
        "SELECT user_id, node_id, validator_id, event_data
         FROM billing.usage_events
         WHERE rental_id = $1::uuid
         AND event_type = 'rental_start'",
    )
    .bind(&track_response.tracking_id)
    .fetch_one(&context.pool)
    .await
    .expect("Failed to get event");

    assert_eq!(event.0, user_id, "User ID should be preserved");
    assert_eq!(event.1, node_id, "Node ID should be preserved");
    assert_eq!(event.2, validator_id, "Validator ID should be preserved");

    assert_eq!(
        event.3.get("hourly_rate").and_then(|v| v.as_str()),
        Some("3.5"),
        "Hourly rate should be H100 package rate in event data"
    );

    context.cleanup().await;
}

#[tokio::test]
async fn test_concurrent_event_creation() {
    let context = TestContext::new().await;
    let user_id = "test_concurrent_events";

    context.create_test_user(user_id, "5000.0").await;

    let mut handles = Vec::new();
    let client = context.client.clone();

    for i in 0..5 {
        let mut client = client.clone();
        let user_id = user_id.to_string();

        let handle = tokio::spawn(async move {
            let rental_id = Uuid::new_v4().to_string();
            let request = TrackRentalRequest {
                rental_id,
                user_id,
                start_time: None,
                metadata: std::collections::HashMap::new(),
                resource_spec: None,
                cloud_type: Some(CloudType::Community(CommunityCloudData {
                    node_id: format!("node_concurrent_{}", i),
                    validator_id: format!("validator_concurrent_{}", i),
                    base_price_per_gpu: 2.5,
                    gpu_count: 1,
                    miner_uid: 1,
                    miner_hotkey: "test_hotkey".to_string(),
                })),
            };

            client
                .track_rental(request)
                .await
                .expect("Failed to track rental")
                .into_inner()
        });

        handles.push(handle);
    }

    let results: Vec<_> = futures::future::join_all(handles)
        .await
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .expect("Failed to join tasks");

    assert_eq!(results.len(), 5, "All rentals should be created");

    for result in results {
        assert!(result.success, "Each rental should succeed");

        let event_count = context.count_usage_events(&result.tracking_id).await;
        assert!(event_count > 0, "Each rental should have events");
    }

    let total_events = sqlx::query_scalar::<_, i64>(
        "SELECT COUNT(*) FROM billing.usage_events WHERE user_id = $1",
    )
    .bind(user_id)
    .fetch_one(&context.pool)
    .await
    .expect("Failed to count total events");

    assert!(
        total_events >= 5,
        "Should have at least one event per rental"
    );

    context.cleanup().await;
}
