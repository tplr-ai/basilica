use crate::bdd::TestContext;
use basilica_protocol::billing::{
    track_rental_request::CloudType, CommunityCloudData, FinalizeRentalRequest,
    GetActiveRentalsRequest, GpuSpec, RentalStatus, ResourceSpec, TrackRentalRequest,
    UpdateRentalStatusRequest,
};
use uuid::Uuid;

#[tokio::test]
async fn test_track_rental_creates_new_rental() {
    let mut context = TestContext::new().await;
    let user_id = "test_rental_track_001";
    let node_id = "node_001";
    let validator_id = "validator_001";

    context.create_test_user(user_id, "1000.0").await;

    let rental_id = Uuid::new_v4().to_string();
    let request = TrackRentalRequest {
        rental_id: rental_id.clone(),
        user_id: user_id.to_string(),
        start_time: None,
        metadata: std::collections::HashMap::new(),
        resource_spec: Some(ResourceSpec {
            cpu_cores: 8,
            memory_mb: 32768,
            gpus: vec![GpuSpec {
                model: "NVIDIA H100".to_string(),
                memory_mb: 81920,
                count: 1,
            }],
            disk_gb: 100,
            network_bandwidth_mbps: 1000,
        }),
        cloud_type: Some(CloudType::Community(CommunityCloudData {
            node_id: node_id.to_string(),
            validator_id: validator_id.to_string(),
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

    assert!(response.success, "Tracking rental should succeed");
    assert!(
        !response.tracking_id.is_empty(),
        "Should return tracking ID"
    );

    assert!(
        context.rental_exists(&response.tracking_id).await,
        "Rental should exist in database"
    );

    context.cleanup().await;
}

#[tokio::test]
async fn test_track_rental_fails_with_insufficient_balance() {
    let mut context = TestContext::new().await;
    let user_id = "test_rental_insufficient";

    context.create_test_user(user_id, "10.0").await;
    let initial_balance = context.get_user_balance(user_id).await;

    let rental_id = Uuid::new_v4().to_string();
    let request = TrackRentalRequest {
        rental_id: rental_id.clone(),
        user_id: user_id.to_string(),
        start_time: None,
        metadata: std::collections::HashMap::new(),
        resource_spec: None,
        cloud_type: Some(CloudType::Community(CommunityCloudData {
            node_id: "node_002".to_string(),
            validator_id: "validator_002".to_string(),
            base_price_per_gpu: 2.5,
            gpu_count: 1,
            miner_uid: 1,
            miner_hotkey: "test_hotkey".to_string(),
        })),
    };

    let result = context.client.track_rental(request).await;
    assert!(
        result.is_ok(),
        "Pay-as-you-go: rental creation should succeed even with low balance"
    );

    let tracking_id = result.unwrap().into_inner().tracking_id;
    assert!(
        context.rental_exists(&tracking_id).await,
        "Rental should be created"
    );

    let status = context.get_rental_status(&tracking_id).await;
    assert_eq!(
        status,
        Some("pending".to_string()),
        "Rental should start in pending state"
    );

    let final_balance = context.get_user_balance(user_id).await;
    assert_eq!(
        final_balance, initial_balance,
        "Balance unchanged - pay-as-you-go deducts on telemetry, not at creation"
    );

    context.cleanup().await;
}

#[tokio::test]
async fn test_update_rental_status_transitions() {
    let mut context = TestContext::new().await;
    let user_id = "test_status_update";

    context.create_test_user(user_id, "1000.0").await;

    let rental_id = Uuid::new_v4().to_string();
    let track_request = TrackRentalRequest {
        rental_id: rental_id.clone(),
        user_id: user_id.to_string(),
        start_time: None,
        metadata: std::collections::HashMap::new(),
        resource_spec: None,
        cloud_type: Some(CloudType::Community(CommunityCloudData {
            node_id: "node_003".to_string(),
            validator_id: "validator_003".to_string(),
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

    let update_request = UpdateRentalStatusRequest {
        rental_id: track_response.tracking_id.clone(),
        status: RentalStatus::Active.into(),
        timestamp: None,
        reason: "Container started successfully".to_string(),
    };

    let update_response = context
        .client
        .update_rental_status(update_request)
        .await
        .expect("Failed to update status")
        .into_inner();

    assert!(update_response.success, "Status update should succeed");
    assert!(
        update_response.updated_at.is_some(),
        "Should return update timestamp"
    );

    let status = context.get_rental_status(&track_response.tracking_id).await;
    assert_eq!(
        status,
        Some("active".to_string()),
        "Status should be updated to active"
    );

    let stop_request = UpdateRentalStatusRequest {
        rental_id: track_response.tracking_id.clone(),
        status: RentalStatus::Stopping.into(),
        timestamp: None,
        reason: "User requested termination".to_string(),
    };

    let stop_response = context
        .client
        .update_rental_status(stop_request)
        .await
        .expect("Failed to stop rental")
        .into_inner();

    assert!(stop_response.success);

    let final_status = context.get_rental_status(&track_response.tracking_id).await;
    assert_eq!(
        final_status,
        Some("terminating".to_string()),
        "Status should be terminating"
    );

    context.cleanup().await;
}

#[tokio::test]
async fn test_get_active_rentals_by_user() {
    let mut context = TestContext::new().await;
    let user_id = "test_active_rentals_user";

    context.create_test_user(user_id, "1000.0").await;

    for i in 0..3 {
        let rental_id = Uuid::new_v4().to_string();
        let request = TrackRentalRequest {
            rental_id,
            user_id: user_id.to_string(),
            start_time: None,
            metadata: std::collections::HashMap::new(),
            resource_spec: None,
            cloud_type: Some(CloudType::Community(CommunityCloudData {
                node_id: format!("node_{}", i),
                validator_id: format!("validator_{}", i),
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

        if i < 2 {
            let update = UpdateRentalStatusRequest {
                rental_id: response.tracking_id,
                status: RentalStatus::Active.into(),
                timestamp: None,
                reason: String::new(),
            };

            context
                .client
                .update_rental_status(update)
                .await
                .expect("Failed to activate rental");
        }
    }

    let request = GetActiveRentalsRequest {
        user_id: user_id.to_string(),
        limit: 100,
        offset: 0,
        status_filter: vec![],
    };

    let response = context
        .client
        .get_active_rentals(request)
        .await
        .expect("Failed to get active rentals")
        .into_inner();

    assert_eq!(
        response.rentals.len(),
        3,
        "Should return 3 rentals (2 active + 1 pending)"
    );
    assert_eq!(response.total_count, 3, "Total count should be 3");

    let mut active_count = 0;
    let mut pending_count = 0;

    for rental in &response.rentals {
        assert_eq!(rental.user_id, user_id);
        match rental.status() {
            RentalStatus::Active => active_count += 1,
            RentalStatus::Pending => pending_count += 1,
            _ => panic!("Unexpected rental status"),
        }
        assert!(rental.start_time.is_some());
        assert!(rental.last_updated.is_some());
    }

    assert_eq!(active_count, 2, "Should have 2 active rentals");
    assert_eq!(pending_count, 1, "Should have 1 pending rental");

    context.cleanup().await;
}

#[tokio::test]
async fn test_finalize_rental_charges_correct_amount() {
    let mut context = TestContext::new().await;
    let user_id = "test_finalize_rental";

    context.create_test_user(user_id, "1000.0").await;

    let rental_id = Uuid::new_v4().to_string();
    let track_request = TrackRentalRequest {
        rental_id: rental_id.clone(),
        user_id: user_id.to_string(),
        start_time: None,
        metadata: std::collections::HashMap::new(),
        resource_spec: None,
        cloud_type: Some(CloudType::Community(CommunityCloudData {
            node_id: "node_final".to_string(),
            validator_id: "validator_final".to_string(),
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

    let initial_balance = context.get_user_balance(user_id).await;

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

    let finalize_request = FinalizeRentalRequest {
        rental_id: track_response.tracking_id.clone(),
        end_time: None,
        termination_reason: String::new(),
        target_status: basilica_protocol::billing::RentalStatus::Stopped.into(),
    };

    let finalize_response = context
        .client
        .finalize_rental(finalize_request)
        .await
        .expect("Failed to finalize rental")
        .into_inner();

    assert!(finalize_response.success, "Finalization should succeed");
    // finalize_rental is now a no-op that returns telemetry-based charges
    // It doesn't charge anything itself - all charging happens via telemetry events
    assert_eq!(
        finalize_response.charged_amount, "0.00",
        "No additional charge at finalization"
    );
    assert_eq!(
        finalize_response.refunded_amount, "0.00",
        "No refund in pay-as-you-go model"
    );

    // Balance should remain unchanged since finalize_rental doesn't charge
    let final_balance = context.get_user_balance(user_id).await;
    assert_eq!(
        final_balance, initial_balance,
        "Balance unchanged - charging happens via telemetry"
    );

    let rental_status = context.get_rental_status(&track_response.tracking_id).await;
    assert_eq!(
        rental_status,
        Some("completed".to_string()),
        "Rental should be completed"
    );

    context.cleanup().await;
}
