use crate::bdd::TestContext;
use basilica_protocol::billing::{
    get_active_rentals_request::Filter, FinalizeRentalRequest, GetActiveRentalsRequest, GpuSpec,
    ReleaseReservationRequest, RentalStatus, ResourceSpec, TrackRentalRequest,
    UpdateRentalStatusRequest,
};
use uuid::Uuid;

// Helper function to convert hours to protobuf Duration
fn hours_to_duration(hours: u32) -> prost_types::Duration {
    prost_types::Duration {
        seconds: (hours as i64) * 3600,
        nanos: 0,
    }
}

#[tokio::test]
async fn test_track_rental_creates_new_rental_with_reservation() {
    let mut context = TestContext::new().await;
    let user_id = "test_rental_track_001";
    let node_id = "node_001";
    let validator_id = "validator_001";

    context.create_test_user(user_id, "1000.0").await;

    let rental_id = Uuid::new_v4().to_string();
    let request = TrackRentalRequest {
        rental_id: rental_id.clone(),
        user_id: user_id.to_string(),
        node_id: node_id.to_string(),
        validator_id: validator_id.to_string(),
        hourly_rate: "10.0".to_string(),
        max_duration: Some(hours_to_duration(24)),
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
        !response.reservation_id.is_empty(),
        "Should return reservation ID"
    );
    assert_eq!(
        response.estimated_cost, "240",
        "Estimated cost should be 10 * 24"
    );

    assert!(
        context.rental_exists(&response.tracking_id).await,
        "Rental should exist in database"
    );
    assert!(
        context.reservation_exists(&response.reservation_id).await,
        "Reservation should exist"
    );

    let reserved = context.get_reserved_balance(user_id).await;
    assert_eq!(
        reserved,
        rust_decimal::Decimal::from(240),
        "Should reserve estimated cost"
    );

    context.cleanup().await;
}

#[tokio::test]
async fn test_track_rental_fails_with_insufficient_balance() {
    let mut context = TestContext::new().await;
    let user_id = "test_rental_insufficient";

    context.create_test_user(user_id, "10.0").await;

    let rental_id = Uuid::new_v4().to_string();
    let request = TrackRentalRequest {
        rental_id: rental_id.clone(),
        user_id: user_id.to_string(),
        node_id: "node_002".to_string(),
        validator_id: "validator_002".to_string(),
        hourly_rate: "100.0".to_string(),
        max_duration: Some(hours_to_duration(10)),
        start_time: None,
        metadata: std::collections::HashMap::new(),
        resource_spec: None,
    };

    let result = context.client.track_rental(request).await;

    assert!(result.is_err(), "Should fail with insufficient balance");
    let error = result.unwrap_err();
    assert!(
        error.message().contains("Insufficient balance"),
        "Error should mention insufficient balance"
    );

    assert!(
        !context.rental_exists(&rental_id).await,
        "Rental should not be created"
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
        node_id: "node_003".to_string(),
        validator_id: "validator_003".to_string(),
        hourly_rate: "5.0".to_string(),
        max_duration: Some(hours_to_duration(10)),
        start_time: None,
        metadata: std::collections::HashMap::new(),
        resource_spec: None,
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
            node_id: format!("node_{}", i),
            validator_id: format!("validator_{}", i),
            hourly_rate: "2.0".to_string(),
            max_duration: Some(hours_to_duration(5)),
            start_time: None,
            metadata: std::collections::HashMap::new(),
            resource_spec: None,
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
        limit: 100,
        offset: 0,
        filter: Some(Filter::UserId(user_id.to_string())),
    };

    let response = context
        .client
        .get_active_rentals(request)
        .await
        .expect("Failed to get active rentals")
        .into_inner();

    assert_eq!(response.rentals.len(), 2, "Should return 2 active rentals");
    assert_eq!(response.total_count, 2, "Total count should be 2");

    for rental in &response.rentals {
        assert_eq!(rental.user_id, user_id);
        assert!(
            rental.status() == RentalStatus::Active || rental.status() == RentalStatus::Pending
        );
        assert!(rental.start_time.is_some());
        assert!(rental.last_updated.is_some());
    }

    context.cleanup().await;
}

#[tokio::test]
async fn test_get_active_rentals_by_node() {
    let mut context = TestContext::new().await;
    let node_id = "node_specific_001";

    for i in 0..2 {
        let user_id = format!("user_{}", i);
        context.create_test_user(&user_id, "1000.0").await;

        let rental_id = Uuid::new_v4().to_string();
        let request = TrackRentalRequest {
            rental_id: rental_id.clone(),
            user_id: user_id.clone(),
            node_id: node_id.to_string(),
            validator_id: "validator_001".to_string(),
            hourly_rate: "3.0".to_string(),
            max_duration: Some(hours_to_duration(8)),
            start_time: None,
            metadata: std::collections::HashMap::new(),
            resource_spec: None,
        };

        let track_response = context
            .client
            .track_rental(request)
            .await
            .expect("Failed to track rental")
            .into_inner();

        // Activate the rental so it shows up in active rentals query
        let update_request = UpdateRentalStatusRequest {
            rental_id: track_response.tracking_id,
            status: RentalStatus::Active.into(),
            reason: String::new(),
            timestamp: None,
        };

        context
            .client
            .update_rental_status(update_request)
            .await
            .expect("Failed to activate rental");
    }

    let request = GetActiveRentalsRequest {
        limit: 100,
        offset: 0,
        filter: Some(Filter::NodeId(node_id.to_string())),
    };

    let response = context
        .client
        .get_active_rentals(request)
        .await
        .expect("Failed to get rentals by node")
        .into_inner();

    assert!(
        response.rentals.len() >= 2,
        "Should return at least 2 rentals for node"
    );

    for rental in &response.rentals {
        assert_eq!(rental.node_id, node_id);
    }

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
        node_id: "node_final".to_string(),
        validator_id: "validator_final".to_string(),
        hourly_rate: "10.0".to_string(),
        max_duration: Some(hours_to_duration(10)),
        start_time: None,
        metadata: std::collections::HashMap::new(),
        resource_spec: None,
    };

    let track_response = context
        .client
        .track_rental(track_request)
        .await
        .expect("Failed to track rental")
        .into_inner();

    let initial_balance = context.get_user_balance(user_id).await;
    let reserved_amount = context.get_reserved_balance(user_id).await;
    assert_eq!(
        reserved_amount,
        rust_decimal::Decimal::from(100),
        "Should reserve 10 * 10"
    );

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
        final_cost: "25.0".to_string(),
        end_time: None,
        termination_reason: String::new(),
    };

    let finalize_response = context
        .client
        .finalize_rental(finalize_request)
        .await
        .expect("Failed to finalize rental")
        .into_inner();

    assert!(finalize_response.success, "Finalization should succeed");
    assert_eq!(
        finalize_response.total_cost, "25",
        "Total cost should match requested"
    );
    assert_eq!(finalize_response.charged_amount, "25", "Should charge 25");
    assert_eq!(
        finalize_response.refunded_amount, "75",
        "Should refund 75 (100 - 25)"
    );

    let final_balance = context.get_user_balance(user_id).await;
    let expected_balance = initial_balance - rust_decimal::Decimal::from(25);
    assert_eq!(
        final_balance, expected_balance,
        "Balance should be reduced by 25"
    );

    let final_reserved = context.get_reserved_balance(user_id).await;
    assert_eq!(
        final_reserved.to_string(),
        "0",
        "Reserved amount should be cleared"
    );

    let rental_status = context.get_rental_status(&track_response.tracking_id).await;
    assert_eq!(
        rental_status,
        Some("completed".to_string()),
        "Rental should be completed"
    );

    context.cleanup().await;
}

#[tokio::test]
async fn test_finalize_rental_without_reservation_charges_directly() {
    let mut context = TestContext::new().await;
    let user_id = "test_finalize_no_reservation";

    context.create_test_user(user_id, "1000.0").await;

    let rental_id = Uuid::new_v4().to_string();

    // Create rental normally with reservation
    let track_request = TrackRentalRequest {
        rental_id: rental_id.clone(),
        user_id: user_id.to_string(),
        node_id: "node_direct".to_string(),
        validator_id: "validator_direct".to_string(),
        hourly_rate: "5.0".to_string(),
        max_duration: Some(hours_to_duration(2)), // Small duration to ensure we have enough balance
        start_time: None,
        metadata: std::collections::HashMap::new(),
        resource_spec: None,
    };

    let track_response = context
        .client
        .track_rental(track_request)
        .await
        .expect("Failed to track rental")
        .into_inner();

    // Activate the rental
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

    // Now simulate a scenario where the reservation was already released/lost
    // by manually releasing it through the API
    let release_request = ReleaseReservationRequest {
        reservation_id: track_response.reservation_id.clone(),
        final_amount: "0".to_string(), // Release with no charge
    };

    context
        .client
        .release_reservation(release_request)
        .await
        .expect("Failed to release reservation");

    let initial_balance = context.get_user_balance(user_id).await;

    // Now finalize the rental - it should charge directly since reservation is gone
    let finalize_request = FinalizeRentalRequest {
        rental_id: track_response.tracking_id.clone(),
        final_cost: "15.0".to_string(),
        end_time: None,
        termination_reason: String::new(),
    };

    let finalize_response = context
        .client
        .finalize_rental(finalize_request)
        .await
        .expect("Failed to finalize rental")
        .into_inner();

    assert!(finalize_response.success);
    assert_eq!(finalize_response.total_cost, "15");
    assert_eq!(finalize_response.charged_amount, "15");
    assert_eq!(
        finalize_response.refunded_amount, "0",
        "No refund without reservation"
    );

    let final_balance = context.get_user_balance(user_id).await;
    let expected_balance = initial_balance - rust_decimal::Decimal::from(15);
    assert_eq!(
        final_balance, expected_balance,
        "Balance should be reduced by 15"
    );

    context.cleanup().await;
}
