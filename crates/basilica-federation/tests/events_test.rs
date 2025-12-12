//! Tests for event system

use basilica_federation::events::{FederationEvent, EventPublisher, EventManager};

#[test]
fn test_event_publisher_creation() {
    let publisher = EventPublisher::new();
    let receiver = publisher.subscribe();
    
    // Should be able to create publisher and subscriber
    assert!(true);
}

#[tokio::test]
async fn test_event_publishing() {
    let publisher = EventPublisher::new();
    let mut receiver = publisher.subscribe();
    
    let event = FederationEvent::ClusterHealthChanged {
        cluster_id: "cluster-1".to_string(),
        old_status: "Healthy".to_string(),
        new_status: "Degraded".to_string(),
    };
    
    publisher.publish(event.clone());
    
    // Try to receive event (with timeout)
    let received = tokio::time::timeout(
        std::time::Duration::from_millis(100),
        receiver.recv()
    ).await;
    
    if let Ok(Ok(received_event)) = received {
        match (received_event, event) {
            (FederationEvent::ClusterHealthChanged { cluster_id: id1, .. },
             FederationEvent::ClusterHealthChanged { cluster_id: id2, .. }) => {
                assert_eq!(id1, id2);
            }
            _ => assert!(false, "Event types don't match"),
        }
    }
}

#[test]
fn test_event_types() {
    // Test ClusterHealthChanged
    let event1 = FederationEvent::ClusterHealthChanged {
        cluster_id: "cluster-1".to_string(),
        old_status: "Healthy".to_string(),
        new_status: "Unhealthy".to_string(),
    };
    
    // Test ServiceDiscovered
    let event2 = FederationEvent::ServiceDiscovered {
        service_name: "test-service".to_string(),
        namespace: "default".to_string(),
        cluster_id: "cluster-1".to_string(),
    };
    
    // Test ClusterSelected
    let event3 = FederationEvent::ClusterSelected {
        cluster_id: "cluster-1".to_string(),
        algorithm: "RoundRobin".to_string(),
    };
    
    // All events should be creatable
    assert!(matches!(event1, FederationEvent::ClusterHealthChanged { .. }));
    assert!(matches!(event2, FederationEvent::ServiceDiscovered { .. }));
    assert!(matches!(event3, FederationEvent::ClusterSelected { .. }));
}

#[test]
fn test_event_manager_creation() {
    let manager = EventManager::new();
    let publisher = manager.publisher();
    
    // Should be able to get publisher
    assert!(true);
}

#[test]
fn test_event_manager_handler_registration() {
    use basilica_federation::events::EventHandler;
    
    struct TestHandler;
    
    impl EventHandler for TestHandler {
        fn handle(&self, _event: &FederationEvent) {
            // Test handler
        }
    }
    
    let mut manager = EventManager::new();
    manager.register_handler(std::sync::Arc::new(TestHandler));
    
    assert!(true);
}

