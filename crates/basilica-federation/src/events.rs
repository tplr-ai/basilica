//! Federation event system

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::broadcast;
use tracing::{debug, info};

/// Federation event types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FederationEvent {
    /// Cluster health changed
    ClusterHealthChanged {
        cluster_id: String,
        old_status: String,
        new_status: String,
    },
    
    /// Service discovered
    ServiceDiscovered {
        service_name: String,
        namespace: String,
        cluster_id: String,
    },
    
    /// Service removed
    ServiceRemoved {
        service_name: String,
        namespace: String,
        cluster_id: String,
    },
    
    /// Cluster selected by load balancer
    ClusterSelected {
        cluster_id: String,
        algorithm: String,
    },
    
    /// Resource synchronized
    ResourceSynced {
        resource_type: String,
        namespace: String,
        cluster_id: String,
    },
    
    /// Health check failed
    HealthCheckFailed {
        cluster_id: String,
        error: String,
    },
    
    /// Discovery error
    DiscoveryError {
        cluster_id: String,
        error: String,
    },
}

/// Event publisher for federation events
pub struct EventPublisher {
    sender: broadcast::Sender<FederationEvent>,
}

impl EventPublisher {
    /// Create a new event publisher
    pub fn new() -> Self {
        let (sender, _) = broadcast::channel(1000);
        Self { sender }
    }
    
    /// Subscribe to events
    pub fn subscribe(&self) -> broadcast::Receiver<FederationEvent> {
        self.sender.subscribe()
    }
    
    /// Publish an event
    pub fn publish(&self, event: FederationEvent) {
        if let Err(e) = self.sender.send(event.clone()) {
            debug!(error = %e, "No subscribers for event");
        } else {
            info!(event = ?event, "Published federation event");
        }
    }
}

impl Default for EventPublisher {
    fn default() -> Self {
        Self::new()
    }
}

/// Event handler trait
pub trait EventHandler: Send + Sync {
    /// Handle an event
    fn handle(&self, event: &FederationEvent);
}

/// Event manager for federation
pub struct EventManager {
    publisher: Arc<EventPublisher>,
    handlers: Vec<Arc<dyn EventHandler>>,
}

impl EventManager {
    /// Create a new event manager
    pub fn new() -> Self {
        Self {
            publisher: Arc::new(EventPublisher::new()),
            handlers: Vec::new(),
        }
    }
    
    /// Register an event handler
    pub fn register_handler(&mut self, handler: Arc<dyn EventHandler>) {
        self.handlers.push(handler);
    }
    
    /// Get event publisher
    pub fn publisher(&self) -> &Arc<EventPublisher> {
        &self.publisher
    }
    
    /// Start event processing
    pub fn start(&self) {
        let publisher = self.publisher.clone();
        let handlers = self.handlers.clone();
        
        tokio::spawn(async move {
            let mut receiver = publisher.subscribe();
            
            while let Ok(event) = receiver.recv().await {
                for handler in &handlers {
                    handler.handle(&event);
                }
            }
        });
    }
}

impl Default for EventManager {
    fn default() -> Self {
        Self::new()
    }
}

