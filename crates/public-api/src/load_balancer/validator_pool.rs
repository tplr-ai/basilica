//! Validator pool management

use crate::discovery::ValidatorInfo;
use dashmap::DashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Validator pool for load balancing
pub struct ValidatorPool {
    /// Validators in the pool
    validators: DashMap<u16, ValidatorInfo>,

    /// Active connection counts per validator
    connection_counts: DashMap<u16, AtomicUsize>,

    /// Failure counts per validator
    failure_counts: DashMap<u16, AtomicUsize>,
}

impl ValidatorPool {
    /// Create a new validator pool
    pub fn new() -> Self {
        Self {
            validators: DashMap::new(),
            connection_counts: DashMap::new(),
            failure_counts: DashMap::new(),
        }
    }

    /// Update the pool with current validators
    pub fn update(&self, validators: Vec<ValidatorInfo>) {
        // Clear existing validators
        self.validators.clear();

        // Add new validators
        for validator in validators {
            let uid = validator.uid;
            self.validators.insert(uid, validator);

            // Initialize counters if not present
            self.connection_counts
                .entry(uid)
                .or_insert_with(|| AtomicUsize::new(0));
            self.failure_counts
                .entry(uid)
                .or_insert_with(|| AtomicUsize::new(0));
        }

        // Clean up counters for removed validators
        let current_uids: Vec<u16> = self.validators.iter().map(|e| *e.key()).collect();

        self.connection_counts
            .retain(|uid, _| current_uids.contains(uid));
        self.failure_counts
            .retain(|uid, _| current_uids.contains(uid));
    }

    /// Get all healthy validators
    pub fn get_healthy_validators(&self) -> Vec<ValidatorInfo> {
        self.validators
            .iter()
            .filter(|entry| entry.is_healthy)
            .map(|entry| entry.value().clone())
            .collect()
    }

    /// Get total validator count
    pub fn total_count(&self) -> usize {
        self.validators.len()
    }

    /// Get healthy validator count
    pub fn healthy_count(&self) -> usize {
        self.validators
            .iter()
            .filter(|entry| entry.is_healthy)
            .count()
    }

    /// Get total active connections
    pub fn total_connections(&self) -> usize {
        self.connection_counts
            .iter()
            .map(|entry| entry.value().load(Ordering::Relaxed))
            .sum()
    }

    /// Increment connection count for a validator
    pub fn increment_connections(&self, uid: u16) {
        if let Some(counter) = self.connection_counts.get(&uid) {
            counter.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Decrement connection count for a validator
    pub fn decrement_connections(&self, uid: u16) {
        if let Some(counter) = self.connection_counts.get(&uid) {
            counter.fetch_sub(1, Ordering::Relaxed);
        }
    }

    /// Get connection count for a validator
    pub fn get_connection_count(&self, uid: u16) -> usize {
        self.connection_counts
            .get(&uid)
            .map(|counter| counter.load(Ordering::Relaxed))
            .unwrap_or(0)
    }

    /// Increment failure count for a validator
    pub fn increment_failures(&self, uid: u16) {
        if let Some(counter) = self.failure_counts.get(&uid) {
            counter.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Get failure count for a validator
    pub fn get_failure_count(&self, uid: u16) -> usize {
        self.failure_counts
            .get(&uid)
            .map(|counter| counter.load(Ordering::Relaxed))
            .unwrap_or(0)
    }

    /// Reset failure count for a validator
    pub fn reset_failures(&self, uid: u16) {
        if let Some(counter) = self.failure_counts.get(&uid) {
            counter.store(0, Ordering::Relaxed);
        }
    }
}

impl Default for ValidatorPool {
    fn default() -> Self {
        Self::new()
    }
}
