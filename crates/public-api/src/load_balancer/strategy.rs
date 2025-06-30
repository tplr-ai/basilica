//! Load balancing strategies

use super::ValidatorPool;
use crate::{discovery::ValidatorInfo, Error, Result};
use async_trait::async_trait;
use rand::Rng;
use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};
// weighted-rs 0.1 doesn't have SmoothWeightedRoundRobin, we'll implement our own simple version

/// Load balancing strategy trait
#[async_trait]
pub trait Strategy: Send + Sync {
    /// Select a validator for the next request
    async fn select(&self) -> Result<ValidatorInfo>;

    /// Select a validator using a specific key (for consistent hashing)
    async fn select_with_key(&self, key: &str) -> Result<ValidatorInfo>;
}

/// Round-robin strategy
pub struct RoundRobinStrategy {
    pool: Arc<ValidatorPool>,
    counter: AtomicUsize,
}

impl RoundRobinStrategy {
    pub fn new(pool: Arc<ValidatorPool>) -> Self {
        Self {
            pool,
            counter: AtomicUsize::new(0),
        }
    }
}

#[async_trait]
impl Strategy for RoundRobinStrategy {
    async fn select(&self) -> Result<ValidatorInfo> {
        let validators = self.pool.get_healthy_validators();
        if validators.is_empty() {
            return Err(Error::NoValidatorsAvailable);
        }

        let index = self.counter.fetch_add(1, Ordering::Relaxed) % validators.len();
        let validator = validators[index].clone();

        self.pool.increment_connections(validator.uid);
        Ok(validator)
    }

    async fn select_with_key(&self, _key: &str) -> Result<ValidatorInfo> {
        self.select().await
    }
}

/// Least connections strategy
pub struct LeastConnectionsStrategy {
    pool: Arc<ValidatorPool>,
}

impl LeastConnectionsStrategy {
    pub fn new(pool: Arc<ValidatorPool>) -> Self {
        Self { pool }
    }
}

#[async_trait]
impl Strategy for LeastConnectionsStrategy {
    async fn select(&self) -> Result<ValidatorInfo> {
        let validators = self.pool.get_healthy_validators();
        if validators.is_empty() {
            return Err(Error::NoValidatorsAvailable);
        }

        // Find validator with least connections
        let validator = validators
            .into_iter()
            .min_by_key(|v| self.pool.get_connection_count(v.uid))
            .ok_or(Error::NoValidatorsAvailable)?;

        self.pool.increment_connections(validator.uid);
        Ok(validator)
    }

    async fn select_with_key(&self, _key: &str) -> Result<ValidatorInfo> {
        self.select().await
    }
}

/// Weighted score strategy
pub struct WeightedScoreStrategy {
    pool: Arc<ValidatorPool>,
}

impl WeightedScoreStrategy {
    pub fn new(pool: Arc<ValidatorPool>) -> Self {
        Self { pool }
    }
}

#[async_trait]
impl Strategy for WeightedScoreStrategy {
    async fn select(&self) -> Result<ValidatorInfo> {
        let validators = self.pool.get_healthy_validators();
        if validators.is_empty() {
            return Err(Error::NoValidatorsAvailable);
        }

        // Simple weighted selection based on scores
        let total_weight: f64 = validators.iter().map(|v| v.score).sum();
        let mut rng = rand::thread_rng();
        let mut random_weight = rng.gen::<f64>() * total_weight;

        for validator in validators.iter() {
            random_weight -= validator.score;
            if random_weight <= 0.0 {
                self.pool.increment_connections(validator.uid);
                return Ok(validator.clone());
            }
        }

        // Fallback to first validator
        let validator = validators
            .into_iter()
            .next()
            .ok_or(Error::NoValidatorsAvailable)?;
        self.pool.increment_connections(validator.uid);
        Ok(validator)
    }

    async fn select_with_key(&self, _key: &str) -> Result<ValidatorInfo> {
        self.select().await
    }
}

/// Random strategy
pub struct RandomStrategy {
    pool: Arc<ValidatorPool>,
}

impl RandomStrategy {
    pub fn new(pool: Arc<ValidatorPool>) -> Self {
        Self { pool }
    }
}

#[async_trait]
impl Strategy for RandomStrategy {
    async fn select(&self) -> Result<ValidatorInfo> {
        let validators = self.pool.get_healthy_validators();
        if validators.is_empty() {
            return Err(Error::NoValidatorsAvailable);
        }

        let mut rng = rand::thread_rng();
        let index = rng.gen_range(0..validators.len());
        let validator = validators[index].clone();

        self.pool.increment_connections(validator.uid);
        Ok(validator)
    }

    async fn select_with_key(&self, _key: &str) -> Result<ValidatorInfo> {
        self.select().await
    }
}

/// Consistent hash strategy
pub struct ConsistentHashStrategy {
    pool: Arc<ValidatorPool>,
}

impl ConsistentHashStrategy {
    pub fn new(pool: Arc<ValidatorPool>) -> Self {
        Self { pool }
    }

    fn hash_key(&self, key: &str) -> u64 {
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        hasher.finish()
    }
}

#[async_trait]
impl Strategy for ConsistentHashStrategy {
    async fn select(&self) -> Result<ValidatorInfo> {
        // For requests without a key, fall back to random selection
        let validators = self.pool.get_healthy_validators();
        if validators.is_empty() {
            return Err(Error::NoValidatorsAvailable);
        }

        let mut rng = rand::thread_rng();
        let index = rng.gen_range(0..validators.len());
        let validator = validators[index].clone();

        self.pool.increment_connections(validator.uid);
        Ok(validator)
    }

    async fn select_with_key(&self, key: &str) -> Result<ValidatorInfo> {
        let validators = self.pool.get_healthy_validators();
        if validators.is_empty() {
            return Err(Error::NoValidatorsAvailable);
        }

        // Sort validators by UID for consistent ordering
        let mut sorted_validators = validators;
        sorted_validators.sort_by_key(|v| v.uid);

        // Use consistent hashing to select a validator
        let hash = self.hash_key(key);
        let index = (hash as usize) % sorted_validators.len();
        let validator = sorted_validators[index].clone();

        self.pool.increment_connections(validator.uid);
        Ok(validator)
    }
}
