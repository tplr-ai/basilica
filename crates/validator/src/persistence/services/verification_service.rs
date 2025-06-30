use std::sync::Arc;
use std::collections::HashMap;
use uuid::Uuid;
use chrono::{DateTime, Utc, Duration};
use serde_json::Value;

use common::persistence::{PaginatedResponse, Pagination};
use common::PersistenceError;
use common::identity::{ExecutorId, MinerUid};
use crate::persistence::{
    entities::{VerificationLog, ChallengeResult, EnvironmentValidation},
    repositories::{
        VerificationLogRepository, ChallengeResultRepository, EnvironmentValidationRepository,
    },
};
use crate::journal::VerificationLogger;
use crate::validation::types::AttestationResult;
use crate::metrics::ValidatorBusinessMetrics;

/// Configuration for hardware verification scoring algorithms
#[derive(Clone, Debug)]
pub struct ScoringConfig {
    pub performance_weight: f64,      // Weight for performance scores (0.0-1.0)
    pub reliability_weight: f64,      // Weight for reliability scores (0.0-1.0) 
    pub security_weight: f64,         // Weight for security scores (0.0-1.0)
    pub latency_weight: f64,          // Weight for latency scores (0.0-1.0)
    pub min_success_rate: f64,        // Minimum success rate threshold
    pub penalty_decay: f64,           // Decay factor for failure penalties
    pub time_window_hours: i64,       // Time window for historical analysis
    pub min_verifications: usize,     // Minimum verifications needed for stable score
}

impl Default for ScoringConfig {
    fn default() -> Self {
        Self {
            performance_weight: 0.4,
            reliability_weight: 0.3,
            security_weight: 0.2,
            latency_weight: 0.1,
            min_success_rate: 0.7,
            penalty_decay: 0.9,
            time_window_hours: 24,
            min_verifications: 5,
        }
    }
}

/// Comprehensive scoring metrics for an executor
#[derive(Clone, Debug)]
pub struct ExecutorScore {
    pub executor_id: ExecutorId,
    pub overall_score: f64,
    pub performance_score: f64,
    pub reliability_score: f64,
    pub security_score: f64,
    pub latency_score: f64,
    pub success_rate: f64,
    pub total_verifications: u64,
    pub recent_verifications: u64,
    pub average_duration_ms: f64,
    pub last_verification: Option<DateTime<Utc>>,
    pub score_confidence: f64,        // Confidence level based on verification count
}

/// Hardware performance benchmarking results
#[derive(Clone, Debug)]
pub struct HardwarePerformance {
    pub cpu_score: f64,              // Normalized CPU performance score
    pub memory_score: f64,           // Memory bandwidth and latency score
    pub gpu_score: f64,              // GPU compute performance score
    pub disk_io_score: f64,          // Disk I/O performance score
    pub network_score: f64,          // Network throughput and latency score
    pub overall_performance: f64,    // Weighted average of all components
}

/// Security assessment results
#[derive(Clone, Debug)]
pub struct SecurityAssessment {
    pub attestation_validity: f64,   // GPU attestation verification score
    pub os_integrity: f64,           // Operating system integrity score
    pub docker_security: f64,        // Docker environment security score
    pub network_security: f64,       // Network configuration security score
    pub overall_security: f64,       // Weighted security score
}

/// High-level service for managing verification operations
pub struct VerificationService<V, C, E>
where
    V: VerificationLogRepository + Send + Sync,
    C: ChallengeResultRepository + Send + Sync,
    E: EnvironmentValidationRepository + Send + Sync,
{
    verification_repo: Arc<V>,
    challenge_repo: Arc<C>,
    environment_repo: Arc<E>,
    logger: Arc<VerificationLogger>,
    scoring_config: ScoringConfig,
    metrics: Option<Arc<ValidatorBusinessMetrics>>,
}

impl<V, C, E> VerificationService<V, C, E>
where
    V: VerificationLogRepository + Send + Sync,
    C: ChallengeResultRepository + Send + Sync,
    E: EnvironmentValidationRepository + Send + Sync,
{
    pub fn new(
        verification_repo: Arc<V>,
        challenge_repo: Arc<C>,
        environment_repo: Arc<E>,
        logger: Arc<VerificationLogger>,
    ) -> Self {
        Self {
            verification_repo,
            challenge_repo,
            environment_repo,
            logger,
            scoring_config: ScoringConfig::default(),
            metrics: None,
        }
    }

    pub fn new_with_config(
        verification_repo: Arc<V>,
        challenge_repo: Arc<C>,
        environment_repo: Arc<E>,
        logger: Arc<VerificationLogger>,
        scoring_config: ScoringConfig,
    ) -> Self {
        Self {
            verification_repo,
            challenge_repo,
            environment_repo,
            logger,
            scoring_config,
            metrics: None,
        }
    }

    pub fn new_with_metrics(
        verification_repo: Arc<V>,
        challenge_repo: Arc<C>,
        environment_repo: Arc<E>,
        logger: Arc<VerificationLogger>,
        scoring_config: ScoringConfig,
        metrics: Arc<ValidatorBusinessMetrics>,
    ) -> Self {
        Self {
            verification_repo,
            challenge_repo,
            environment_repo,
            logger,
            scoring_config,
            metrics: Some(metrics),
        }
    }

    /// Create a new verification log entry
    pub async fn create_verification_log(
        &self,
        log: &VerificationLog,
    ) -> Result<(), PersistenceError> {
        self.logger
            .log_verification_started(
                log.id,
                &log.executor_id,
                &log.verification_type,
            )
            .await;

        let result = self.verification_repo.create(log).await;

        if result.is_ok() {
            self.logger
                .log_verification_completed(
                    log.id,
                    &log.executor_id,
                    log.success,
                    log.score,
                    log.duration_ms,
                )
                .await;

            // Record metrics if available
            if let Some(ref metrics) = self.metrics {
                let duration = std::time::Duration::from_millis(log.duration_ms.unwrap_or(0) as u64);
                let score = log.score.unwrap_or(0.0);
                metrics.record_validation_workflow(
                    &log.executor_id,
                    log.success,
                    score,
                    duration,
                    &log.verification_type,
                ).await;
            }
        }

        result
    }

    /// Get verification logs for a specific executor
    pub async fn get_executor_verification_logs(
        &self,
        executor_id: &str,
        pagination: Pagination,
    ) -> Result<PaginatedResponse<VerificationLog>, PersistenceError> {
        self.verification_repo
            .get_by_executor(executor_id, pagination)
            .await
    }

    /// Create a new challenge result
    pub async fn create_challenge_result(
        &self,
        result: &ChallengeResult,
    ) -> Result<(), PersistenceError> {
        self.logger
            .log_challenge_issued(
                result.id,
                &result.executor_id,
                &result.challenge_type,
                result.difficulty_level,
            )
            .await;

        let repo_result = self.challenge_repo.create(result).await;

        if repo_result.is_ok() {
            self.logger
                .log_challenge_completed(
                    result.id,
                    &result.executor_id,
                    result.success,
                    result.score,
                    result.execution_time_ms,
                )
                .await;
        }

        repo_result
    }

    /// Get challenge results for a specific executor
    pub async fn get_executor_challenge_results(
        &self,
        executor_id: &str,
        pagination: Pagination,
    ) -> Result<PaginatedResponse<ChallengeResult>, PersistenceError> {
        self.challenge_repo
            .get_by_executor(executor_id, pagination)
            .await
    }

    /// Create a new environment validation
    pub async fn create_environment_validation(
        &self,
        validation: &EnvironmentValidation,
    ) -> Result<(), PersistenceError> {
        let repo_result = self.environment_repo.create(validation).await;

        if repo_result.is_ok() {
            let issues_count = if let serde_json::Value::Array(ref issues) = validation.issues {
                issues.len()
            } else {
                0
            };

            let warnings_count = if let serde_json::Value::Array(ref warnings) = validation.warnings {
                warnings.len()
            } else {
                0
            };

            self.logger
                .log_environment_validated(
                    validation.id,
                    &validation.executor_id,
                    validation.overall_score,
                    issues_count,
                    warnings_count,
                )
                .await;
        }

        repo_result
    }

    /// Get environment validations for a specific executor
    pub async fn get_executor_environment_validations(
        &self,
        executor_id: &str,
        pagination: Pagination,
    ) -> Result<PaginatedResponse<EnvironmentValidation>, PersistenceError> {
        self.environment_repo
            .get_by_executor(executor_id, pagination)
            .await
    }

    /// Get the latest environment validation for an executor
    pub async fn get_latest_environment_validation(
        &self,
        executor_id: &str,
    ) -> Result<Option<EnvironmentValidation>, PersistenceError> {
        self.environment_repo
            .get_latest_by_executor(executor_id)
            .await
    }

    /// Store attestation result as a verification log entry
    ///
    /// Converts an AttestationResult into a VerificationLog and stores it
    /// with appropriate logging and error handling.
    pub async fn store_attestation_result(
        &self,
        attestation_result: &AttestationResult,
    ) -> Result<(), PersistenceError> {
        // Convert AttestationResult to VerificationLog
        let verification_log = VerificationLog {
            id: Uuid::new_v4(),
            executor_id: attestation_result.executor_id.to_string(),
            verification_type: "attestation".to_string(),
            timestamp: attestation_result.validated_at
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs() as i64,
            success: attestation_result.is_valid,
            score: 0.0, // Will be calculated separately by scoring algorithms
            duration_ms: attestation_result.validation_duration.as_millis() as i64,
            details: serde_json::to_value(&attestation_result.hardware_specs)
                .unwrap_or(serde_json::Value::Null),
            error_message: attestation_result.error_message.clone(),
            signature: attestation_result.signature.clone(),
        };

        // Log the attestation storage attempt
        self.logger
            .log_verification_started(
                verification_log.id,
                &verification_log.executor_id,
                &verification_log.verification_type,
            )
            .await;

        // Store in repository
        let result = self.verification_repo.create(&verification_log).await;

        // Log the completion
        if result.is_ok() {
            self.logger
                .log_verification_completed(
                    verification_log.id,
                    &verification_log.executor_id,
                    verification_log.success,
                    verification_log.score,
                    verification_log.duration_ms,
                )
                .await;
        }

        result
    }

    /// Get successful challenge results for an executor
    pub async fn get_successful_challenges(
        &self,
        executor_id: &str,
        pagination: Pagination,
    ) -> Result<PaginatedResponse<ChallengeResult>, PersistenceError> {
        self.challenge_repo
            .get_successful_challenges(executor_id, pagination)
            .await
    }

    /// Get challenge results by type
    pub async fn get_challenges_by_type(
        &self,
        challenge_type: &str,
        pagination: Pagination,
    ) -> Result<PaginatedResponse<ChallengeResult>, PersistenceError> {
        self.challenge_repo
            .get_by_challenge_type(challenge_type, pagination)
            .await
    }

    /// Get passing environment validations
    pub async fn get_passing_environment_validations(
        &self,
        pagination: Pagination,
    ) -> Result<PaginatedResponse<EnvironmentValidation>, PersistenceError> {
        self.environment_repo
            .get_passing_validations(pagination)
            .await
    }

    /// Update an existing verification log
    pub async fn update_verification_log(
        &self,
        log: &VerificationLog,
    ) -> Result<(), PersistenceError> {
        self.verification_repo.update(log).await
    }

    /// Update an existing challenge result
    pub async fn update_challenge_result(
        &self,
        result: &ChallengeResult,
    ) -> Result<(), PersistenceError> {
        self.challenge_repo.update(result).await
    }

    /// Update an existing environment validation
    pub async fn update_environment_validation(
        &self,
        validation: &EnvironmentValidation,
    ) -> Result<(), PersistenceError> {
        self.environment_repo.update(validation).await
    }

    /// Delete a verification log
    pub async fn delete_verification_log(&self, id: &Uuid) -> Result<bool, PersistenceError> {
        self.verification_repo.delete(id).await
    }

    /// Delete a challenge result
    pub async fn delete_challenge_result(&self, id: &Uuid) -> Result<bool, PersistenceError> {
        self.challenge_repo.delete(id).await
    }

    /// Delete an environment validation
    pub async fn delete_environment_validation(&self, id: &Uuid) -> Result<bool, PersistenceError> {
        self.environment_repo.delete(id).await
    }

    /// Get verification log by ID
    pub async fn get_verification_log(
        &self,
        id: &Uuid,
    ) -> Result<Option<VerificationLog>, PersistenceError> {
        self.verification_repo.find_by_id(id).await
    }

    /// Get challenge result by ID
    pub async fn get_challenge_result(
        &self,
        id: &Uuid,
    ) -> Result<Option<ChallengeResult>, PersistenceError> {
        self.challenge_repo.find_by_id(id).await
    }

    /// Get environment validation by ID
    pub async fn get_environment_validation(
        &self,
        id: &Uuid,
    ) -> Result<Option<EnvironmentValidation>, PersistenceError> {
        self.environment_repo.find_by_id(id).await
    }

    /// Log executor connection failure
    pub async fn log_executor_connection_failed(&self, executor_id: &str, error: &str) {
        self.logger
            .log_executor_connection_failed(executor_id, error)
            .await;
    }

    /// Log security violation
    pub async fn log_security_violation(
        &self,
        executor_id: &str,
        violation_type: &str,
        severity: crate::journal::events::SecuritySeverity,
        details: &str,
    ) {
        self.logger
            .log_security_violation(executor_id, violation_type, severity, details)
            .await;
    }

    // ===== HARDWARE VERIFICATION SCORING ALGORITHMS =====

    /// Calculate comprehensive executor score using multiple metrics
    pub async fn calculate_executor_score(
        &self,
        executor_id: &ExecutorId,
    ) -> Result<ExecutorScore, PersistenceError> {
        let time_window = Utc::now() - Duration::hours(self.scoring_config.time_window_hours);
        
        // Get all verification data within time window
        let recent_logs = self.get_executor_verifications_since(executor_id, time_window).await?;
        let recent_challenges = self.get_executor_challenges_since(executor_id, time_window).await?;
        let environment_validation = self.get_latest_environment_validation(&executor_id.to_string()).await?;

        // Calculate component scores
        let performance_score = self.calculate_performance_score(&recent_challenges).await?;
        let reliability_score = self.calculate_reliability_score(&recent_logs).await?;
        let security_score = self.calculate_security_score(&environment_validation).await?;
        let latency_score = self.calculate_latency_score(&recent_logs, &recent_challenges).await?;

        // Calculate overall weighted score
        let overall_score = self.scoring_config.performance_weight * performance_score
            + self.scoring_config.reliability_weight * reliability_score
            + self.scoring_config.security_weight * security_score
            + self.scoring_config.latency_weight * latency_score;

        // Calculate additional metrics
        let success_rate = self.calculate_success_rate(&recent_logs, &recent_challenges);
        let average_duration_ms = self.calculate_average_duration(&recent_logs, &recent_challenges);
        let total_verifications = (recent_logs.len() + recent_challenges.len()) as u64;
        let score_confidence = self.calculate_score_confidence(total_verifications);
        let last_verification = self.get_last_verification_time(&recent_logs, &recent_challenges);

        Ok(ExecutorScore {
            executor_id: executor_id.clone(),
            overall_score,
            performance_score,
            reliability_score,
            security_score,
            latency_score,
            success_rate,
            total_verifications,
            recent_verifications: total_verifications,
            average_duration_ms,
            last_verification,
            score_confidence,
        })
    }

    /// Calculate performance score based on challenge results
    async fn calculate_performance_score(
        &self,
        challenges: &[ChallengeResult],
    ) -> Result<f64, PersistenceError> {
        if challenges.is_empty() {
            return Ok(0.5); // Neutral score for no data
        }

        let mut performance_scores = Vec::new();

        for challenge in challenges {
            if challenge.success {
                // Normalize score based on challenge difficulty and execution time
                let difficulty_factor = challenge.difficulty_level as f64 / 10.0; // Assume max difficulty 10
                let time_penalty = self.calculate_time_penalty(challenge.execution_time_ms);
                let normalized_score = challenge.score * difficulty_factor * time_penalty;
                performance_scores.push(normalized_score.min(1.0).max(0.0));
            } else {
                // Apply penalty for failed challenges
                performance_scores.push(0.0);
            }
        }

        Ok(performance_scores.iter().sum::<f64>() / performance_scores.len() as f64)
    }

    /// Calculate reliability score based on success rate and consistency
    async fn calculate_reliability_score(
        &self,
        logs: &[VerificationLog],
    ) -> Result<f64, PersistenceError> {
        if logs.is_empty() {
            return Ok(0.5); // Neutral score for no data
        }

        let successful_count = logs.iter().filter(|log| log.success).count();
        let success_rate = successful_count as f64 / logs.len() as f64;

        // Penalize if below minimum success rate threshold
        let reliability_base = if success_rate >= self.scoring_config.min_success_rate {
            success_rate
        } else {
            success_rate * self.scoring_config.penalty_decay
        };

        // Calculate consistency score (lower variance in scores = higher reliability)
        let scores: Vec<f64> = logs.iter().filter(|log| log.success).map(|log| log.score).collect();
        let consistency_score = if scores.len() > 1 {
            let mean = scores.iter().sum::<f64>() / scores.len() as f64;
            let variance = scores.iter().map(|score| (score - mean).powi(2)).sum::<f64>() / scores.len() as f64;
            let std_dev = variance.sqrt();
            (1.0 - std_dev).max(0.0) // Lower std dev = higher consistency
        } else {
            0.8 // Default for insufficient data
        };

        Ok((reliability_base * 0.7 + consistency_score * 0.3).min(1.0))
    }

    /// Calculate security score based on environment validation
    async fn calculate_security_score(
        &self,
        environment: &Option<EnvironmentValidation>,
    ) -> Result<f64, PersistenceError> {
        let Some(env) = environment else {
            return Ok(0.3); // Low score for no security validation
        };

        let mut security_score = env.overall_score;

        // Parse issues and warnings to assess security impact
        let issues_count = if let Value::Array(ref issues) = env.issues {
            issues.len()
        } else {
            0
        };

        let warnings_count = if let Value::Array(ref warnings) = env.warnings {
            warnings.len()
        } else {
            0
        };

        // Apply penalties for security issues
        let issue_penalty = (issues_count as f64 * 0.1).min(0.5);
        let warning_penalty = (warnings_count as f64 * 0.05).min(0.3);

        security_score = (security_score - issue_penalty - warning_penalty).max(0.0);

        Ok(security_score)
    }

    /// Calculate latency score based on response times
    async fn calculate_latency_score(
        &self,
        logs: &[VerificationLog],
        challenges: &[ChallengeResult],
    ) -> Result<f64, PersistenceError> {
        let mut durations = Vec::new();

        for log in logs {
            if let Some(duration) = log.duration_ms {
                durations.push(duration as f64);
            }
        }

        for challenge in challenges {
            if let Some(duration) = challenge.execution_time_ms {
                durations.push(duration as f64);
            }
        }

        if durations.is_empty() {
            return Ok(0.5); // Neutral score for no data
        }

        let average_duration = durations.iter().sum::<f64>() / durations.len() as f64;

        // Define latency thresholds (in milliseconds)
        let excellent_threshold = 1000.0; // 1 second
        let good_threshold = 5000.0;      // 5 seconds
        let acceptable_threshold = 15000.0; // 15 seconds

        let latency_score = if average_duration <= excellent_threshold {
            1.0
        } else if average_duration <= good_threshold {
            1.0 - (average_duration - excellent_threshold) / (good_threshold - excellent_threshold) * 0.3
        } else if average_duration <= acceptable_threshold {
            0.7 - (average_duration - good_threshold) / (acceptable_threshold - good_threshold) * 0.4
        } else {
            0.3 * (20000.0 / average_duration).min(1.0) // Steep penalty for very slow responses
        };

        Ok(latency_score.max(0.0))
    }

    /// Calculate time penalty factor for challenge execution
    fn calculate_time_penalty(&self, execution_time_ms: Option<i64>) -> f64 {
        let Some(time) = execution_time_ms else {
            return 0.8; // Moderate penalty for missing timing data
        };

        let time_f64 = time as f64;
        
        // Define time thresholds for different penalties
        if time_f64 <= 2000.0 {
            1.0 // No penalty for fast execution
        } else if time_f64 <= 10000.0 {
            1.0 - (time_f64 - 2000.0) / 8000.0 * 0.2 // Linear penalty up to 20%
        } else {
            0.8 * (15000.0 / time_f64).min(1.0) // Steeper penalty for very slow execution
        }
    }

    /// Calculate success rate across all verification types
    fn calculate_success_rate(
        &self,
        logs: &[VerificationLog],
        challenges: &[ChallengeResult],
    ) -> f64 {
        let total_items = logs.len() + challenges.len();
        if total_items == 0 {
            return 0.0;
        }

        let successful_logs = logs.iter().filter(|log| log.success).count();
        let successful_challenges = challenges.iter().filter(|challenge| challenge.success).count();
        let total_successful = successful_logs + successful_challenges;

        total_successful as f64 / total_items as f64
    }

    /// Calculate average duration across all verification types
    fn calculate_average_duration(
        &self,
        logs: &[VerificationLog],
        challenges: &[ChallengeResult],
    ) -> f64 {
        let mut durations = Vec::new();

        for log in logs {
            if let Some(duration) = log.duration_ms {
                durations.push(duration as f64);
            }
        }

        for challenge in challenges {
            if let Some(duration) = challenge.execution_time_ms {
                durations.push(duration as f64);
            }
        }

        if durations.is_empty() {
            0.0
        } else {
            durations.iter().sum::<f64>() / durations.len() as f64
        }
    }

    /// Calculate score confidence based on number of verifications
    fn calculate_score_confidence(&self, verification_count: u64) -> f64 {
        if verification_count >= self.scoring_config.min_verifications as u64 {
            (verification_count as f64 / (self.scoring_config.min_verifications as f64 * 2.0)).min(1.0)
        } else {
            verification_count as f64 / self.scoring_config.min_verifications as f64
        }
    }

    /// Get the most recent verification timestamp
    fn get_last_verification_time(
        &self,
        logs: &[VerificationLog],
        challenges: &[ChallengeResult],
    ) -> Option<DateTime<Utc>> {
        let mut latest: Option<DateTime<Utc>> = None;

        for log in logs {
            if latest.is_none() || log.created_at > latest.unwrap() {
                latest = Some(log.created_at);
            }
        }

        for challenge in challenges {
            if latest.is_none() || challenge.created_at > latest.unwrap() {
                latest = Some(challenge.created_at);
            }
        }

        latest
    }

    /// Get verification logs since a specific timestamp
    async fn get_executor_verifications_since(
        &self,
        executor_id: &ExecutorId,
        since: DateTime<Utc>,
    ) -> Result<Vec<VerificationLog>, PersistenceError> {
        // Note: This assumes the repository supports time-based queries
        // In practice, you'd need to add this method to the repository trait
        let pagination = Pagination::new(0, 1000); // Get up to 1000 recent items
        let response = self.verification_repo
            .get_by_executor(&executor_id.to_string(), pagination)
            .await?;
        
        Ok(response.items.into_iter()
            .filter(|log| log.created_at >= since)
            .collect())
    }

    /// Get challenge results since a specific timestamp
    async fn get_executor_challenges_since(
        &self,
        executor_id: &ExecutorId,
        since: DateTime<Utc>,
    ) -> Result<Vec<ChallengeResult>, PersistenceError> {
        let pagination = Pagination::new(0, 1000);
        let response = self.challenge_repo
            .get_by_executor(&executor_id.to_string(), pagination)
            .await?;
        
        Ok(response.items.into_iter()
            .filter(|challenge| challenge.created_at >= since)
            .collect())
    }

    /// Calculate scores for multiple executors in batch
    pub async fn calculate_batch_executor_scores(
        &self,
        executor_ids: &[ExecutorId],
    ) -> Result<HashMap<ExecutorId, ExecutorScore>, PersistenceError> {
        let mut scores = HashMap::new();

        for executor_id in executor_ids {
            match self.calculate_executor_score(executor_id).await {
                Ok(score) => {
                    scores.insert(executor_id.clone(), score);
                }
                Err(e) => {
                    tracing::warn!("Failed to calculate score for executor {}: {}", executor_id, e);
                    // Continue with other executors
                }
            }
        }

        Ok(scores)
    }

    /// Get top performing executors based on overall score
    pub async fn get_top_performers(
        &self,
        executor_ids: &[ExecutorId],
        limit: usize,
    ) -> Result<Vec<ExecutorScore>, PersistenceError> {
        let scores = self.calculate_batch_executor_scores(executor_ids).await?;
        
        let mut ranked_scores: Vec<ExecutorScore> = scores.into_values().collect();
        ranked_scores.sort_by(|a, b| b.overall_score.partial_cmp(&a.overall_score).unwrap());
        ranked_scores.truncate(limit);

        Ok(ranked_scores)
    }

    /// Calculate hardware performance score from environment validation data
    pub fn calculate_hardware_performance(
        &self,
        environment: &EnvironmentValidation,
    ) -> Result<HardwarePerformance, PersistenceError> {
        // Parse hardware info from the environment validation JSON
        let hardware_info = &environment.hardware_info;
        
        let cpu_score = self.extract_cpu_score(hardware_info);
        let memory_score = self.extract_memory_score(hardware_info);
        let gpu_score = self.extract_gpu_score(hardware_info);
        let disk_io_score = self.extract_disk_score(hardware_info);
        let network_score = self.extract_network_score(hardware_info);

        // Calculate weighted overall performance
        let overall_performance = (cpu_score * 0.3 + memory_score * 0.2 + gpu_score * 0.3 
                                  + disk_io_score * 0.1 + network_score * 0.1).min(1.0);

        Ok(HardwarePerformance {
            cpu_score,
            memory_score,
            gpu_score,
            disk_io_score,
            network_score,
            overall_performance,
        })
    }

    /// Calculate security assessment from environment validation
    pub fn calculate_security_assessment(
        &self,
        environment: &EnvironmentValidation,
    ) -> Result<SecurityAssessment, PersistenceError> {
        let attestation_validity = self.extract_attestation_score(&environment.hardware_info);
        let os_integrity = self.extract_os_integrity_score(&environment.issues, &environment.warnings);
        let docker_security = self.extract_docker_security_score(&environment.hardware_info);
        let network_security = self.extract_network_security_score(&environment.hardware_info);

        let overall_security = (attestation_validity * 0.4 + os_integrity * 0.3 
                               + docker_security * 0.2 + network_security * 0.1).min(1.0);

        Ok(SecurityAssessment {
            attestation_validity,
            os_integrity,
            docker_security,
            network_security,
            overall_security,
        })
    }

    // Helper methods for extracting scores from JSON data
    fn extract_cpu_score(&self, hardware_info: &Value) -> f64 {
        let system_info = match hardware_info.get("system_info") {
            Some(info) => info,
            None => return 0.3,
        };

        let cpu_info = match system_info.get("cpu") {
            Some(cpu) => cpu,
            None => return 0.3,
        };

        let mut score = 0.0;
        let mut weight_sum = 0.0;

        // Core count scoring (0.3 weight)
        if let Some(cores) = cpu_info.get("cores").and_then(|v| v.as_u64()) {
            let core_score = ((cores as f64).ln() / 4.0_f64.ln()).min(1.0); // Log scale, max at 16+ cores
            score += core_score * 0.3;
            weight_sum += 0.3;
        }

        // Frequency scoring (0.25 weight)
        if let Some(freq) = cpu_info.get("frequency_mhz").and_then(|v| v.as_f64()) {
            let freq_score = ((freq - 1000.0) / 4000.0).clamp(0.0, 1.0); // 1-5GHz range
            score += freq_score * 0.25;
            weight_sum += 0.25;
        }

        // CPU features scoring (0.2 weight)
        if let Some(features) = cpu_info.get("features").and_then(|v| v.as_array()) {
            let important_features = ["avx", "avx2", "fma", "sse4_1", "sse4_2"];
            let feature_count = important_features.iter()
                .filter(|&feat| features.iter().any(|f| f.as_str() == Some(feat)))
                .count();
            let feature_score = (feature_count as f64) / (important_features.len() as f64);
            score += feature_score * 0.2;
            weight_sum += 0.2;
        }

        // Benchmark scoring (0.15 weight)
        if let Some(benchmarks) = system_info.get("benchmarks") {
            if let Some(bench_score) = benchmarks.get("cpu_benchmark_score").and_then(|v| v.as_f64()) {
                let normalized_bench = (bench_score.ln() / 25.0_f64.ln()).clamp(0.0, 1.0); // Log normalize
                score += normalized_bench * 0.15;
                weight_sum += 0.15;
            }
        }

        // Temperature penalty (0.1 weight)
        let temp_score = if let Some(temp) = cpu_info.get("temperature").and_then(|v| v.as_f64()) {
            if temp > 90.0 {
                0.0 // Critical temperature
            } else if temp > 80.0 {
                1.0 - ((temp - 80.0) / 10.0) // Linear penalty above 80°C
            } else {
                1.0 // Good temperature
            }
        } else {
            0.8 // Unknown temperature, moderate score
        };
        score += temp_score * 0.1;
        weight_sum += 0.1;

        if weight_sum > 0.0 {
            score / weight_sum
        } else {
            0.3 // Fallback for no data
        }
    }

    fn extract_memory_score(&self, hardware_info: &Value) -> f64 {
        let system_info = match hardware_info.get("system_info") {
            Some(info) => info,
            None => return 0.3,
        };

        let memory_info = match system_info.get("memory") {
            Some(mem) => mem,
            None => return 0.3,
        };

        let mut score = 0.0;
        let mut weight_sum = 0.0;

        // Total memory capacity scoring (0.4 weight)
        if let Some(total_bytes) = memory_info.get("total_bytes").and_then(|v| v.as_u64()) {
            let gb = total_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
            let capacity_score = if gb >= 128.0 {
                1.0
            } else if gb >= 64.0 {
                0.9
            } else if gb >= 32.0 {
                0.8
            } else if gb >= 16.0 {
                0.6
            } else if gb >= 8.0 {
                0.4
            } else {
                0.2
            };
            score += capacity_score * 0.4;
            weight_sum += 0.4;
        }

        // Memory utilization efficiency (0.25 weight)
        if let (Some(total), Some(available)) = (
            memory_info.get("total_bytes").and_then(|v| v.as_u64()),
            memory_info.get("available_bytes").and_then(|v| v.as_u64())
        ) {
            let utilization = 1.0 - (available as f64 / total as f64);
            let efficiency_score = if utilization < 0.7 {
                1.0
            } else if utilization < 0.85 {
                1.0 - ((utilization - 0.7) / 0.15) * 0.3
            } else {
                0.4 // High utilization penalty
            };
            score += efficiency_score * 0.25;
            weight_sum += 0.25;
        }

        // Memory module quality (0.2 weight)
        if let Some(modules) = memory_info.get("memory_modules").and_then(|v| v.as_array()) {
            let mut module_score = 0.0;
            let mut module_count = 0;
            
            for module in modules {
                if let Some(speed) = module.get("speed_mhz").and_then(|v| v.as_f64()) {
                    let speed_score = if speed >= 3200.0 {
                        1.0
                    } else if speed >= 2400.0 {
                        0.8
                    } else if speed >= 1600.0 {
                        0.6
                    } else {
                        0.4
                    };
                    module_score += speed_score;
                    module_count += 1;
                }
            }
            
            if module_count > 0 {
                score += (module_score / module_count as f64) * 0.2;
                weight_sum += 0.2;
            }
        }

        // Memory bandwidth benchmark (0.15 weight)
        if let Some(benchmarks) = system_info.get("benchmarks") {
            if let Some(bandwidth) = benchmarks.get("memory_bandwidth_mbps").and_then(|v| v.as_f64()) {
                let bandwidth_score = (bandwidth / 50000.0).clamp(0.0, 1.0); // Normalize to ~50GB/s max
                score += bandwidth_score * 0.15;
                weight_sum += 0.15;
            }
        }

        if weight_sum > 0.0 {
            score / weight_sum
        } else {
            0.3 // Fallback for no data
        }
    }

    fn extract_gpu_score(&self, hardware_info: &Value) -> f64 {
        let gpu_info = match hardware_info.get("gpu_info").and_then(|v| v.as_array()) {
            Some(gpus) if !gpus.is_empty() => gpus,
            _ => return 0.1, // No GPU is very low score for mining
        };

        let mut total_score = 0.0;
        let gpu_count = gpu_info.len();

        for gpu in gpu_info {
            let mut gpu_score = 0.0;
            let mut weight_sum = 0.0;

            // GPU tier scoring based on name (0.4 weight)
            if let Some(name) = gpu.get("name").and_then(|v| v.as_str()) {
                let tier_score = self.get_gpu_tier_score(name);
                gpu_score += tier_score * 0.4;
                weight_sum += 0.4;
            }

            // VRAM capacity and utilization (0.3 weight)
            if let (Some(total_vram), Some(used_vram)) = (
                gpu.get("memory_total").and_then(|v| v.as_u64()),
                gpu.get("memory_used").and_then(|v| v.as_u64())
            ) {
                let vram_gb = total_vram as f64 / (1024.0 * 1024.0 * 1024.0);
                let utilization = used_vram as f64 / total_vram as f64;
                
                let vram_score = if vram_gb >= 24.0 {
                    1.0
                } else if vram_gb >= 16.0 {
                    0.9
                } else if vram_gb >= 12.0 {
                    0.8
                } else if vram_gb >= 8.0 {
                    0.7
                } else if vram_gb >= 6.0 {
                    0.5
                } else {
                    0.3
                };
                
                let util_penalty = if utilization > 0.9 { 0.7 } else { 1.0 };
                gpu_score += vram_score * util_penalty * 0.3;
                weight_sum += 0.3;
            }

            // Temperature and power efficiency (0.2 weight)
            let thermal_score = if let Some(temp) = gpu.get("temperature").and_then(|v| v.as_f64()) {
                if temp > 85.0 {
                    0.3 // Too hot
                } else if temp > 75.0 {
                    0.7 // Warm but acceptable
                } else {
                    1.0 // Good temperature
                }
            } else {
                0.8 // Unknown temperature
            };
            gpu_score += thermal_score * 0.2;
            weight_sum += 0.2;

            // Compute capability (0.1 weight)
            if let Some(compute_cap) = gpu.get("compute_capability").and_then(|v| v.as_str()) {
                let cap_score = match compute_cap {
                    cap if cap >= "8.0" => 1.0,
                    cap if cap >= "7.5" => 0.9,
                    cap if cap >= "6.1" => 0.8,
                    cap if cap >= "5.0" => 0.6,
                    _ => 0.4,
                };
                gpu_score += cap_score * 0.1;
                weight_sum += 0.1;
            }

            if weight_sum > 0.0 {
                total_score += gpu_score / weight_sum;
            }
        }

        let avg_score = total_score / gpu_count as f64;
        
        // Multi-GPU bonus
        if gpu_count > 1 {
            (avg_score * 1.1).min(1.0)
        } else {
            avg_score
        }
    }

    fn extract_disk_score(&self, hardware_info: &Value) -> f64 {
        let system_info = match hardware_info.get("system_info") {
            Some(info) => info,
            None => return 0.3,
        };

        let mut score = 0.0;
        let mut weight_sum = 0.0;

        // Storage device assessment (0.4 weight)
        if let Some(storage) = system_info.get("storage").and_then(|v| v.as_array()) {
            let mut storage_score = 0.0;
            let mut total_capacity = 0u64;
            let mut total_available = 0u64;
            let mut has_ssd = false;
            let mut has_nvme = false;

            for device in storage {
                if let Some(disk_type) = device.get("disk_type").and_then(|v| v.as_str()) {
                    match disk_type {
                        "NVMe" => { has_nvme = true; storage_score += 1.0; }
                        "SSD" => { has_ssd = true; storage_score += 0.8; }
                        "HDD" => { storage_score += 0.4; }
                        _ => { storage_score += 0.5; }
                    }
                }

                if let Some(total) = device.get("total_space").and_then(|v| v.as_u64()) {
                    total_capacity += total;
                }
                if let Some(available) = device.get("available_space").and_then(|v| v.as_u64()) {
                    total_available += available;
                }
            }

            if !storage.is_empty() {
                let type_score = storage_score / storage.len() as f64;
                let capacity_gb = total_capacity as f64 / (1024.0 * 1024.0 * 1024.0);
                let available_ratio = if total_capacity > 0 {
                    total_available as f64 / total_capacity as f64
                } else {
                    0.0
                };

                let capacity_score = if capacity_gb >= 2000.0 {
                    1.0
                } else if capacity_gb >= 1000.0 {
                    0.9
                } else if capacity_gb >= 500.0 {
                    0.8
                } else if capacity_gb >= 250.0 {
                    0.6
                } else {
                    0.4
                };

                let space_penalty = if available_ratio < 0.1 {
                    0.5 // Very low space
                } else if available_ratio < 0.2 {
                    0.8 // Low space
                } else {
                    1.0 // Adequate space
                };

                score += (type_score * 0.6 + capacity_score * 0.4) * space_penalty * 0.4;
                weight_sum += 0.4;
            }
        }

        // Disk benchmark performance (0.4 weight)
        if let Some(benchmarks) = system_info.get("benchmarks") {
            let mut bench_score = 0.0;
            let mut bench_count = 0;

            if let Some(seq_read) = benchmarks.get("disk_sequential_read_mbps").and_then(|v| v.as_f64()) {
                let read_score = (seq_read / 3000.0).clamp(0.0, 1.0); // Normalize to ~3GB/s
                bench_score += read_score;
                bench_count += 1;
            }

            if let Some(seq_write) = benchmarks.get("disk_sequential_write_mbps").and_then(|v| v.as_f64()) {
                let write_score = (seq_write / 2000.0).clamp(0.0, 1.0); // Normalize to ~2GB/s
                bench_score += write_score;
                bench_count += 1;
            }

            if bench_count > 0 {
                score += (bench_score / bench_count as f64) * 0.4;
                weight_sum += 0.4;
            }
        }

        // Docker storage performance (0.2 weight)
        if let Some(docker_attestation) = hardware_info.get("docker_attestation") {
            if let Some(docker_benchmarks) = docker_attestation.get("benchmarks") {
                if let Some(docker_io) = docker_benchmarks.get("disk_io_performance_mbps").and_then(|v| v.as_f64()) {
                    let docker_score = (docker_io / 1000.0).clamp(0.0, 1.0); // Normalize to 1GB/s
                    score += docker_score * 0.2;
                    weight_sum += 0.2;
                }
            }
        }

        if weight_sum > 0.0 {
            score / weight_sum
        } else {
            0.3 // Fallback for no data
        }
    }

    fn extract_network_score(&self, hardware_info: &Value) -> f64 {
        let network_bench = match hardware_info.get("network_benchmark") {
            Some(bench) => bench,
            None => return 0.3,
        };

        let mut score = 0.0;
        let mut weight_sum = 0.0;

        // Latency scoring (0.3 weight)
        if let Some(latency_tests) = network_bench.get("latency_tests").and_then(|v| v.as_array()) {
            let mut latency_score = 0.0;
            let mut test_count = 0;

            for test in latency_tests {
                if let Some(avg_latency) = test.get("avg_latency_ms").and_then(|v| v.as_f64()) {
                    let test_score = if avg_latency <= 20.0 {
                        1.0
                    } else if avg_latency <= 50.0 {
                        1.0 - ((avg_latency - 20.0) / 30.0) * 0.3
                    } else if avg_latency <= 100.0 {
                        0.7 - ((avg_latency - 50.0) / 50.0) * 0.4
                    } else {
                        0.3
                    };
                    latency_score += test_score;
                    test_count += 1;
                }
            }

            if test_count > 0 {
                score += (latency_score / test_count as f64) * 0.3;
                weight_sum += 0.3;
            }
        }

        // Throughput scoring (0.3 weight)
        if let Some(throughput_tests) = network_bench.get("throughput_tests").and_then(|v| v.as_array()) {
            let mut throughput_score = 0.0;
            let mut test_count = 0;

            for test in throughput_tests {
                if let Some(mbps) = test.get("throughput_mbps").and_then(|v| v.as_f64()) {
                    let test_score = if mbps >= 1000.0 {
                        1.0
                    } else if mbps >= 500.0 {
                        0.9
                    } else if mbps >= 100.0 {
                        0.8
                    } else if mbps >= 50.0 {
                        0.6
                    } else if mbps >= 10.0 {
                        0.4
                    } else {
                        0.2
                    };
                    throughput_score += test_score;
                    test_count += 1;
                }
            }

            if test_count > 0 {
                score += (throughput_score / test_count as f64) * 0.3;
                weight_sum += 0.3;
            }
        }

        // Packet loss and reliability (0.2 weight)
        if let Some(packet_loss_test) = network_bench.get("packet_loss_test") {
            if let Some(loss_percent) = packet_loss_test.get("packet_loss_percent").and_then(|v| v.as_f64()) {
                let reliability_score = if loss_percent <= 0.0 {
                    1.0
                } else if loss_percent <= 1.0 {
                    0.9
                } else if loss_percent <= 5.0 {
                    0.7
                } else {
                    0.3
                };
                score += reliability_score * 0.2;
                weight_sum += 0.2;
            }
        }

        // DNS resolution performance (0.1 weight)
        if let Some(dns_test) = network_bench.get("dns_resolution_test") {
            if let Some(resolution_time) = dns_test.get("resolution_time_ms").and_then(|v| v.as_f64()) {
                let dns_score = if resolution_time <= 10.0 {
                    1.0
                } else if resolution_time <= 50.0 {
                    0.8
                } else if resolution_time <= 100.0 {
                    0.6
                } else {
                    0.3
                };
                score += dns_score * 0.1;
                weight_sum += 0.1;
            }
        }

        // Geographic/ISP bonus (0.1 weight)
        if let Some(ipinfo) = hardware_info.get("ipinfo") {
            let geo_score = if let Some(org) = ipinfo.get("org").and_then(|v| v.as_str()) {
                if org.contains("Google") || org.contains("Amazon") || org.contains("Microsoft") {
                    1.0 // Premium cloud providers
                } else if org.contains("Fiber") || org.contains("Telecom") {
                    0.9 // Good ISPs
                } else {
                    0.7 // Standard ISPs
                }
            } else {
                0.7
            };
            score += geo_score * 0.1;
            weight_sum += 0.1;
        }

        if weight_sum > 0.0 {
            score / weight_sum
        } else {
            0.3 // Fallback for no data
        }
    }

    fn extract_attestation_score(&self, hardware_info: &Value) -> f64 {
        // Extract GPU attestation validation results
        0.9 // Placeholder implementation
    }

    fn extract_os_integrity_score(&self, issues: &Value, warnings: &Value) -> f64 {
        // Analyze OS security issues and warnings
        let issue_count = if let Value::Array(ref arr) = issues { arr.len() } else { 0 };
        let warning_count = if let Value::Array(ref arr) = warnings { arr.len() } else { 0 };
        
        let penalty = (issue_count as f64 * 0.2 + warning_count as f64 * 0.1).min(0.8);
        (1.0 - penalty).max(0.0)
    }

    fn extract_docker_security_score(&self, hardware_info: &Value) -> f64 {
        // Extract Docker security configuration assessment
        0.9 // Placeholder implementation
    }

    fn extract_network_security_score(&self, hardware_info: &Value) -> f64 {
        // Extract network security configuration assessment
        0.9 // Placeholder implementation
    }

    /// Get GPU tier score based on GPU name/model
    fn get_gpu_tier_score(&self, gpu_name: &str) -> f64 {
        let name_lower = gpu_name.to_lowercase();
        
        // High-end datacenter GPUs
        if name_lower.contains("h100") || name_lower.contains("a100") || name_lower.contains("v100") {
            return 1.0;
        }
        
        // High-end consumer/prosumer GPUs
        if name_lower.contains("rtx 4090") || name_lower.contains("rtx 4080") || 
           name_lower.contains("rtx 3090") || name_lower.contains("rtx 3080") ||
           name_lower.contains("titan") {
            return 0.95;
        }
        
        // Upper mid-range GPUs
        if name_lower.contains("rtx 4070") || name_lower.contains("rtx 3070") ||
           name_lower.contains("rtx 2080") || name_lower.contains("rx 7800") ||
           name_lower.contains("rx 6800") {
            return 0.85;
        }
        
        // Mid-range GPUs
        if name_lower.contains("rtx 4060") || name_lower.contains("rtx 3060") ||
           name_lower.contains("rtx 2070") || name_lower.contains("rx 7600") ||
           name_lower.contains("rx 6600") {
            return 0.75;
        }
        
        // Lower mid-range GPUs
        if name_lower.contains("rtx 2060") || name_lower.contains("gtx 1660") ||
           name_lower.contains("rx 5600") || name_lower.contains("rx 580") {
            return 0.65;
        }
        
        // Entry-level GPUs
        if name_lower.contains("gtx 1650") || name_lower.contains("gtx 1050") ||
           name_lower.contains("rx 570") || name_lower.contains("rx 560") {
            return 0.5;
        }
        
        // Very old or low-end GPUs
        if name_lower.contains("gtx") || name_lower.contains("rx") ||
           name_lower.contains("radeon") || name_lower.contains("geforce") {
            return 0.3;
        }
        
        // Integrated or unknown GPUs
        if name_lower.contains("intel") || name_lower.contains("integrated") {
            return 0.1;
        }
        
        // Default for unrecognized GPUs
        0.4
    }
}