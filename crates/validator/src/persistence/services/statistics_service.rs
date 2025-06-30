use std::sync::Arc;
use chrono::{DateTime, Utc};

use common::persistence::Pagination;
use common::PersistenceError;
use crate::persistence::{
    entities::ExecutorVerificationStats,
    repositories::VerificationLogRepository,
};
use crate::journal::{VerificationLogger, VerificationStats};

/// Service for generating verification statistics and analytics
pub struct StatisticsService<V>
where
    V: VerificationLogRepository + Send + Sync,
{
    verification_repo: Arc<V>,
    logger: Arc<VerificationLogger>,
}

impl<V> StatisticsService<V>
where
    V: VerificationLogRepository + Send + Sync,
{
    pub fn new(verification_repo: Arc<V>, logger: Arc<VerificationLogger>) -> Self {
        Self {
            verification_repo,
            logger,
        }
    }

    /// Get comprehensive executor statistics
    pub async fn get_executor_stats(
        &self,
        executor_id: &str,
        days: Option<i32>,
    ) -> Result<ExecutorVerificationStats, PersistenceError> {
        self.verification_repo
            .get_executor_stats(executor_id, days)
            .await
    }

    /// Get system-wide verification statistics from journal
    pub async fn get_system_verification_stats(
        &self,
        days: u32,
    ) -> Result<VerificationStats, Box<dyn std::error::Error>> {
        self.logger.get_verification_stats(None, days).await
    }

    /// Get executor-specific verification statistics from journal
    pub async fn get_executor_verification_stats(
        &self,
        executor_id: &str,
        days: u32,
    ) -> Result<VerificationStats, Box<dyn std::error::Error>> {
        self.logger
            .get_verification_stats(Some(executor_id), days)
            .await
    }

    /// Get top performing executors by success rate
    pub async fn get_top_executors_by_success_rate(
        &self,
        limit: u32,
        days: Option<i32>,
    ) -> Result<Vec<ExecutorPerformance>, PersistenceError> {
        let logs = self
            .verification_repo
            .find_all(Pagination::new(0, limit as usize))
            .await?;

        let mut executor_stats = std::collections::HashMap::new();

        for log in logs.data {
            let stats = executor_stats
                .entry(log.executor_id.clone())
                .or_insert(ExecutorPerformance {
                    executor_id: log.executor_id.clone(),
                    total_verifications: 0,
                    successful_verifications: 0,
                    average_score: 0.0,
                    last_verification: None,
                });

            stats.total_verifications += 1;
            if log.success {
                stats.successful_verifications += 1;
            }

            stats.average_score = (stats.average_score * (stats.total_verifications - 1) as f64
                + log.score)
                / stats.total_verifications as f64;

            if stats.last_verification.is_none()
                || stats.last_verification.unwrap() < log.timestamp
            {
                stats.last_verification = Some(log.timestamp);
            }
        }

        let mut performance_list: Vec<ExecutorPerformance> = executor_stats.into_values().collect();
        performance_list.sort_by(|a, b| {
            b.success_rate()
                .partial_cmp(&a.success_rate())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(performance_list.into_iter().take(limit as usize).collect())
    }

    /// Get verification trends over time
    pub async fn get_verification_trends(
        &self,
        days: u32,
    ) -> Result<Vec<DailyVerificationStats>, PersistenceError> {
        let start_date = Utc::now() - chrono::Duration::days(days as i64);
        let logs = self
            .verification_repo
            .find_all(Pagination::new(0, 10000))
            .await?;

        let mut daily_stats = std::collections::HashMap::new();

        for log in logs.data {
            if log.timestamp >= start_date {
                let date = log.timestamp.date_naive();
                let stats = daily_stats.entry(date).or_insert(DailyVerificationStats {
                    date,
                    total_verifications: 0,
                    successful_verifications: 0,
                    total_score: 0.0,
                    unique_executors: std::collections::HashSet::new(),
                });

                stats.total_verifications += 1;
                if log.success {
                    stats.successful_verifications += 1;
                }
                stats.total_score += log.score;
                stats.unique_executors.insert(log.executor_id);
            }
        }

        let mut trends: Vec<DailyVerificationStats> = daily_stats.into_values().collect();
        trends.sort_by_key(|s| s.date);

        Ok(trends)
    }

    /// Get executor health summary
    pub async fn get_executor_health_summary(
        &self,
        executor_id: &str,
    ) -> Result<ExecutorHealthSummary, Box<dyn std::error::Error>> {
        let db_stats = self.get_executor_stats(executor_id, Some(30)).await?;
        let journal_stats = self.get_executor_verification_stats(executor_id, 30).await?;

        let health_score = calculate_health_score(&db_stats, &journal_stats);
        let status = determine_health_status(health_score);

        Ok(ExecutorHealthSummary {
            executor_id: executor_id.to_string(),
            health_score,
            status,
            db_stats,
            journal_stats,
            last_checked: Utc::now(),
        })
    }

    /// Get system health overview
    pub async fn get_system_health_overview(&self) -> Result<SystemHealthOverview, Box<dyn std::error::Error>> {
        let journal_stats = self.get_system_verification_stats(7).await?;
        let total_logs = self.verification_repo.count().await?;

        let health_score = if journal_stats.total_verifications > 0 {
            let success_rate = journal_stats.verification_success_rate();
            let security_penalty = if journal_stats.security_violations > 0 {
                0.2
            } else {
                0.0
            };
            let connection_penalty = if journal_stats.connection_failures > journal_stats.total_verifications / 10 {
                0.1
            } else {
                0.0
            };

            (success_rate - security_penalty - connection_penalty).max(0.0)
        } else {
            0.0
        };

        Ok(SystemHealthOverview {
            health_score,
            total_verification_logs: total_logs,
            weekly_stats: journal_stats,
            last_updated: Utc::now(),
        })
    }
}

#[derive(Debug, Clone)]
pub struct ExecutorPerformance {
    pub executor_id: String,
    pub total_verifications: u64,
    pub successful_verifications: u64,
    pub average_score: f64,
    pub last_verification: Option<DateTime<Utc>>,
}

impl ExecutorPerformance {
    pub fn success_rate(&self) -> f64 {
        if self.total_verifications == 0 {
            0.0
        } else {
            self.successful_verifications as f64 / self.total_verifications as f64
        }
    }
}

#[derive(Debug)]
pub struct DailyVerificationStats {
    pub date: chrono::NaiveDate,
    pub total_verifications: u64,
    pub successful_verifications: u64,
    pub total_score: f64,
    pub unique_executors: std::collections::HashSet<String>,
}

impl DailyVerificationStats {
    pub fn success_rate(&self) -> f64 {
        if self.total_verifications == 0 {
            0.0
        } else {
            self.successful_verifications as f64 / self.total_verifications as f64
        }
    }

    pub fn average_score(&self) -> f64 {
        if self.total_verifications == 0 {
            0.0
        } else {
            self.total_score / self.total_verifications as f64
        }
    }

    pub fn unique_executor_count(&self) -> usize {
        self.unique_executors.len()
    }
}

#[derive(Debug)]
pub struct ExecutorHealthSummary {
    pub executor_id: String,
    pub health_score: f64,
    pub status: HealthStatus,
    pub db_stats: ExecutorVerificationStats,
    pub journal_stats: VerificationStats,
    pub last_checked: DateTime<Utc>,
}

#[derive(Debug)]
pub struct SystemHealthOverview {
    pub health_score: f64,
    pub total_verification_logs: u64,
    pub weekly_stats: VerificationStats,
    pub last_updated: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub enum HealthStatus {
    Healthy,
    Warning,
    Critical,
    Unknown,
}

fn calculate_health_score(
    db_stats: &ExecutorVerificationStats,
    journal_stats: &VerificationStats,
) -> f64 {
    if db_stats.total_verifications == 0 {
        return 0.0;
    }

    let success_rate = db_stats.successful_verifications as f64 / db_stats.total_verifications as f64;
    let score_factor = db_stats.average_score.unwrap_or(0.0);
    
    let security_penalty = if journal_stats.security_violations > 0 {
        0.3
    } else {
        0.0
    };
    
    let connection_penalty = if journal_stats.connection_failures > db_stats.total_verifications / 10 {
        0.2
    } else {
        0.0
    };

    ((success_rate * 0.4 + score_factor * 0.6) - security_penalty - connection_penalty).max(0.0)
}

fn determine_health_status(health_score: f64) -> HealthStatus {
    match health_score {
        score if score >= 0.8 => HealthStatus::Healthy,
        score if score >= 0.6 => HealthStatus::Warning,
        score if score > 0.0 => HealthStatus::Critical,
        _ => HealthStatus::Unknown,
    }
}