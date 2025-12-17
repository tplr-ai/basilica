use crate::error::{BillingError, Result};
use crate::storage::{MinerRevenueRepository, MinerRevenueSummary, MinerRevenueSummaryFilter};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use std::sync::Arc;
use tracing::{info, warn};

/// Operations for miner revenue reconciliation
#[async_trait]
pub trait MinerRevenueOperations: Send + Sync {
    /// Refresh miner revenue summaries for a given time period
    async fn refresh_summary(
        &self,
        period_start: DateTime<Utc>,
        period_end: DateTime<Utc>,
        computation_version: i32,
    ) -> Result<i32>;

    /// Get miner revenue summaries with filtering and pagination
    async fn get_summaries(
        &self,
        filter: MinerRevenueSummaryFilter,
    ) -> Result<(Vec<MinerRevenueSummary>, i64)>;
}

/// Service for managing miner revenue reconciliation
pub struct MinerRevenueService {
    repository: Arc<dyn MinerRevenueRepository + Send + Sync>,
}

impl MinerRevenueService {
    pub fn new(repository: Arc<dyn MinerRevenueRepository + Send + Sync>) -> Self {
        Self { repository }
    }

    /// Validate period boundaries
    fn validate_period(period_start: DateTime<Utc>, period_end: DateTime<Utc>) -> Result<()> {
        if period_end <= period_start {
            return Err(BillingError::InvalidState {
                message: format!(
                    "Period end ({}) must be after period start ({})",
                    period_end, period_start
                ),
            });
        }

        // Warn if period is unusually long (more than 1 year)
        let duration = period_end - period_start;
        if duration.num_days() > 365 {
            warn!(
                "Unusually long period: {} days (start: {}, end: {})",
                duration.num_days(),
                period_start,
                period_end
            );
        }

        Ok(())
    }
}

#[async_trait]
impl MinerRevenueOperations for MinerRevenueService {
    async fn refresh_summary(
        &self,
        period_start: DateTime<Utc>,
        period_end: DateTime<Utc>,
        computation_version: i32,
    ) -> Result<i32> {
        // Validate inputs
        Self::validate_period(period_start, period_end)?;

        if computation_version < 1 {
            return Err(BillingError::InvalidState {
                message: format!(
                    "Computation version must be >= 1, got {}",
                    computation_version
                ),
            });
        }

        // Check for overlapping periods
        let overlaps = self
            .repository
            .find_overlapping_periods(period_start, period_end)
            .await?;
        if !overlaps.is_empty() {
            let overlap_details: Vec<String> = overlaps
                .iter()
                .map(|o| {
                    format!(
                        "{} to {} (computed {})",
                        o.period_start.format("%Y-%m-%d"),
                        o.period_end.format("%Y-%m-%d"),
                        o.computed_at.format("%Y-%m-%d %H:%M:%S")
                    )
                })
                .collect();
            return Err(BillingError::InvalidState {
                message: format!(
                    "Cannot calculate for {} to {}: overlaps with existing periods: {}",
                    period_start.format("%Y-%m-%d"),
                    period_end.format("%Y-%m-%d"),
                    overlap_details.join(", ")
                ),
            });
        }

        info!(
            "Refreshing miner revenue summary for period {} to {} (version {})",
            period_start, period_end, computation_version
        );

        // Call repository to execute stored function
        let rows_created = self
            .repository
            .refresh_summary_for_period(period_start, period_end, computation_version)
            .await?;

        info!(
            "Created {} miner revenue summary rows for period {} to {}",
            rows_created, period_start, period_end
        );

        Ok(rows_created)
    }

    async fn get_summaries(
        &self,
        filter: MinerRevenueSummaryFilter,
    ) -> Result<(Vec<MinerRevenueSummary>, i64)> {
        // Validate period if provided
        if let (Some(start), Some(end)) = (filter.period_start, filter.period_end) {
            Self::validate_period(start, end)?;
        }

        // Validate pagination
        if let Some(limit) = filter.limit {
            if !(1..=1000).contains(&limit) {
                return Err(BillingError::InvalidState {
                    message: format!("Limit must be between 1 and 1000, got {}", limit),
                });
            }
        }

        if let Some(offset) = filter.offset {
            if offset < 0 {
                return Err(BillingError::InvalidState {
                    message: format!("Offset must be >= 0, got {}", offset),
                });
            }
        }

        // Fetch summaries and count in parallel
        let summaries_future = self.repository.get_summaries(&filter);
        let count_future = self.repository.count_summaries(&filter);

        let (summaries, total_count) = tokio::try_join!(summaries_future, count_future)?;

        info!(
            "Retrieved {} miner revenue summaries (total: {})",
            summaries.len(),
            total_count
        );

        Ok((summaries, total_count))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;

    #[test]
    fn test_validate_period_valid() {
        let start = Utc::now();
        let end = start + Duration::days(30);
        assert!(MinerRevenueService::validate_period(start, end).is_ok());
    }

    #[test]
    fn test_validate_period_end_before_start() {
        let start = Utc::now();
        let end = start - Duration::days(1);
        let result = MinerRevenueService::validate_period(start, end);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("must be after period start"));
    }

    #[test]
    fn test_validate_period_equal() {
        let time = Utc::now();
        let result = MinerRevenueService::validate_period(time, time);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_period_very_long() {
        let start = Utc::now();
        let end = start + Duration::days(400);
        // Should succeed but log a warning
        assert!(MinerRevenueService::validate_period(start, end).is_ok());
    }
}
