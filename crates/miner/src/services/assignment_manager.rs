//! # Assignment Manager Service
//!
//! Manages executor assignments and provides intelligent suggestions
//! for optimal stake coverage.

use anyhow::{anyhow, Result};
use sqlx::SqlitePool;
use std::collections::{HashMap, HashSet};
use tracing::{debug, info, warn};

use crate::persistence::{AssignmentDb, CoverageStats, ExecutorAssignment, ValidatorStake};

/// Assignment suggestion with reasoning
#[derive(Debug, Clone)]
pub struct SuggestedAssignment {
    pub executor_id: String,
    pub validator_hotkey: String,
    pub reason: String,
    pub priority: AssignmentPriority,
}

/// Priority levels for assignment suggestions
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum AssignmentPriority {
    Critical, // Required for minimum coverage
    High,     // Significant stake improvement
    Medium,   // Proportional optimization
    Low,      // Minor optimization
}

/// Assignment manager for handling executor assignments
pub struct AssignmentManager {
    assignment_db: AssignmentDb,
    executors: Vec<String>, // Cached from config
}

impl AssignmentManager {
    /// Create a new assignment manager
    pub fn new(pool: SqlitePool) -> Self {
        Self {
            assignment_db: AssignmentDb::new(pool),
            executors: Vec::new(),
        }
    }

    /// Create assignment manager with executor list from config
    pub fn with_executors(mut self, executors: Vec<String>) -> Self {
        self.executors = executors;
        self
    }

    /// Get all executors (from config)
    pub async fn get_all_executors(&self) -> Result<Vec<String>> {
        Ok(self.executors.clone())
    }

    /// Get validator stakes
    pub async fn get_validator_stakes(&self) -> Result<Vec<ValidatorStake>> {
        self.assignment_db.get_all_validator_stakes().await
    }

    /// Get all assignments
    pub async fn get_all_assignments(&self) -> Result<Vec<ExecutorAssignment>> {
        self.assignment_db.get_all_assignments().await
    }

    /// Get coverage statistics
    pub async fn get_coverage_stats(&self) -> Result<CoverageStats> {
        self.assignment_db
            .get_coverage_stats(self.executors.len())
            .await
    }
}

/// Assignment suggester that implements the stake-weighted algorithm
pub struct AssignmentSuggester {
    min_coverage: f64,
    min_stake_for_assignment: f64,
}

impl AssignmentSuggester {
    /// Create a new assignment suggester
    pub fn new(min_coverage: f64) -> Self {
        Self {
            min_coverage,
            min_stake_for_assignment: 1.0, // 1% minimum stake to get assignments
        }
    }

    /// Generate assignment suggestions
    pub async fn suggest_assignments(
        &self,
        executors: Vec<String>,
        validator_stakes: Vec<ValidatorStake>,
        current_assignments: Vec<ExecutorAssignment>,
    ) -> Result<Vec<SuggestedAssignment>> {
        if executors.is_empty() {
            return Err(anyhow!("No executors available for assignment"));
        }

        if validator_stakes.is_empty() {
            return Err(anyhow!("No validator stake information available"));
        }

        let mut suggestions = Vec::new();

        // Build current assignment map
        let mut assignment_map: HashMap<String, String> = HashMap::new();
        let mut assigned_validators: HashSet<String> = HashSet::new();

        for assignment in &current_assignments {
            assignment_map.insert(
                assignment.executor_id.clone(),
                assignment.validator_hotkey.clone(),
            );
            assigned_validators.insert(assignment.validator_hotkey.clone());
        }

        // Get unassigned executors
        let unassigned_executors: Vec<String> = executors
            .iter()
            .filter(|e| !assignment_map.contains_key(*e))
            .cloned()
            .collect();

        if unassigned_executors.is_empty() {
            info!("All executors are already assigned");
            return Ok(suggestions);
        }

        // Sort validators by stake (descending)
        let mut sorted_validators = validator_stakes.clone();
        sorted_validators.sort_by(|a, b| {
            b.percentage_of_total
                .partial_cmp(&a.percentage_of_total)
                .unwrap()
        });

        // Calculate current coverage
        let current_coverage = self.calculate_coverage(&assigned_validators, &validator_stakes);

        info!(
            "Current coverage: {:.2}%, Target: {:.2}%, Unassigned executors: {}",
            current_coverage * 100.0,
            self.min_coverage * 100.0,
            unassigned_executors.len()
        );

        // Phase 1: Ensure minimum coverage (Critical priority)
        let mut remaining_executors = unassigned_executors;
        if current_coverage < self.min_coverage {
            let coverage_suggestions = self.suggest_for_coverage(
                &mut remaining_executors,
                &sorted_validators,
                &assigned_validators,
                current_coverage,
            )?;
            suggestions.extend(coverage_suggestions);
        }

        // Phase 2: Optimize proportional distribution (Medium/Low priority)
        if !remaining_executors.is_empty() {
            let proportional_suggestions = self.suggest_proportional_assignments(
                remaining_executors,
                &sorted_validators,
                &current_assignments,
                executors.len(),
            )?;
            suggestions.extend(proportional_suggestions);
        }

        // Sort suggestions by priority
        suggestions.sort_by(|a, b| a.priority.cmp(&b.priority));

        Ok(suggestions)
    }

    /// Calculate coverage percentage
    fn calculate_coverage(
        &self,
        assigned_validators: &HashSet<String>,
        all_validators: &[ValidatorStake],
    ) -> f64 {
        let covered_stake: f64 = all_validators
            .iter()
            .filter(|v| assigned_validators.contains(&v.validator_hotkey))
            .map(|v| v.percentage_of_total)
            .sum();

        covered_stake / 100.0
    }

    /// Suggest assignments to reach minimum coverage
    fn suggest_for_coverage(
        &self,
        available_executors: &mut Vec<String>,
        sorted_validators: &[ValidatorStake],
        assigned_validators: &HashSet<String>,
        current_coverage: f64,
    ) -> Result<Vec<SuggestedAssignment>> {
        let mut suggestions = Vec::new();
        let mut coverage = current_coverage;
        let mut assigned = assigned_validators.clone();

        for validator in sorted_validators {
            if coverage >= self.min_coverage {
                break;
            }

            // Skip if already assigned
            if assigned.contains(&validator.validator_hotkey) {
                continue;
            }

            // Skip if stake too low
            if validator.percentage_of_total < self.min_stake_for_assignment {
                continue;
            }

            if let Some(executor_id) = available_executors.pop() {
                debug!(
                    "Suggested {} for coverage: {} -> {} ({:.2}% stake)",
                    available_executors.len() + 1,
                    &executor_id[..8],
                    &validator.validator_hotkey[..8],
                    validator.percentage_of_total
                );

                suggestions.push(SuggestedAssignment {
                    executor_id,
                    validator_hotkey: validator.validator_hotkey.clone(),
                    reason: format!(
                        "Critical: Required for {}% minimum coverage (adds {:.2}% stake)",
                        (self.min_coverage * 100.0) as u32,
                        validator.percentage_of_total
                    ),
                    priority: AssignmentPriority::Critical,
                });

                coverage += validator.percentage_of_total / 100.0;
                assigned.insert(validator.validator_hotkey.clone());
            } else {
                warn!(
                    "Not enough executors to reach minimum coverage (need {:.2}% more)",
                    (self.min_coverage - coverage) * 100.0
                );
                break;
            }
        }

        Ok(suggestions)
    }

    /// Suggest proportional assignments for remaining executors
    fn suggest_proportional_assignments(
        &self,
        available_executors: Vec<String>,
        sorted_validators: &[ValidatorStake],
        current_assignments: &[ExecutorAssignment],
        total_executors: usize,
    ) -> Result<Vec<SuggestedAssignment>> {
        let mut suggestions = Vec::new();

        // Count current assignments per validator
        let mut assignment_counts: HashMap<String, usize> = HashMap::new();
        for assignment in current_assignments {
            *assignment_counts
                .entry(assignment.validator_hotkey.clone())
                .or_insert(0) += 1;
        }

        // Calculate target assignments per validator
        let mut validator_targets: Vec<(String, f64, usize, f64)> = Vec::new();

        for validator in sorted_validators {
            if validator.percentage_of_total < self.min_stake_for_assignment {
                continue;
            }

            let target = (validator.percentage_of_total / 100.0) * total_executors as f64;
            let current = assignment_counts
                .get(&validator.validator_hotkey)
                .copied()
                .unwrap_or(0);
            let deficit = target - current as f64;

            if deficit > 0.5 {
                // Only consider if deficit is significant
                validator_targets.push((
                    validator.validator_hotkey.clone(),
                    validator.percentage_of_total,
                    current,
                    deficit,
                ));
            }
        }

        // Sort by deficit (descending)
        validator_targets.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap());

        // Assign executors to validators with largest deficits
        for (i, executor_id) in available_executors.into_iter().enumerate() {
            if let Some((validator_hotkey, stake_pct, current, deficit)) =
                validator_targets.get(i % validator_targets.len())
            {
                let priority = if *deficit > 2.0 {
                    AssignmentPriority::High
                } else if *deficit > 1.0 {
                    AssignmentPriority::Medium
                } else {
                    AssignmentPriority::Low
                };

                suggestions.push(SuggestedAssignment {
                    executor_id,
                    validator_hotkey: validator_hotkey.clone(),
                    reason: format!(
                        "Proportional: Validator has {:.2}% stake but only {} executors (target: {:.1})",
                        stake_pct, current, (*current as f64) + deficit
                    ),
                    priority,
                });
            }
        }

        Ok(suggestions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn create_test_validator(hotkey: &str, stake_pct: f64) -> ValidatorStake {
        ValidatorStake {
            validator_hotkey: hotkey.to_string(),
            stake_amount: stake_pct * 1000.0, // Convert percentage to TAO amount
            percentage_of_total: stake_pct,
            last_updated: Utc::now(),
        }
    }

    #[tokio::test]
    async fn test_assignment_suggester_coverage() {
        let suggester = AssignmentSuggester::new(0.5); // 50% coverage

        let executors = vec![
            "exec1".to_string(),
            "exec2".to_string(),
            "exec3".to_string(),
        ];

        let validators = vec![
            create_test_validator("val1", 30.0),
            create_test_validator("val2", 25.0),
            create_test_validator("val3", 20.0),
            create_test_validator("val4", 15.0),
            create_test_validator("val5", 10.0),
        ];

        let current_assignments = vec![]; // No current assignments

        let suggestions = suggester
            .suggest_assignments(executors, validators, current_assignments)
            .await
            .unwrap();

        // Should suggest assignments to reach 50% coverage
        assert!(!suggestions.is_empty());

        // First suggestions should be critical priority
        assert_eq!(suggestions[0].priority, AssignmentPriority::Critical);

        // Should assign to highest stake validators first
        assert_eq!(suggestions[0].validator_hotkey, "val1");
        assert_eq!(suggestions[1].validator_hotkey, "val2");
    }

    #[tokio::test]
    async fn test_assignment_suggester_proportional() {
        let suggester = AssignmentSuggester::new(0.3); // 30% coverage

        let executors = vec![
            "exec1".to_string(),
            "exec2".to_string(),
            "exec3".to_string(),
            "exec4".to_string(),
            "exec5".to_string(),
        ];

        let validators = vec![
            create_test_validator("val1", 40.0),
            create_test_validator("val2", 30.0),
            create_test_validator("val3", 20.0),
            create_test_validator("val4", 10.0),
        ];

        // Already have coverage with val1
        let current_assignments = vec![ExecutorAssignment {
            id: 1,
            executor_id: "exec0".to_string(),
            validator_hotkey: "val1".to_string(),
            assigned_at: Utc::now(),
            assigned_by: "test".to_string(),
            notes: None,
        }];

        let suggestions = suggester
            .suggest_assignments(executors.clone(), validators, current_assignments)
            .await
            .unwrap();

        // Should suggest proportional distribution
        assert_eq!(suggestions.len(), 5);

        // val1 should get more executors due to 40% stake
        let val1_suggestions: Vec<_> = suggestions
            .iter()
            .filter(|s| s.validator_hotkey == "val1")
            .collect();
        assert!(!val1_suggestions.is_empty());
    }

    #[tokio::test]
    async fn test_assignment_suggester_no_executors() {
        let suggester = AssignmentSuggester::new(0.5);

        let executors = vec![]; // No executors
        let validators = vec![create_test_validator("val1", 100.0)];
        let current_assignments = vec![];

        let result = suggester
            .suggest_assignments(executors, validators, current_assignments)
            .await;

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("No executors"));
    }
}
