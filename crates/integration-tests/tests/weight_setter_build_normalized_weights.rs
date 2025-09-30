//! Integration test for WeightSetter build_normalized_weights function

use anyhow::Result;
use basilica_common::identity::MinerUid;
use basilica_validator::bittensor_core::weight_allocation::NormalizedWeight;
use basilica_validator::bittensor_core::weight_allocation::{
    WeightAllocationEngine, WeightDistribution,
};
use basilica_validator::config::emission::{EmissionConfig, GpuAllocation};
use std::collections::HashMap;

struct WeightSetterTester {
    weight_allocation_engine: WeightAllocationEngine,
}

impl WeightSetterTester {
    fn new() -> Self {
        let emission_config = EmissionConfig {
            burn_percentage: 10.0,
            burn_uid: 999,
            gpu_allocations: {
                let mut allocations = HashMap::new();
                allocations.insert("H100".to_string(), GpuAllocation::new(60.0));
                allocations.insert("H200".to_string(), GpuAllocation::new(40.0));
                allocations
            },
            min_miners_per_category: 1,
            weight_set_interval_blocks: 100,
            weight_version_key: 0,
        };

        let weight_allocation_engine = WeightAllocationEngine::new(emission_config, 0.5);

        Self {
            weight_allocation_engine,
        }
    }

    fn test_build_normalized_weights(
        &self,
        weight_distribution: &WeightDistribution,
    ) -> Result<Vec<NormalizedWeight>> {
        let normalized_weights: Vec<NormalizedWeight> = weight_distribution
            .weights
            .iter()
            .map(|w| NormalizedWeight {
                uid: w.uid,
                weight: w.weight,
            })
            .collect();

        assert!(
            !normalized_weights.is_empty(),
            "Weight allocation engine produced no weights - this should never happen"
        );

        Ok(normalized_weights)
    }
}

#[tokio::test]
async fn test_build_normalized_weights_with_miners() -> Result<()> {
    let tester = WeightSetterTester::new();

    let mut miners_by_category = HashMap::new();
    miners_by_category.insert(
        "H100".to_string(),
        vec![(MinerUid::new(1), 0.85), (MinerUid::new(2), 0.72)],
    );
    miners_by_category.insert("H200".to_string(), vec![(MinerUid::new(3), 0.91)]);

    let weight_distribution = tester
        .weight_allocation_engine
        .calculate_weight_distribution(miners_by_category)?;

    println!(
        "Weight distribution: {} regular weights, burn: {:?}",
        weight_distribution.weights.len(),
        weight_distribution.burn_allocation.is_some()
    );

    println!("Regular weights:");
    for w in &weight_distribution.weights {
        println!("  UID {}: weight {}", w.uid, w.weight);
    }

    if let Some(burn) = &weight_distribution.burn_allocation {
        println!("Burn allocation: UID {}, weight {}", burn.uid, burn.weight);
    }

    let normalized_weights = tester.test_build_normalized_weights(&weight_distribution)?;

    println!("Normalized weights: {}", normalized_weights.len());
    for w in &normalized_weights {
        println!("  UID {}: weight {}", w.uid, w.weight);
    }

    let miner_weights: Vec<_> = normalized_weights.iter().filter(|w| w.uid != 999).collect();

    let burn_weights: Vec<_> = normalized_weights.iter().filter(|w| w.uid == 999).collect();

    assert_eq!(burn_weights.len(), 1);
    assert_eq!(miner_weights.len(), 3);

    Ok(())
}

#[tokio::test]
async fn test_build_normalized_weights_burn_only() -> Result<()> {
    let tester = WeightSetterTester::new();

    let miners_by_category = HashMap::new();

    let weight_distribution = tester
        .weight_allocation_engine
        .calculate_weight_distribution(miners_by_category)?;

    let normalized_weights = tester.test_build_normalized_weights(&weight_distribution)?;

    assert_eq!(normalized_weights.len(), 1);

    let miner_weights: Vec<_> = normalized_weights.iter().filter(|w| w.uid != 999).collect();
    assert_eq!(miner_weights.len(), 0);

    let burn_weights: Vec<_> = normalized_weights.iter().filter(|w| w.uid == 999).collect();
    assert_eq!(burn_weights.len(), 1);

    let burn_weight = burn_weights[0];
    assert!(burn_weight.weight > 50000);

    Ok(())
}

#[tokio::test]
async fn test_build_normalized_weights_partial_miners() -> Result<()> {
    let tester = WeightSetterTester::new();

    let mut miners_by_category = HashMap::new();
    miners_by_category.insert(
        "H100".to_string(),
        vec![(MinerUid::new(1), 0.88), (MinerUid::new(2), 0.76)],
    );

    let weight_distribution = tester
        .weight_allocation_engine
        .calculate_weight_distribution(miners_by_category)?;

    let normalized_weights = tester.test_build_normalized_weights(&weight_distribution)?;

    assert_eq!(normalized_weights.len(), 3);

    let miner_weights: Vec<_> = normalized_weights.iter().filter(|w| w.uid != 999).collect();
    assert_eq!(miner_weights.len(), 2);

    let burn_weights: Vec<_> = normalized_weights.iter().filter(|w| w.uid == 999).collect();
    assert_eq!(burn_weights.len(), 1);

    let burn_weight = burn_weights[0];
    assert!(burn_weight.weight > 6000);

    Ok(())
}

#[tokio::test]
async fn test_impossible_condition_eliminated() -> Result<()> {
    let tester = WeightSetterTester::new();

    let test_cases = vec![
        ("with_miners", {
            let mut miners = HashMap::new();
            miners.insert("H100".to_string(), vec![(MinerUid::new(1), 0.85)]);
            miners
        }),
        ("no_miners", HashMap::new()),
        ("partial_miners", {
            let mut miners = HashMap::new();
            miners.insert("H100".to_string(), vec![(MinerUid::new(1), 0.88)]);
            miners
        }),
    ];

    for (_, miners_by_category) in test_cases {
        let weight_distribution = tester
            .weight_allocation_engine
            .calculate_weight_distribution(miners_by_category)?;

        let normalized_weights = tester.test_build_normalized_weights(&weight_distribution)?;

        assert!(!normalized_weights.is_empty());
    }

    Ok(())
}

#[tokio::test]
async fn test_weight_conservation() -> Result<()> {
    let tester = WeightSetterTester::new();

    let mut miners_by_category = HashMap::new();
    miners_by_category.insert(
        "H100".to_string(),
        vec![(MinerUid::new(1), 0.85), (MinerUid::new(2), 0.72)],
    );

    let weight_distribution = tester
        .weight_allocation_engine
        .calculate_weight_distribution(miners_by_category)?;

    let normalized_weights = tester.test_build_normalized_weights(&weight_distribution)?;

    let total_weight: u64 = normalized_weights.iter().map(|w| w.weight as u64).sum();
    let expected_total = weight_distribution.total_weight;

    assert!(total_weight <= expected_total);

    let mut uids = std::collections::HashSet::new();
    for weight in &normalized_weights {
        assert!(uids.insert(weight.uid));
    }

    Ok(())
}
