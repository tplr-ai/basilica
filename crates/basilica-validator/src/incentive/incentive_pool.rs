use crate::basilica_api::{CuLedgerRowResponse, IncentiveConfigResponse, RuLedgerRowResponse};
use crate::bittensor_core::weight_allocation::{
    BurnAllocation, CategoryAllocation, NormalizedWeight, WeightDistribution,
};
use anyhow::Result;
use chrono::{DateTime, Utc};
use rust_decimal::prelude::ToPrimitive;
use rust_decimal::Decimal;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use tracing::{debug, info, warn};

#[derive(Debug, Clone)]
pub struct IncentivePoolResult {
    pub distribution: WeightDistribution,
    pub burn_rate: Decimal,
    pub burn_percentage: f64,
    pub usd_required_epoch: Decimal,
    pub usd_emission_capacity: Decimal,
    pub category_payouts: HashMap<String, Decimal>,
    pub miner_payouts: HashMap<String, Decimal>,
}

pub fn compute_cu_vested_fraction(
    row: &CuLedgerRowResponse,
    epoch_start: DateTime<Utc>,
    epoch_end: DateTime<Utc>,
) -> Decimal {
    compute_vested_fraction(row.earned_at, row.window_hours, epoch_start, epoch_end)
}

pub fn compute_ru_vested_fraction(
    row: &RuLedgerRowResponse,
    epoch_start: DateTime<Utc>,
    epoch_end: DateTime<Utc>,
) -> Decimal {
    compute_vested_fraction(row.earned_at, row.window_hours, epoch_start, epoch_end)
}

#[allow(clippy::too_many_arguments)]
pub fn compute_incentive_pool(
    config: &IncentiveConfigResponse,
    cu_rows: &[CuLedgerRowResponse],
    ru_rows: &[RuLedgerRowResponse],
    epoch_start: DateTime<Utc>,
    epoch_end: DateTime<Utc>,
    usd_emission_capacity: Decimal,
    burn_uid: u16,
    hotkey_to_uid: &HashMap<String, u16>,
    forced_burn_percentage: Option<f64>,
) -> Result<IncentivePoolResult> {
    let forced_pct = Decimal::from_f64_retain(forced_burn_percentage.unwrap_or(0.0) / 100.0)
        .unwrap_or(Decimal::ZERO);
    let available_fraction = Decimal::ONE - forced_pct;
    let effective_usd_capacity = usd_emission_capacity * available_fraction;
    let forced_weight = (forced_pct * Decimal::from(u16::MAX))
        .round()
        .to_u16()
        .unwrap_or(0);
    let available_weight = u16::MAX - forced_weight;

    let active_cu_rows: Vec<&CuLedgerRowResponse> = cu_rows
        .iter()
        .filter(|row| !row.is_slashed)
        .filter(|row| hotkey_to_uid.contains_key(&row.hotkey))
        .filter(|row| config.gpu_categories.contains_key(&row.gpu_category))
        .collect();
    let active_ru_rows: Vec<&RuLedgerRowResponse> = ru_rows
        .iter()
        .filter(|row| !row.is_slashed)
        .filter(|row| hotkey_to_uid.contains_key(&row.hotkey))
        .collect();

    info!(
        total_cu_rows = cu_rows.len(),
        active_cu_rows = active_cu_rows.len(),
        total_ru_rows = ru_rows.len(),
        active_ru_rows = active_ru_rows.len(),
        "Incentive pool input filtering"
    );

    let mut category_cu_supply: HashMap<String, Decimal> = HashMap::new();
    for row in &active_cu_rows {
        *category_cu_supply
            .entry(row.gpu_category.clone())
            .or_insert(Decimal::ZERO) += row.cu_amount;
    }

    info!(
        category_cu_supply = ?category_cu_supply,
        "Category CU supply (dilution denominators)"
    );

    let mut miner_payouts = HashMap::new();
    let mut category_payouts = HashMap::new();
    let mut category_miners: HashMap<String, HashSet<String>> = HashMap::new();

    for row in active_cu_rows {
        let vested_fraction = compute_cu_vested_fraction(row, epoch_start, epoch_end);
        if vested_fraction <= Decimal::ZERO {
            continue;
        }

        let Some(category_config) = config.gpu_categories.get(&row.gpu_category) else {
            continue;
        };
        let category_supply = category_cu_supply
            .get(&row.gpu_category)
            .copied()
            .unwrap_or(Decimal::ZERO);
        if category_supply <= Decimal::ZERO {
            continue;
        }

        let target_gpus = Decimal::from(category_config.target_count) * Decimal::from(8u32);
        let row_capacity_budget = target_gpus * Decimal::from(row.window_hours) * row.price_usd;
        let per_cu_budget = row_capacity_budget / category_supply;
        let effective_price = min_decimal(
            row.price_usd,
            min_decimal(per_cu_budget, config.max_cu_value_usd),
        );
        let row_payout = vested_fraction * row.cu_amount * effective_price;
        if row_payout <= Decimal::ZERO {
            continue;
        }

        *miner_payouts
            .entry(row.hotkey.clone())
            .or_insert(Decimal::ZERO) += row_payout;
        *category_payouts
            .entry(row.gpu_category.clone())
            .or_insert(Decimal::ZERO) += row_payout;
        category_miners
            .entry(row.gpu_category.clone())
            .or_default()
            .insert(row.hotkey.clone());
    }

    for row in active_ru_rows {
        let vested_fraction = compute_ru_vested_fraction(row, epoch_start, epoch_end);
        if vested_fraction <= Decimal::ZERO {
            continue;
        }

        let row_payout = vested_fraction * row.ru_amount * Decimal::from(row.revenue_share_pct)
            / Decimal::from(100u32);
        if row_payout <= Decimal::ZERO {
            continue;
        }

        *miner_payouts
            .entry(row.hotkey.clone())
            .or_insert(Decimal::ZERO) += row_payout;
        *category_payouts
            .entry(row.gpu_category.clone())
            .or_insert(Decimal::ZERO) += row_payout;
        category_miners
            .entry(row.gpu_category.clone())
            .or_default()
            .insert(row.hotkey.clone());
    }

    let raw_usd_required_epoch = sum_decimals(miner_payouts.values().copied());

    if raw_usd_required_epoch <= Decimal::ZERO || effective_usd_capacity <= Decimal::ZERO {
        warn!(
            raw_usd_required = %raw_usd_required_epoch,
            effective_usd_capacity = %effective_usd_capacity,
            "All weight to burn: no payouts or zero emission capacity"
        );
        return Ok(all_burn_result(
            burn_uid,
            usd_emission_capacity,
            category_payouts,
            miner_payouts,
        ));
    }

    let scale_factor = if raw_usd_required_epoch > effective_usd_capacity {
        effective_usd_capacity / raw_usd_required_epoch
    } else {
        Decimal::ONE
    };

    if scale_factor < Decimal::ONE {
        info!(
            scale_factor = %scale_factor,
            "Emission cap active: scaling all payouts"
        );
    }

    let scaled_miner_payouts = scale_decimal_map(&miner_payouts, scale_factor);
    let scaled_category_payouts = scale_decimal_map(&category_payouts, scale_factor);

    debug!(
        miner_payouts = ?scaled_miner_payouts,
        "Per-miner payouts (scaled)"
    );

    let usd_required_epoch = sum_decimals(scaled_miner_payouts.values().copied());
    if usd_required_epoch <= Decimal::ZERO {
        warn!("All weight to burn: scaled payouts rounded to zero");
        return Ok(all_burn_result(
            burn_uid,
            usd_emission_capacity,
            scaled_category_payouts,
            scaled_miner_payouts,
        ));
    }

    let burn_rate =
        (Decimal::ONE - (usd_required_epoch / effective_usd_capacity)).max(Decimal::ZERO);
    let burn_share = if usd_required_epoch >= effective_usd_capacity {
        Decimal::ZERO
    } else {
        burn_rate
    };

    let mut shares_by_uid: HashMap<u16, Decimal> = HashMap::new();
    for (hotkey, payout) in &scaled_miner_payouts {
        if *payout <= Decimal::ZERO {
            continue;
        }
        let Some(uid) = hotkey_to_uid.get(hotkey).copied() else {
            continue;
        };
        *shares_by_uid.entry(uid).or_insert(Decimal::ZERO) += *payout / effective_usd_capacity;
    }
    if burn_share > Decimal::ZERO {
        *shares_by_uid.entry(burn_uid).or_insert(Decimal::ZERO) += burn_share;
    } else {
        shares_by_uid.entry(burn_uid).or_insert(Decimal::ZERO);
    }

    let mut weights = normalize_uid_shares(&shares_by_uid, available_weight);

    // Add forced burn weight to burn_uid
    if forced_weight > 0 {
        if let Some(entry) = weights.iter_mut().find(|w| w.uid == burn_uid) {
            entry.weight += forced_weight;
        } else {
            weights.push(NormalizedWeight {
                uid: burn_uid,
                weight: forced_weight,
            });
            weights.sort_by_key(|w| w.uid);
        }
    }

    let burn_weight = weights
        .iter()
        .find(|weight| weight.uid == burn_uid)
        .map(|weight| weight.weight)
        .unwrap_or(0);
    let total_weight = u16::MAX as u64;
    let miners_served = weights
        .iter()
        .filter(|weight| weight.uid != burn_uid && weight.weight > 0)
        .count() as u32;

    let category_allocations = build_category_allocations(
        &scaled_category_payouts,
        &category_miners,
        total_weight,
        burn_weight,
        usd_required_epoch,
    );
    let burn_percentage = ((Decimal::from(burn_weight) * Decimal::from(100u32))
        / Decimal::from(total_weight))
    .to_f64()
    .unwrap_or(0.0);

    Ok(IncentivePoolResult {
        distribution: WeightDistribution {
            weights,
            burn_allocation: Some(BurnAllocation {
                uid: burn_uid,
                weight: burn_weight,
                percentage: burn_percentage,
            }),
            category_allocations,
            total_weight,
            miners_served,
        },
        burn_rate,
        burn_percentage,
        usd_required_epoch,
        usd_emission_capacity,
        category_payouts: scaled_category_payouts,
        miner_payouts: scaled_miner_payouts,
    })
}

fn compute_vested_fraction(
    earned_at: DateTime<Utc>,
    window_hours: u32,
    epoch_start: DateTime<Utc>,
    epoch_end: DateTime<Utc>,
) -> Decimal {
    if window_hours == 0 {
        return Decimal::ZERO;
    }

    let window_ms = (window_hours as i128) * 3_600_000;
    let row_start_ms = earned_at.timestamp_millis() as i128;
    let row_end_ms = row_start_ms + window_ms;
    let overlap_start_ms = row_start_ms.max(epoch_start.timestamp_millis() as i128);
    let overlap_end_ms = row_end_ms.min(epoch_end.timestamp_millis() as i128);
    if overlap_end_ms <= overlap_start_ms {
        return Decimal::ZERO;
    }

    Decimal::from_i128_with_scale(overlap_end_ms - overlap_start_ms, 0)
        / Decimal::from_i128_with_scale(window_ms, 0)
}

fn scale_decimal_map(
    values: &HashMap<String, Decimal>,
    scale_factor: Decimal,
) -> HashMap<String, Decimal> {
    values
        .iter()
        .map(|(key, value)| (key.clone(), *value * scale_factor))
        .collect()
}

fn all_burn_result(
    burn_uid: u16,
    usd_emission_capacity: Decimal,
    category_payouts: HashMap<String, Decimal>,
    miner_payouts: HashMap<String, Decimal>,
) -> IncentivePoolResult {
    IncentivePoolResult {
        distribution: WeightDistribution {
            weights: vec![NormalizedWeight {
                uid: burn_uid,
                weight: u16::MAX,
            }],
            burn_allocation: Some(BurnAllocation {
                uid: burn_uid,
                weight: u16::MAX,
                percentage: 100.0,
            }),
            category_allocations: HashMap::new(),
            total_weight: u16::MAX as u64,
            miners_served: 0,
        },
        burn_rate: Decimal::ONE,
        burn_percentage: 100.0,
        usd_required_epoch: Decimal::ZERO,
        usd_emission_capacity,
        category_payouts,
        miner_payouts,
    }
}

fn normalize_uid_shares(
    shares_by_uid: &HashMap<u16, Decimal>,
    target_total: u16,
) -> Vec<NormalizedWeight> {
    let total_weight = Decimal::from(target_total);
    let mut candidates: Vec<WeightCandidate> = shares_by_uid
        .iter()
        .filter(|(_, share)| **share > Decimal::ZERO)
        .map(|(uid, share)| {
            let raw_weight = *share * total_weight;
            let floored = raw_weight.trunc().to_u64().unwrap_or(0);
            WeightCandidate {
                uid: *uid,
                base_weight: floored,
                remainder: raw_weight - Decimal::from(floored),
            }
        })
        .collect();

    let base_total: u64 = candidates
        .iter()
        .map(|candidate| candidate.base_weight)
        .sum();
    let mut leftover = (target_total as u64).saturating_sub(base_total);
    candidates.sort_by(weight_candidate_order);
    for candidate in &mut candidates {
        if leftover == 0 {
            break;
        }
        candidate.base_weight += 1;
        leftover -= 1;
    }

    let mut weights: Vec<NormalizedWeight> = candidates
        .into_iter()
        .filter(|candidate| candidate.base_weight > 0)
        .map(|candidate| NormalizedWeight {
            uid: candidate.uid,
            weight: candidate.base_weight.min(u16::MAX as u64) as u16,
        })
        .collect();
    weights.sort_by_key(|weight| weight.uid);
    weights
}

fn weight_candidate_order(left: &WeightCandidate, right: &WeightCandidate) -> Ordering {
    right
        .remainder
        .cmp(&left.remainder)
        .then_with(|| left.uid.cmp(&right.uid))
}

fn build_category_allocations(
    category_payouts: &HashMap<String, Decimal>,
    category_miners: &HashMap<String, HashSet<String>>,
    total_weight: u64,
    burn_weight: u16,
    usd_required_epoch: Decimal,
) -> HashMap<String, CategoryAllocation> {
    let miner_weight_total = total_weight.saturating_sub(burn_weight as u64);
    let mut categories = HashMap::new();

    for (category, payout) in category_payouts {
        if *payout <= Decimal::ZERO || usd_required_epoch <= Decimal::ZERO {
            continue;
        }

        let share = *payout / usd_required_epoch;
        let weight_pool = (share * Decimal::from(miner_weight_total))
            .round_dp(0)
            .to_u64()
            .unwrap_or(0);
        let allocation_percentage = (share * Decimal::from(100u32)).to_f64().unwrap_or(0.0);
        let miner_count = category_miners
            .get(category)
            .map(|miners| miners.len() as u32)
            .unwrap_or(0);

        categories.insert(
            category.clone(),
            CategoryAllocation {
                gpu_model: category.clone(),
                miner_count,
                total_score: payout.to_f64().unwrap_or(0.0),
                weight_pool,
                allocation_percentage,
            },
        );
    }

    categories
}

fn sum_decimals<I>(values: I) -> Decimal
where
    I: IntoIterator<Item = Decimal>,
{
    values
        .into_iter()
        .fold(Decimal::ZERO, |acc, value| acc + value)
}

fn min_decimal(left: Decimal, right: Decimal) -> Decimal {
    if left <= right {
        left
    } else {
        right
    }
}

#[derive(Debug, Clone)]
struct WeightCandidate {
    uid: u16,
    base_weight: u64,
    remainder: Decimal,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basilica_api::{IncentiveGpuCategoryConfig, PostSlashResponse};
    use chrono::TimeZone;
    use std::str::FromStr;
    use uuid::Uuid;

    fn d(value: &str) -> Decimal {
        Decimal::from_str(value).unwrap()
    }

    fn ts(hour: i64) -> DateTime<Utc> {
        Utc.timestamp_opt(hour * 3600, 0).unwrap()
    }

    fn ts_minutes(minutes: i64) -> DateTime<Utc> {
        Utc.timestamp_opt(minutes * 60, 0).unwrap()
    }

    fn miner_payout(result: &IncentivePoolResult, hotkey: &str) -> Decimal {
        result
            .miner_payouts
            .get(hotkey)
            .copied()
            .unwrap_or(Decimal::ZERO)
    }

    fn make_hotkey_map(n: usize) -> HashMap<String, u16> {
        (0..n).map(|i| (format!("miner-{i}"), i as u16)).collect()
    }

    fn make_config(
        categories: &[(&str, u32, &str)],
        window_hours: u32,
        max_cu_value_usd: &str,
        revenue_share_pct: Option<u32>,
    ) -> IncentiveConfigResponse {
        let mut gpu_categories = HashMap::new();
        for (name, target_count, price_per_gpu_usd) in categories {
            gpu_categories.insert(
                name.to_string(),
                IncentiveGpuCategoryConfig {
                    target_count: *target_count,
                    price_per_gpu_usd: d(price_per_gpu_usd),
                },
            );
        }
        IncentiveConfigResponse {
            gpu_categories,
            window_hours,
            max_cu_value_usd: d(max_cu_value_usd),
            revenue_share_pct,
            slash_pct: 100,
        }
    }

    fn test_config() -> IncentiveConfigResponse {
        let mut gpu_categories = HashMap::new();
        gpu_categories.insert(
            "H100".to_string(),
            IncentiveGpuCategoryConfig {
                target_count: 1,
                price_per_gpu_usd: d("10"),
            },
        );
        gpu_categories.insert(
            "A100".to_string(),
            IncentiveGpuCategoryConfig {
                target_count: 2,
                price_per_gpu_usd: d("8"),
            },
        );

        IncentiveConfigResponse {
            gpu_categories,
            window_hours: 4,
            max_cu_value_usd: d("100"),
            revenue_share_pct: Some(25),
            slash_pct: 100,
        }
    }

    type CuRowArgs<'a> = (
        &'a str,
        u32,
        &'a str,
        &'a str,
        DateTime<Utc>,
        &'a str,
        u32,
        &'a str,
    );

    fn cu_row(
        (
            hotkey,
            miner_uid,
            node_id,
            cu_amount,
            earned_at,
            gpu_category,
            window_hours,
            price_usd,
        ): CuRowArgs<'_>,
    ) -> CuLedgerRowResponse {
        CuLedgerRowResponse {
            id: Uuid::new_v4(),
            hotkey: hotkey.to_string(),
            miner_uid,
            node_id: node_id.to_string(),
            cu_amount: d(cu_amount),
            earned_at,
            is_rented: false,
            gpu_category: gpu_category.to_string(),
            window_hours,
            price_usd: d(price_usd),
            idempotency_key: format!("{hotkey}-{node_id}"),
            is_slashed: false,
            slash_audit_id: None,
            created_at: earned_at,
        }
    }

    type RuRowArgs<'a> = (
        &'a str,
        u32,
        &'a str,
        &'a str,
        DateTime<Utc>,
        &'a str,
        u32,
        u32,
    );

    fn ru_row(
        (
            hotkey,
            miner_uid,
            node_id,
            ru_amount,
            earned_at,
            gpu_category,
            window_hours,
            revenue_share_pct,
        ): RuRowArgs<'_>,
    ) -> RuLedgerRowResponse {
        RuLedgerRowResponse {
            id: Uuid::new_v4(),
            hotkey: hotkey.to_string(),
            miner_uid,
            node_id: node_id.to_string(),
            ru_amount: d(ru_amount),
            earned_at,
            gpu_category: gpu_category.to_string(),
            window_hours,
            revenue_share_pct,
            period_start: earned_at - chrono::Duration::hours(1),
            period_end: earned_at,
            is_slashed: false,
            slash_audit_id: None,
            created_at: earned_at,
        }
    }

    fn hotkey_to_uid() -> HashMap<String, u16> {
        HashMap::from([
            ("miner-1".to_string(), 11),
            ("miner-2".to_string(), 22),
            ("miner-3".to_string(), 33),
        ])
    }

    fn miner_weight(result: &IncentivePoolResult, uid: u16) -> u16 {
        result
            .distribution
            .weights
            .iter()
            .find(|weight| weight.uid == uid)
            .map(|weight| weight.weight)
            .unwrap_or(0)
    }

    #[test]
    fn test_cu_vesting_behavior() {
        let row = cu_row(("miner-1", 11, "node-1", "4", ts(0), "H100", 4, "10"));

        let vested_fraction = compute_cu_vested_fraction(&row, ts(2), ts(4));

        assert_eq!(vested_fraction, d("0.5"));
    }

    #[test]
    fn test_ru_vesting_behavior() {
        let row = ru_row(("miner-1", 11, "node-1", "40", ts(0), "H100", 4, 25));

        let vested_fraction = compute_ru_vested_fraction(&row, ts(1), ts(3));

        assert_eq!(vested_fraction, d("0.5"));
    }

    #[test]
    fn test_cu_dilution_by_category_target_and_supply() {
        let config = test_config();
        let result = compute_incentive_pool(
            &config,
            &[
                cu_row(("miner-1", 11, "node-1", "4", ts(0), "H100", 4, "10")),
                cu_row(("miner-2", 22, "node-2", "4", ts(0), "H100", 4, "10")),
            ],
            &[],
            ts(0),
            ts(4),
            d("100"),
            999,
            &hotkey_to_uid(),
            None,
        )
        .unwrap();

        assert!(miner_weight(&result, 11) > 0);
        assert_eq!(miner_weight(&result, 11), miner_weight(&result, 22));
    }

    #[test]
    fn test_ru_revenue_share_payout() {
        let config = test_config();
        let result = compute_incentive_pool(
            &config,
            &[],
            &[ru_row((
                "miner-1",
                11,
                "node-1",
                "80",
                ts(0),
                "H100",
                4,
                25,
            ))],
            ts(0),
            ts(4),
            d("100"),
            999,
            &hotkey_to_uid(),
            None,
        )
        .unwrap();

        assert!(miner_weight(&result, 11) > 0);
        assert_eq!(miner_weight(&result, 22), 0);
    }

    #[test]
    fn test_combined_cu_and_ru_payout_behavior() {
        let config = test_config();
        let result = compute_incentive_pool(
            &config,
            &[cu_row((
                "miner-1",
                11,
                "node-1",
                "4",
                ts(0),
                "H100",
                4,
                "10",
            ))],
            &[ru_row((
                "miner-2",
                22,
                "node-2",
                "40",
                ts(0),
                "H100",
                4,
                25,
            ))],
            ts(0),
            ts(4),
            d("100"),
            999,
            &hotkey_to_uid(),
            None,
        )
        .unwrap();

        assert!(miner_weight(&result, 11) > 0);
        assert!(miner_weight(&result, 22) > 0);
        assert!(result.distribution.burn_allocation.is_some());
    }

    #[test]
    fn test_emission_cap_scale_down_behavior() {
        let config = test_config();
        let result = compute_incentive_pool(
            &config,
            &[
                cu_row(("miner-1", 11, "node-1", "20", ts(0), "H100", 4, "10")),
                cu_row(("miner-2", 22, "node-2", "20", ts(0), "A100", 4, "8")),
            ],
            &[ru_row((
                "miner-3",
                33,
                "node-3",
                "100",
                ts(0),
                "H100",
                4,
                25,
            ))],
            ts(0),
            ts(4),
            d("10"),
            999,
            &hotkey_to_uid(),
            None,
        )
        .unwrap();

        assert_eq!(result.burn_rate, Decimal::ZERO);
        assert_eq!(
            result
                .distribution
                .burn_allocation
                .as_ref()
                .map(|allocation| allocation.weight)
                .unwrap_or(0),
            0
        );
    }

    #[test]
    fn test_dynamic_burn_behavior() {
        let config = test_config();
        let result = compute_incentive_pool(
            &config,
            &[cu_row((
                "miner-1",
                11,
                "node-1",
                "2",
                ts(0),
                "H100",
                4,
                "10",
            ))],
            &[],
            ts(0),
            ts(4),
            d("1000"),
            999,
            &hotkey_to_uid(),
            None,
        )
        .unwrap();

        assert!(result.burn_rate > Decimal::ZERO);
        assert!(result.burn_rate <= Decimal::ONE);
        assert!(
            result
                .distribution
                .burn_allocation
                .as_ref()
                .map(|allocation| allocation.weight)
                .unwrap_or(0)
                > 0
        );
    }

    #[test]
    fn test_alpha_price_zero_routes_all_weight_to_burn() {
        let config = test_config();
        let result = compute_incentive_pool(
            &config,
            &[cu_row((
                "miner-1",
                11,
                "node-1",
                "4",
                ts(0),
                "H100",
                4,
                "10",
            ))],
            &[ru_row((
                "miner-1",
                11,
                "node-1",
                "80",
                ts(0),
                "H100",
                4,
                25,
            ))],
            ts(0),
            ts(4),
            Decimal::ZERO,
            999,
            &hotkey_to_uid(),
            None,
        )
        .unwrap();

        assert_eq!(result.distribution.weights.len(), 1);
        assert_eq!(result.distribution.weights[0].uid, 999);
        assert_eq!(result.distribution.weights[0].weight, u16::MAX);
    }

    #[test]
    fn test_zero_cu_zero_ru_and_both_zero_cases() {
        let config = test_config();

        let only_ru = compute_incentive_pool(
            &config,
            &[],
            &[ru_row((
                "miner-2",
                22,
                "node-2",
                "40",
                ts(0),
                "H100",
                4,
                25,
            ))],
            ts(0),
            ts(4),
            d("100"),
            999,
            &hotkey_to_uid(),
            None,
        )
        .unwrap();
        assert!(miner_weight(&only_ru, 22) > 0);

        let only_cu = compute_incentive_pool(
            &config,
            &[cu_row((
                "miner-1",
                11,
                "node-1",
                "4",
                ts(0),
                "H100",
                4,
                "10",
            ))],
            &[],
            ts(0),
            ts(4),
            d("100"),
            999,
            &hotkey_to_uid(),
            None,
        )
        .unwrap();
        assert!(miner_weight(&only_cu, 11) > 0);

        let neither = compute_incentive_pool(
            &config,
            &[],
            &[],
            ts(0),
            ts(4),
            d("100"),
            999,
            &hotkey_to_uid(),
            None,
        )
        .unwrap();
        assert_eq!(neither.distribution.weights.len(), 1);
        assert_eq!(neither.distribution.weights[0].uid, 999);
    }

    #[test]
    fn test_weight_computation_is_deterministic_for_same_inputs() {
        let config = test_config();
        let cu_rows = vec![
            cu_row(("miner-1", 11, "node-1", "4", ts(0), "H100", 4, "10")),
            cu_row(("miner-2", 22, "node-2", "8", ts(0), "A100", 4, "8")),
        ];
        let ru_rows = vec![ru_row((
            "miner-3",
            33,
            "node-3",
            "20",
            ts(0),
            "H100",
            4,
            25,
        ))];
        let hotkey_to_uid = hotkey_to_uid();

        let left = compute_incentive_pool(
            &config,
            &cu_rows,
            &ru_rows,
            ts(0),
            ts(4),
            d("200"),
            999,
            &hotkey_to_uid,
            None,
        )
        .unwrap();
        let right = compute_incentive_pool(
            &config,
            &cu_rows,
            &ru_rows,
            ts(0),
            ts(4),
            d("200"),
            999,
            &hotkey_to_uid,
            None,
        )
        .unwrap();

        let left_weights: Vec<(u16, u16)> = left
            .distribution
            .weights
            .iter()
            .map(|weight| (weight.uid, weight.weight))
            .collect();
        let right_weights: Vec<(u16, u16)> = right
            .distribution
            .weights
            .iter()
            .map(|weight| (weight.uid, weight.weight))
            .collect();

        assert_eq!(left_weights, right_weights);
        assert_eq!(left.burn_rate, right.burn_rate);
    }

    #[test]
    fn test_post_slash_response_type_stays_constructible() {
        let response = PostSlashResponse {
            slashed_cu_count: 1,
            slashed_ru_count: 2,
        };
        assert_eq!(response.slashed_cu_count + response.slashed_ru_count, 3);
    }

    #[test]
    fn test_low_demand_epoch_weights_sum_to_max() {
        let config = test_config();
        let result = compute_incentive_pool(
            &config,
            &[cu_row((
                "miner-1",
                11,
                "node-1",
                "0.1",
                ts(0),
                "H100",
                4,
                "10",
            ))],
            &[],
            ts(0),
            ts(4),
            d("10000"),
            999,
            &hotkey_to_uid(),
            None,
        )
        .unwrap();

        assert!(result.burn_rate > d("0.99"));
        let total: u64 = result
            .distribution
            .weights
            .iter()
            .map(|w| w.weight as u64)
            .sum();
        assert_eq!(total, u16::MAX as u64, "weights must sum to u16::MAX");
    }

    // ==================== Category 1: Vesting Fraction Exactness ====================

    #[test]
    fn test_vesting_full_overlap() {
        let row = cu_row(("miner-1", 11, "node-1", "8", ts(0), "H100", 4, "10"));
        let fraction = compute_cu_vested_fraction(&row, ts(0), ts(4));
        assert_eq!(fraction, d("1"));
    }

    #[test]
    fn test_vesting_partial_overlap_from_start() {
        let row = cu_row(("miner-1", 11, "node-1", "8", ts(0), "H100", 4, "10"));
        let fraction = compute_cu_vested_fraction(&row, ts(0), ts(2));
        assert_eq!(fraction, d("0.5"));
    }

    #[test]
    fn test_vesting_partial_overlap_from_middle() {
        let row = cu_row(("miner-1", 11, "node-1", "8", ts(0), "H100", 4, "10"));
        let fraction = compute_cu_vested_fraction(&row, ts(2), ts(4));
        assert_eq!(fraction, d("0.5"));
    }

    #[test]
    fn test_vesting_small_slice() {
        let row = cu_row(("miner-1", 11, "node-1", "8", ts(0), "H100", 72, "10"));
        let fraction = compute_cu_vested_fraction(&row, ts(50), ts(51));
        // 1 hour overlap out of 72 hour window
        // overlap_ms = 3_600_000, window_ms = 259_200_000
        let expected = Decimal::from(3_600_000i64) / Decimal::from(259_200_000i64);
        assert_eq!(fraction, expected);
    }

    #[test]
    fn test_vesting_cu_earned_mid_epoch() {
        // earned_at = 50.5 hours = 3030 minutes
        let row = cu_row((
            "miner-1",
            11,
            "node-1",
            "8",
            ts_minutes(3030),
            "H100",
            72,
            "10",
        ));
        let fraction = compute_cu_vested_fraction(&row, ts(50), ts(51));
        // overlap = min(122.5h, 51h) - max(50.5h, 50h) = 51 - 50.5 = 0.5h
        // overlap_ms = 1_800_000, window_ms = 259_200_000
        let expected = Decimal::from(1_800_000i64) / Decimal::from(259_200_000i64);
        assert_eq!(fraction, expected);
    }

    #[test]
    fn test_vesting_no_overlap_cu_before_epoch() {
        // CU window [T0, T24], epoch [T50, T51] → no overlap
        let row = cu_row(("miner-1", 11, "node-1", "8", ts(0), "H100", 24, "10"));
        let fraction = compute_cu_vested_fraction(&row, ts(50), ts(51));
        assert_eq!(fraction, Decimal::ZERO);
    }

    #[test]
    fn test_vesting_no_overlap_cu_after_epoch() {
        // CU starts at T60, epoch ends at T51 → no overlap
        let row = cu_row(("miner-1", 11, "node-1", "8", ts(60), "H100", 72, "10"));
        let fraction = compute_cu_vested_fraction(&row, ts(50), ts(51));
        assert_eq!(fraction, Decimal::ZERO);
    }

    #[test]
    fn test_vesting_window_exactly_matches_epoch() {
        let row = cu_row(("miner-1", 11, "node-1", "8", ts(2), "H100", 1, "10"));
        let fraction = compute_cu_vested_fraction(&row, ts(2), ts(3));
        assert_eq!(fraction, d("1"));
    }

    #[test]
    fn test_vesting_straddles_epoch_start() {
        // CU window [T0, T4], epoch [T3, T5] → overlap = 4 - 3 = 1h
        let row = cu_row(("miner-1", 11, "node-1", "8", ts(0), "H100", 4, "10"));
        let fraction = compute_cu_vested_fraction(&row, ts(3), ts(5));
        assert_eq!(fraction, d("0.25"));
    }

    #[test]
    fn test_vesting_straddles_epoch_end() {
        // CU window [T3, T7], epoch [T2, T4] → overlap = 4 - 3 = 1h
        let row = cu_row(("miner-1", 11, "node-1", "8", ts(3), "H100", 4, "10"));
        let fraction = compute_cu_vested_fraction(&row, ts(2), ts(4));
        assert_eq!(fraction, d("0.25"));
    }

    #[test]
    fn test_vesting_window_hours_zero() {
        let row = cu_row(("miner-1", 11, "node-1", "8", ts(0), "H100", 0, "10"));
        let fraction = compute_cu_vested_fraction(&row, ts(0), ts(4));
        assert_eq!(fraction, Decimal::ZERO);
    }

    #[test]
    fn test_vesting_zero_length_epoch() {
        let row = cu_row(("miner-1", 11, "node-1", "8", ts(0), "H100", 4, "10"));
        let fraction = compute_cu_vested_fraction(&row, ts(2), ts(2));
        assert_eq!(fraction, Decimal::ZERO);
    }

    #[test]
    fn test_vesting_ru_uses_same_formula() {
        // Full overlap
        let ru = ru_row(("miner-1", 11, "node-1", "100", ts(0), "H100", 4, 30));
        assert_eq!(compute_ru_vested_fraction(&ru, ts(0), ts(4)), d("1"));

        // Partial overlap from start
        assert_eq!(compute_ru_vested_fraction(&ru, ts(0), ts(2)), d("0.5"));

        // Partial overlap from middle
        assert_eq!(compute_ru_vested_fraction(&ru, ts(2), ts(4)), d("0.5"));

        // No overlap (before epoch)
        let ru_before = ru_row(("miner-1", 11, "node-1", "100", ts(0), "H100", 24, 30));
        assert_eq!(
            compute_ru_vested_fraction(&ru_before, ts(50), ts(51)),
            Decimal::ZERO
        );

        // Straddle epoch start: window [T0,T4], epoch [T3,T5] → 1/4
        assert_eq!(compute_ru_vested_fraction(&ru, ts(3), ts(5)), d("0.25"));

        // Zero window hours
        let ru_zero = ru_row(("miner-1", 11, "node-1", "100", ts(0), "H100", 0, 30));
        assert_eq!(
            compute_ru_vested_fraction(&ru_zero, ts(0), ts(4)),
            Decimal::ZERO
        );

        // Zero length epoch
        assert_eq!(compute_ru_vested_fraction(&ru, ts(2), ts(2)), Decimal::ZERO);
    }

    // ==================== Category 2: Dilution Mechanics ====================

    #[test]
    fn test_dilution_under_provisioned_capped_at_row_price() {
        // Config: H100 target_count=4 (→ 32 GPUs), max_cu_value_usd=$100
        let config = make_config(&[("H100", 4, "10")], 4, "100", None);
        // 1 miner: cu_amount=8, window_hours=4, price_usd=$10
        let cu_rows = vec![cu_row((
            "miner-0",
            0,
            "node-0",
            "8",
            ts(0),
            "H100",
            4,
            "10",
        ))];
        let result = compute_incentive_pool(
            &config,
            &cu_rows,
            &[],
            ts(0),
            ts(4),
            d("100000"),
            999,
            &make_hotkey_map(1),
            None,
        )
        .unwrap();
        // capacity_budget = 32 × 4 × $10 = $1280, supply = 8
        // per_cu_budget = $1280/8 = $160, effective = MIN($10, $160, $100) = $10
        // payout = 1.0 × 8 × $10 = $80
        assert_eq!(miner_payout(&result, "miner-0"), d("80"));
    }

    #[test]
    fn test_dilution_over_provisioned_4x() {
        // Config: H100 target_count=1 (→ 8 GPUs)
        let config = make_config(&[("H100", 1, "10")], 4, "100", None);
        // 4 miners: each cu_amount=8, window_hours=4, price_usd=$10 → category_supply = 32
        let cu_rows: Vec<CuLedgerRowResponse> = (0..4)
            .map(|i| {
                let hotkey = format!("miner-{i}");
                let node_id = format!("node-{i}");
                cu_row((&hotkey, i as u32, &node_id, "8", ts(0), "H100", 4, "10"))
            })
            .collect();
        let result = compute_incentive_pool(
            &config,
            &cu_rows,
            &[],
            ts(0),
            ts(4),
            d("100000"),
            999,
            &make_hotkey_map(4),
            None,
        )
        .unwrap();
        // capacity_budget = 8 × 4 × $10 = $320, per_cu_budget = $320/32 = $10
        // effective = MIN($10, $10, $100) = $10 → no actual dilution
        // Each miner payout = 1.0 × 8 × $10 = $80
        for i in 0..4 {
            assert_eq!(miner_payout(&result, &format!("miner-{i}")), d("80"));
        }
    }

    #[test]
    fn test_dilution_over_provisioned_8x() {
        // Config: H100 target_count=1 (→ 8 GPUs)
        let config = make_config(&[("H100", 1, "10")], 4, "100", None);
        // 8 miners: each cu_amount=8 → category_supply = 64
        let cu_rows: Vec<CuLedgerRowResponse> = (0..8)
            .map(|i| {
                let hotkey = format!("miner-{i}");
                let node_id = format!("node-{i}");
                cu_row((&hotkey, i as u32, &node_id, "8", ts(0), "H100", 4, "10"))
            })
            .collect();
        let result = compute_incentive_pool(
            &config,
            &cu_rows,
            &[],
            ts(0),
            ts(4),
            d("100000"),
            999,
            &make_hotkey_map(8),
            None,
        )
        .unwrap();
        // per_cu_budget = $320/64 = $5, effective = MIN($10, $5, $100) = $5 (diluted)
        // Each miner payout = 1.0 × 8 × $5 = $40
        for i in 0..8 {
            assert_eq!(miner_payout(&result, &format!("miner-{i}")), d("40"));
        }
    }

    #[test]
    fn test_max_cu_value_usd_caps_effective_price() {
        // Config: H100 target_count=4 (32 GPUs), max_cu_value_usd=$2
        let config = make_config(&[("H100", 4, "10")], 4, "2", None);
        // 1 miner: cu_amount=8, price_usd=$10
        let cu_rows = vec![cu_row((
            "miner-0",
            0,
            "node-0",
            "8",
            ts(0),
            "H100",
            4,
            "10",
        ))];
        let result = compute_incentive_pool(
            &config,
            &cu_rows,
            &[],
            ts(0),
            ts(4),
            d("100000"),
            999,
            &make_hotkey_map(1),
            None,
        )
        .unwrap();
        // per_cu_budget = $1280/8 = $160, effective = MIN($10, $160, $2) = $2
        // payout = 1.0 × 8 × $2 = $16
        assert_eq!(miner_payout(&result, "miner-0"), d("16"));
    }

    #[test]
    fn test_cross_category_dilution_independence() {
        // Config: H100 target_count=1, A100 target_count=2
        let config = make_config(&[("H100", 1, "10"), ("A100", 2, "8")], 4, "100", None);
        // H100: 8 miners × cu_amount=8 → supply=64, heavily overprovisioned
        let mut cu_rows: Vec<CuLedgerRowResponse> = (0..8)
            .map(|i| {
                let hotkey = format!("miner-{i}");
                let node_id = format!("node-{i}");
                cu_row((&hotkey, i as u32, &node_id, "8", ts(0), "H100", 4, "10"))
            })
            .collect();
        // A100: 1 miner × cu_amount=16 → supply=16, exactly provisioned
        cu_rows.push(cu_row((
            "miner-8",
            8,
            "node-8",
            "16",
            ts(0),
            "A100",
            4,
            "8",
        )));
        let result = compute_incentive_pool(
            &config,
            &cu_rows,
            &[],
            ts(0),
            ts(4),
            d("100000"),
            999,
            &make_hotkey_map(9),
            None,
        )
        .unwrap();
        // H100: per_cu_budget = (8×4×$10)/64 = $5, effective = $5 (diluted)
        // Each H100 miner payout = 8 × $5 = $40
        for i in 0..8 {
            assert_eq!(miner_payout(&result, &format!("miner-{i}")), d("40"));
        }
        // A100: per_cu_budget = (16×4×$8)/16 = $32, effective = MIN($8, $32, $100) = $8
        // A100 miner payout = 16 × $8 = $128
        assert_eq!(miner_payout(&result, "miner-8"), d("128"));
    }

    // ==================== Category 3: RU Payout Correctness ====================

    #[test]
    fn test_ru_exact_payout_value() {
        // RU: ru_amount=$100, revenue_share_pct=30, window_hours=4
        // Epoch [T0,T4] → vested_fraction=1.0
        // Expected payout = 1.0 × $100 × 30/100 = $30
        let config = make_config(&[("H100", 1, "10")], 4, "100", None);
        let result = compute_incentive_pool(
            &config,
            &[],
            &[ru_row((
                "miner-0",
                0,
                "node-0",
                "100",
                ts(0),
                "H100",
                4,
                30,
            ))],
            ts(0),
            ts(4),
            d("100000"),
            999,
            &make_hotkey_map(1),
            None,
        )
        .unwrap();
        assert_eq!(miner_payout(&result, "miner-0"), d("30"));
    }

    #[test]
    fn test_ru_different_revenue_share_pct_per_row() {
        // Miner-0: two RU rows with different revenue_share_pct
        // Row A: ru_amount=$100, revenue_share_pct=30 → contributes $30
        // Row B: ru_amount=$100, revenue_share_pct=50 → contributes $50
        // Expected total payout = $80
        let config = make_config(&[("H100", 1, "10")], 4, "100", None);
        let result = compute_incentive_pool(
            &config,
            &[],
            &[
                ru_row(("miner-0", 0, "node-0", "100", ts(0), "H100", 4, 30)),
                ru_row(("miner-0", 0, "node-1", "100", ts(0), "H100", 4, 50)),
            ],
            ts(0),
            ts(4),
            d("100000"),
            999,
            &make_hotkey_map(1),
            None,
        )
        .unwrap();
        assert_eq!(miner_payout(&result, "miner-0"), d("80"));
    }

    #[test]
    fn test_ru_zero_revenue_share_pct() {
        // RU: ru_amount=$100, revenue_share_pct=0
        // Expected payout = $0 → miner should get zero weight
        let config = make_config(&[("H100", 1, "10")], 4, "100", None);
        let result = compute_incentive_pool(
            &config,
            &[],
            &[ru_row(("miner-0", 0, "node-0", "100", ts(0), "H100", 4, 0))],
            ts(0),
            ts(4),
            d("100000"),
            999,
            &make_hotkey_map(1),
            None,
        )
        .unwrap();
        assert_eq!(miner_payout(&result, "miner-0"), Decimal::ZERO);
        assert_eq!(miner_weight(&result, 0), 0);
    }

    #[test]
    fn test_ru_100_percent_revenue_share() {
        // RU: ru_amount=$100, revenue_share_pct=100
        // Expected payout = $100
        let config = make_config(&[("H100", 1, "10")], 4, "100", None);
        let result = compute_incentive_pool(
            &config,
            &[],
            &[ru_row((
                "miner-0",
                0,
                "node-0",
                "100",
                ts(0),
                "H100",
                4,
                100,
            ))],
            ts(0),
            ts(4),
            d("100000"),
            999,
            &make_hotkey_map(1),
            None,
        )
        .unwrap();
        assert_eq!(miner_payout(&result, "miner-0"), d("100"));
    }

    #[test]
    fn test_ru_no_category_dilution() {
        // RU rows are NOT subject to per-category dilution.
        // Even if the GPU category is overprovisioned, RU payout = vested × amount × share%
        let config = make_config(&[("H100", 1, "10")], 4, "100", None);

        // Run 1: RU row alone (no CU oversupply)
        let result_alone = compute_incentive_pool(
            &config,
            &[],
            &[ru_row((
                "miner-0",
                0,
                "node-0",
                "100",
                ts(0),
                "H100",
                4,
                30,
            ))],
            ts(0),
            ts(4),
            d("100000"),
            999,
            &make_hotkey_map(10),
            None,
        )
        .unwrap();

        // Run 2: RU row alongside heavy CU oversupply in the same category
        let cu_rows: Vec<CuLedgerRowResponse> = (1..10)
            .map(|i| {
                let hotkey = format!("miner-{i}");
                let node_id = format!("node-{i}");
                cu_row((&hotkey, i as u32, &node_id, "8", ts(0), "H100", 4, "10"))
            })
            .collect();
        let result_with_oversupply = compute_incentive_pool(
            &config,
            &cu_rows,
            &[ru_row((
                "miner-0",
                0,
                "node-0",
                "100",
                ts(0),
                "H100",
                4,
                30,
            ))],
            ts(0),
            ts(4),
            d("100000"),
            999,
            &make_hotkey_map(10),
            None,
        )
        .unwrap();

        // RU payout should be identical regardless of CU oversupply
        assert_eq!(
            miner_payout(&result_alone, "miner-0"),
            miner_payout(&result_with_oversupply, "miner-0")
        );
        assert_eq!(miner_payout(&result_alone, "miner-0"), d("30"));
    }

    #[test]
    fn test_dilution_uses_row_snapshot_values() {
        // Config: H100 target_count=1 (8 GPUs), config window_hours=4, config price=$10
        // These config values should NOT be used for capacity_budget
        let config = make_config(&[("H100", 1, "10")], 4, "100", None);
        // CU row with DIFFERENT values: window_hours=8, price_usd=$5
        let cu_rows = vec![cu_row(("miner-0", 0, "node-0", "8", ts(0), "H100", 8, "5"))];
        let result = compute_incentive_pool(
            &config,
            &cu_rows,
            &[],
            ts(0),
            ts(8), // epoch covers full 8h window
            d("100000"),
            999,
            &make_hotkey_map(1),
            None,
        )
        .unwrap();
        // capacity_budget uses ROW values: 8 × 8 × $5 = $320 (not 8 × 4 × $10)
        // supply = 8, per_cu_budget = $320/8 = $40
        // effective = MIN($5, $40, $100) = $5
        // payout = 1.0 × 8 × $5 = $40
        assert_eq!(miner_payout(&result, "miner-0"), d("40"));
    }

    // ==================== Category 4: Combined CU + RU ====================

    #[test]
    fn test_cu_and_ru_same_miner_additive() {
        // Miner-0: CU payout = $80 (8 CU × $10), RU payout = $30 (100 × 30%)
        // Expected total = $110
        let config = make_config(&[("H100", 1, "10")], 4, "100", None);
        let result = compute_incentive_pool(
            &config,
            &[cu_row((
                "miner-0",
                0,
                "node-0",
                "8",
                ts(0),
                "H100",
                4,
                "10",
            ))],
            &[ru_row((
                "miner-0",
                0,
                "node-0",
                "100",
                ts(0),
                "H100",
                4,
                30,
            ))],
            ts(0),
            ts(4),
            d("100000"),
            999,
            &make_hotkey_map(1),
            None,
        )
        .unwrap();
        // CU: target_gpus=8, budget=8×4×$10=$320, supply=8, per_cu=$40
        //     effective=MIN($10,$40,$100)=$10, payout=1.0×8×$10=$80
        // RU: payout=1.0×$100×30/100=$30
        // Total = $80 + $30 = $110
        assert_eq!(miner_payout(&result, "miner-0"), d("110"));
    }

    #[test]
    fn test_cu_only_and_ru_only_miners_weight_proportional() {
        // Miner-0: CU only → payout $80
        // Miner-1: RU only → payout $30
        // Total = $110, capacity = $1000 (no scaling)
        let config = make_config(&[("H100", 1, "10")], 4, "100", None);
        let result = compute_incentive_pool(
            &config,
            &[cu_row((
                "miner-0",
                0,
                "node-0",
                "8",
                ts(0),
                "H100",
                4,
                "10",
            ))],
            &[ru_row((
                "miner-1",
                1,
                "node-1",
                "100",
                ts(0),
                "H100",
                4,
                30,
            ))],
            ts(0),
            ts(4),
            d("1000"),
            999,
            &make_hotkey_map(2),
            None,
        )
        .unwrap();
        // Verify payouts
        assert_eq!(miner_payout(&result, "miner-0"), d("80"));
        assert_eq!(miner_payout(&result, "miner-1"), d("30"));
        // Weight ratio should be ≈ 80/30 ≈ 2.667
        let w0 = miner_weight(&result, 0) as f64;
        let w1 = miner_weight(&result, 1) as f64;
        assert!(w0 > 0.0);
        assert!(w1 > 0.0);
        let ratio = w0 / w1;
        let expected_ratio = 80.0 / 30.0;
        assert!(
            (ratio - expected_ratio).abs() < 0.05,
            "Weight ratio {ratio} should be close to {expected_ratio}"
        );
    }

    // ==================== Category 5: Multi-Epoch Vesting Simulation ====================

    #[test]
    fn test_multi_epoch_vesting_sums_to_full_cu_payout() {
        // 1 CU row, cu_amount=8, window_hours=4, price_usd=$10, earned_at=T0
        // Config: H100 target_count=1 → supply=8, budget=8×4×$10=$320, per_cu=$40
        // effective = MIN($10, $40, $100) = $10
        // 4 consecutive 1h epochs: each vested_fraction=1/4, payout=$20
        // Sum = 4 × $20 = $80 = full CU value
        let config = make_config(&[("H100", 1, "10")], 4, "100", None);
        let cu_rows = vec![cu_row((
            "miner-0",
            0,
            "node-0",
            "8",
            ts(0),
            "H100",
            4,
            "10",
        ))];
        let hotkeys = make_hotkey_map(1);

        let mut total_payout = Decimal::ZERO;
        for epoch_hour in 0..4 {
            let result = compute_incentive_pool(
                &config,
                &cu_rows,
                &[],
                ts(epoch_hour),
                ts(epoch_hour + 1),
                d("10000"),
                999,
                &hotkeys,
                None,
            )
            .unwrap();
            total_payout += miner_payout(&result, "miner-0");
        }
        assert_eq!(total_payout, d("80"));
    }

    #[test]
    fn test_multi_epoch_vesting_many_small_epochs() {
        // Divide a 4h vesting window into 24 × 10-minute epochs
        // Each epoch: vested_fraction = 10min / 4h = 1/24
        // Sum of all fractions should = 1.0 exactly
        let row = cu_row(("miner-0", 0, "node-0", "8", ts(0), "H100", 4, "10"));

        let mut total_fraction = Decimal::ZERO;
        for i in 0..24 {
            let epoch_start = ts_minutes(i * 10);
            let epoch_end = ts_minutes((i + 1) * 10);
            total_fraction += compute_cu_vested_fraction(&row, epoch_start, epoch_end);
        }
        assert_eq!(total_fraction.round_dp(20), d("1"));
    }

    #[test]
    fn test_multi_epoch_vesting_staggered_miners() {
        // Miner-0: CU earned at T0, window=4h, cu_amount=8 → full value $80
        // Miner-1: CU earned at T1, window=4h, cu_amount=8 → full value $80
        // Both rows always present → category_supply = 16
        // capacity_budget = 8 × 4 × $10 = $320, per_cu = $320/16 = $20
        // effective = MIN($10, $20, $100) = $10 (no dilution effect on price)
        //
        // Epochs:
        //   [T0,T1]: Miner-0 vests 1/4=$20, Miner-1 vests 0
        //   [T1,T2]: Both vest 1/4, each $20
        //   [T2,T3]: Both vest 1/4, each $20
        //   [T3,T4]: Both vest 1/4, each $20
        //   [T4,T5]: Miner-0 vests 0, Miner-1 vests 1/4=$20
        // Miner-0 total = $80, Miner-1 total = $80
        let config = make_config(&[("H100", 1, "10")], 4, "100", None);
        let cu_rows = vec![
            cu_row(("miner-0", 0, "node-0", "8", ts(0), "H100", 4, "10")),
            cu_row(("miner-1", 1, "node-1", "8", ts(1), "H100", 4, "10")),
        ];
        let hotkeys = make_hotkey_map(2);

        let mut total_miner0 = Decimal::ZERO;
        let mut total_miner1 = Decimal::ZERO;
        for epoch_hour in 0..5 {
            let result = compute_incentive_pool(
                &config,
                &cu_rows,
                &[],
                ts(epoch_hour),
                ts(epoch_hour + 1),
                d("10000"),
                999,
                &hotkeys,
                None,
            )
            .unwrap();
            total_miner0 += miner_payout(&result, "miner-0");
            total_miner1 += miner_payout(&result, "miner-1");
        }
        assert_eq!(total_miner0, d("80"));
        assert_eq!(total_miner1, d("80"));
    }

    #[test]
    fn test_multi_epoch_ru_vesting_sums_to_full_value() {
        // RU: ru_amount=$100, revenue_share_pct=30, window_hours=4
        // Full value = $100 × 30% = $30
        // 4 consecutive 1h epochs: each vested_fraction=1/4
        // Each payout = 0.25 × $100 × 30/100 = $7.5
        // Sum = 4 × $7.5 = $30
        let config = make_config(&[("H100", 1, "10")], 4, "100", None);
        let ru_rows = vec![ru_row((
            "miner-0",
            0,
            "node-0",
            "100",
            ts(0),
            "H100",
            4,
            30,
        ))];
        let hotkeys = make_hotkey_map(1);

        let mut total_payout = Decimal::ZERO;
        for epoch_hour in 0..4 {
            let result = compute_incentive_pool(
                &config,
                &[],
                &ru_rows,
                ts(epoch_hour),
                ts(epoch_hour + 1),
                d("10000"),
                999,
                &hotkeys,
                None,
            )
            .unwrap();
            total_payout += miner_payout(&result, "miner-0");
        }
        assert_eq!(total_payout, d("30"));
    }

    // ==================== Category 6: Emission Cap & Scaling ====================

    #[test]
    fn test_emission_cap_proportional_scaling_preserves_ratios() {
        // 3 miners with CU payouts in ratio 1:2:3 (cu_amounts 2, 4, 6)
        // Config: H100 target_count=1 (8 GPUs), max_cu=$100
        // supply = 2+4+6 = 12, budget = 8×4×$10 = $320, per_cu = $320/12 ≈ $26.67
        // effective = MIN($10, $26.67, $100) = $10 (no dilution on price)
        // Raw payouts: miner-0=2×$10=$20, miner-1=4×$10=$40, miner-2=6×$10=$60
        // Total raw = $120
        // capacity = $60, scale_factor = $60/$120 = 0.5
        // Scaled: $10, $20, $30 (ratio 1:2:3 preserved)
        let config = make_config(&[("H100", 1, "10")], 4, "100", None);
        let cu_rows = vec![
            cu_row(("miner-0", 0, "node-0", "2", ts(0), "H100", 4, "10")),
            cu_row(("miner-1", 1, "node-1", "4", ts(0), "H100", 4, "10")),
            cu_row(("miner-2", 2, "node-2", "6", ts(0), "H100", 4, "10")),
        ];
        let result = compute_incentive_pool(
            &config,
            &cu_rows,
            &[],
            ts(0),
            ts(4),
            d("60"), // capacity well below total raw payouts of $120
            999,
            &make_hotkey_map(3),
            None,
        )
        .unwrap();

        // Verify scaled payouts preserve 1:2:3 ratio
        assert_eq!(miner_payout(&result, "miner-0"), d("10"));
        assert_eq!(miner_payout(&result, "miner-1"), d("20"));
        assert_eq!(miner_payout(&result, "miner-2"), d("30"));

        // Verify weight ratios ≈ 1:2:3
        let w0 = miner_weight(&result, 0) as f64;
        let w1 = miner_weight(&result, 1) as f64;
        let w2 = miner_weight(&result, 2) as f64;
        assert!(w0 > 0.0);
        assert!((w1 / w0 - 2.0).abs() < 0.01, "w1/w0 ratio should be ~2.0");
        assert!((w2 / w0 - 3.0).abs() < 0.01, "w2/w0 ratio should be ~3.0");
    }

    #[test]
    fn test_emission_cap_exactly_at_capacity() {
        // 1 miner: cu_amount=8, price=$10
        // supply=8, target_gpus=8, budget=8×4×$10=$320, per_cu=$40
        // effective = MIN($10, $40, $100) = $10
        // payout = 8 × $10 = $80
        // capacity = $80 (exactly equal)
        let config = make_config(&[("H100", 1, "10")], 4, "100", None);
        let cu_rows = vec![cu_row((
            "miner-0",
            0,
            "node-0",
            "8",
            ts(0),
            "H100",
            4,
            "10",
        ))];
        let result = compute_incentive_pool(
            &config,
            &cu_rows,
            &[],
            ts(0),
            ts(4),
            d("80"), // exactly equals total payouts
            999,
            &make_hotkey_map(1),
            None,
        )
        .unwrap();

        // scale_factor = 1.0 (not over capacity), payout unchanged
        assert_eq!(miner_payout(&result, "miner-0"), d("80"));
        // burn_rate = 1 - ($80/$80) = 0
        assert_eq!(result.burn_rate, Decimal::ZERO);
        // burn_allocation weight should be 0
        assert_eq!(
            result
                .distribution
                .burn_allocation
                .as_ref()
                .map(|a| a.weight)
                .unwrap_or(0),
            0
        );
    }

    #[test]
    fn test_emission_cap_just_barely_exceeds() {
        // 1 miner: cu_amount="10.001", price=$10
        // supply=10.001, target_gpus=16, budget=16×4×$10=$640
        // per_cu=$640/10.001≈$63.99, effective=MIN($10,$63.99,$100)=$10
        // payout = 10.001 × $10 = $100.01
        // capacity = $100
        let config = make_config(&[("H100", 2, "10")], 4, "100", None);
        let cu_rows = vec![cu_row((
            "miner-0",
            0,
            "node-0",
            "10.001",
            ts(0),
            "H100",
            4,
            "10",
        ))];
        let result = compute_incentive_pool(
            &config,
            &cu_rows,
            &[],
            ts(0),
            ts(4),
            d("100"),
            999,
            &make_hotkey_map(1),
            None,
        )
        .unwrap();

        // scale_factor < 1.0 was applied → payout should be less than raw $100.01
        assert!(
            miner_payout(&result, "miner-0") < d("100.01"),
            "Scaling should have reduced the payout"
        );
        // usd_required_epoch ≤ capacity
        assert!(
            result.usd_required_epoch <= d("100"),
            "Scaled total should not exceed emission capacity"
        );
        // burn_rate should be 0 or very near 0 (all capacity consumed)
        assert!(
            result.burn_rate >= Decimal::ZERO,
            "burn_rate must be non-negative"
        );
    }

    #[test]
    fn test_emission_cap_far_exceeds() {
        // 1 miner: cu_amount=100, price=$10
        // Config: H100 target_count=100 (800 GPUs)
        // supply=100, budget=800×4×$10=$32000, per_cu=$320
        // effective = MIN($10, $320, $100) = $10
        // payout = 100 × $10 = $1000
        // capacity = $10
        // scale_factor = $10/$1000 = 0.01
        let config = make_config(&[("H100", 100, "10")], 4, "100", None);
        let cu_rows = vec![cu_row((
            "miner-0",
            0,
            "node-0",
            "100",
            ts(0),
            "H100",
            4,
            "10",
        ))];
        let result = compute_incentive_pool(
            &config,
            &cu_rows,
            &[],
            ts(0),
            ts(4),
            d("10"), // far below total payouts of $1000
            999,
            &make_hotkey_map(1),
            None,
        )
        .unwrap();

        // Payout reduced 100x: $1000 × 0.01 = $10
        assert_eq!(miner_payout(&result, "miner-0"), d("10"));
        // burn_rate = 0 (demand exceeds capacity, no excess to burn)
        assert_eq!(result.burn_rate, Decimal::ZERO);
        // All weight goes to miner (no burn weight since burn_rate=0)
        assert_eq!(
            result
                .distribution
                .burn_allocation
                .as_ref()
                .map(|a| a.weight)
                .unwrap_or(0),
            0
        );
    }

    // ── Category 7: Dynamic Burn ──────────────────────────────────────

    #[test]
    fn test_burn_rate_exact_value() {
        // 1 miner, cu_amount=2, price=$10
        // supply=2, target_gpus=8, budget=8×4×$10=$320, per_cu=$160
        // effective_price = MIN($10, $160, $100) = $10
        // payout = 1.0 × 2 × $10 = $20, capacity = $100
        // burn_rate = 1 - ($20/$100) = 0.80
        let config = make_config(&[("H100", 1, "10")], 4, "100", None);
        let cu_rows = vec![cu_row((
            "miner-0",
            0,
            "node-0",
            "2",
            ts(0),
            "H100",
            4,
            "10",
        ))];
        let result = compute_incentive_pool(
            &config,
            &cu_rows,
            &[],
            ts(0),
            ts(4),
            d("100"),
            999,
            &make_hotkey_map(1),
            None,
        )
        .unwrap();

        assert_eq!(miner_payout(&result, "miner-0"), d("20"));
        assert_eq!(result.burn_rate, d("0.8"));
    }

    #[test]
    fn test_burn_rate_zero_when_at_capacity() {
        // 1 miner, cu_amount=8, price=$10
        // supply=8, target_gpus=8, budget=8×4×$10=$320, per_cu=$40
        // effective_price = MIN($10, $40, $100) = $10
        // payout = 8 × $10 = $80, capacity = $80
        // burn_rate = 1 - ($80/$80) = 0
        let config = make_config(&[("H100", 1, "10")], 4, "100", None);
        let cu_rows = vec![cu_row((
            "miner-0",
            0,
            "node-0",
            "8",
            ts(0),
            "H100",
            4,
            "10",
        ))];
        let result = compute_incentive_pool(
            &config,
            &cu_rows,
            &[],
            ts(0),
            ts(4),
            d("80"),
            999,
            &make_hotkey_map(1),
            None,
        )
        .unwrap();

        assert_eq!(miner_payout(&result, "miner-0"), d("80"));
        assert_eq!(result.burn_rate, Decimal::ZERO);
    }

    #[test]
    fn test_burn_rate_zero_when_over_capacity() {
        // 1 miner, cu_amount=8, price=$10 → raw payout=$80
        // capacity=$40 → scale_factor = 40/80 = 0.5
        // scaled payout = $40
        // burn_rate = 1 - ($40/$40) = 0
        let config = make_config(&[("H100", 1, "10")], 4, "100", None);
        let cu_rows = vec![cu_row((
            "miner-0",
            0,
            "node-0",
            "8",
            ts(0),
            "H100",
            4,
            "10",
        ))];
        let result = compute_incentive_pool(
            &config,
            &cu_rows,
            &[],
            ts(0),
            ts(4),
            d("40"),
            999,
            &make_hotkey_map(1),
            None,
        )
        .unwrap();

        assert_eq!(miner_payout(&result, "miner-0"), d("40"));
        assert_eq!(result.burn_rate, Decimal::ZERO);
    }

    #[test]
    fn test_forced_burn_percentage() {
        // forced_burn_percentage = 20%
        // effective_capacity = 100 × (1 - 0.20) ≈ $80
        // forced_weight = round(0.20 × 65535) = 13107
        // available_weight = 65535 - 13107 = 52428
        //
        // 1 miner, payout=$20 < effective_capacity → no scaling
        // Dynamic burn: burn_share ≈ 1 - (20/80) = 0.75
        // Miner weight in pool ≈ 0.25 × 52428 ≈ 13107
        // Dynamic burn weight ≈ 0.75 × 52428 ≈ 39321
        // Total burn = 39321 + 13107 = 52428
        let config = make_config(&[("H100", 1, "10")], 4, "100", None);
        let cu_rows = vec![cu_row((
            "miner-0",
            0,
            "node-0",
            "2",
            ts(0),
            "H100",
            4,
            "10",
        ))];
        let result = compute_incentive_pool(
            &config,
            &cu_rows,
            &[],
            ts(0),
            ts(4),
            d("100"),
            999,
            &make_hotkey_map(1),
            Some(20.0),
        )
        .unwrap();

        // Miner payout unchanged (no scaling needed)
        assert_eq!(miner_payout(&result, "miner-0"), d("20"));

        // Total weights must sum to 65535
        let total_weight: u64 = result
            .distribution
            .weights
            .iter()
            .map(|w| w.weight as u64)
            .sum();
        assert_eq!(total_weight, u16::MAX as u64);

        // Compute expected forced_weight using same formula as production code
        let forced_pct = Decimal::from_f64_retain(20.0 / 100.0).unwrap();
        let expected_forced_weight = (forced_pct * Decimal::from(u16::MAX))
            .round()
            .to_u16()
            .unwrap();

        // Burn weight includes forced component + dynamic burn
        let burn_w = miner_weight(&result, 999);
        assert!(
            burn_w >= expected_forced_weight,
            "burn weight ({burn_w}) should include forced component ({expected_forced_weight})"
        );
        assert!(
            burn_w > expected_forced_weight,
            "burn weight ({burn_w}) should exceed forced component due to dynamic burn"
        );

        // Miner + burn = 65535
        let miner_w = miner_weight(&result, 0);
        assert_eq!(
            miner_w as u32 + burn_w as u32,
            u16::MAX as u32,
            "miner ({miner_w}) + burn ({burn_w}) should equal 65535"
        );
    }

    #[test]
    fn test_forced_burn_combined_with_emission_cap() {
        // forced_burn_percentage = 50% → effective_capacity = 50% of $100 = $50
        // 0.5 is exactly representable in f64, so exact assertions are safe
        // forced_weight = round(0.5 × 65535) = 32768
        // available_weight = 65535 - 32768 = 32767
        //
        // 1 miner, raw payout=$80 > $50 → scale_factor = 50/80 = 0.625
        // scaled payout = $50 (fills effective capacity exactly)
        // No dynamic burn (demand = capacity)
        // Miner gets all available_weight = 32767
        let config = make_config(&[("H100", 1, "10")], 4, "100", None);
        let cu_rows = vec![cu_row((
            "miner-0",
            0,
            "node-0",
            "8",
            ts(0),
            "H100",
            4,
            "10",
        ))];
        let result = compute_incentive_pool(
            &config,
            &cu_rows,
            &[],
            ts(0),
            ts(4),
            d("100"),
            999,
            &make_hotkey_map(1),
            Some(50.0),
        )
        .unwrap();

        // Miner payout scaled to effective capacity
        assert_eq!(miner_payout(&result, "miner-0"), d("50"));

        // Total weights = 65535
        let total_weight: u64 = result
            .distribution
            .weights
            .iter()
            .map(|w| w.weight as u64)
            .sum();
        assert_eq!(total_weight, u16::MAX as u64);

        // Exact weight split: miner gets available_weight, burn gets forced_weight
        let miner_w = miner_weight(&result, 0);
        let burn_w = miner_weight(&result, 999);
        assert_eq!(miner_w, 32767);
        assert_eq!(burn_w, 32768);
    }

    #[test]
    fn test_forced_burn_near_100_percent() {
        // forced_burn_percentage = 99%
        // effective_capacity ≈ 1% of $100 ≈ $1
        // forced_weight ≈ round(0.99 × 65535) ≈ 64880
        // available_weight ≈ 655
        //
        // 1 miner, raw payout=$20 >> $1 → heavy scaling
        // Payout scaled to ~$1, miner gets all ~655 available weight
        let config = make_config(&[("H100", 1, "10")], 4, "100", None);
        let cu_rows = vec![cu_row((
            "miner-0",
            0,
            "node-0",
            "2",
            ts(0),
            "H100",
            4,
            "10",
        ))];
        let result = compute_incentive_pool(
            &config,
            &cu_rows,
            &[],
            ts(0),
            ts(4),
            d("100"),
            999,
            &make_hotkey_map(1),
            Some(99.0),
        )
        .unwrap();

        // Total weights = 65535
        let total_weight: u64 = result
            .distribution
            .weights
            .iter()
            .map(|w| w.weight as u64)
            .sum();
        assert_eq!(total_weight, u16::MAX as u64);

        // Almost all weight to burn
        let burn_w = miner_weight(&result, 999);
        let miner_w = miner_weight(&result, 0);
        assert!(
            burn_w > 64000,
            "burn should have almost all weight, got {burn_w}"
        );
        assert!(miner_w > 0, "miner should have nonzero weight");
        assert!(
            miner_w <= 700,
            "miner weight ({miner_w}) should be within available pool (~655)"
        );

        // Miner payout heavily reduced
        let payout = miner_payout(&result, "miner-0");
        assert!(
            payout < d("2"),
            "payout ({payout}) should be heavily reduced from $20"
        );
        assert!(payout > Decimal::ZERO, "payout should be nonzero");
    }

    // ==================== Category 8: Filtering & Exclusion ====================

    #[test]
    fn test_slashed_cu_excluded_from_payout_and_dilution() {
        // 2 CU rows same category: row A (not slashed), row B (slashed)
        // Row B should NOT appear in miner_payouts
        // Row B should NOT contribute to category_cu_supply (dilution denominator)
        // Row A gets full undiluted payout as if row B doesn't exist
        let config = make_config(&[("H100", 1, "10")], 4, "100", None);

        // Row A: normal
        let row_a = cu_row(("miner-0", 0, "node-0", "8", ts(0), "H100", 4, "10"));
        // Row B: slashed
        let mut row_b = cu_row(("miner-1", 1, "node-1", "8", ts(0), "H100", 4, "10"));
        row_b.is_slashed = true;

        let result = compute_incentive_pool(
            &config,
            &[row_a, row_b],
            &[],
            ts(0),
            ts(4),
            d("100000"),
            999,
            &make_hotkey_map(2),
            None,
        )
        .unwrap();

        // Row B (slashed) should have zero payout
        assert_eq!(miner_payout(&result, "miner-1"), Decimal::ZERO);

        // Row A should get full undiluted payout:
        // category_supply = 8 (only row A counted)
        // budget = 8 × 4 × $10 = $320, per_cu = $320/8 = $40
        // effective = MIN($10, $40, $100) = $10
        // payout = 1.0 × 8 × $10 = $80
        assert_eq!(miner_payout(&result, "miner-0"), d("80"));
    }

    #[test]
    fn test_slashed_ru_excluded() {
        // Same pattern for RU rows
        let config = make_config(&[("H100", 1, "10")], 4, "100", None);

        // Row A: normal
        let row_a = ru_row(("miner-0", 0, "node-0", "100", ts(0), "H100", 4, 30));
        // Row B: slashed
        let mut row_b = ru_row(("miner-1", 1, "node-1", "100", ts(0), "H100", 4, 30));
        row_b.is_slashed = true;

        let result = compute_incentive_pool(
            &config,
            &[],
            &[row_a, row_b],
            ts(0),
            ts(4),
            d("100000"),
            999,
            &make_hotkey_map(2),
            None,
        )
        .unwrap();

        // Slashed RU row should produce zero payout
        assert_eq!(miner_payout(&result, "miner-1"), Decimal::ZERO);
        // Normal RU row: payout = 1.0 × $100 × 30/100 = $30
        assert_eq!(miner_payout(&result, "miner-0"), d("30"));
    }

    #[test]
    fn test_unknown_hotkey_excluded() {
        // CU row with hotkey not in hotkey_to_uid map
        // Should be completely filtered out — no payout, no dilution contribution
        let config = make_config(&[("H100", 1, "10")], 4, "100", None);

        // "unknown-miner" is NOT in the hotkey map
        let unknown_row = cu_row(("unknown-miner", 99, "node-99", "8", ts(0), "H100", 4, "10"));
        let known_row = cu_row(("miner-0", 0, "node-0", "8", ts(0), "H100", 4, "10"));

        let result = compute_incentive_pool(
            &config,
            &[unknown_row, known_row],
            &[],
            ts(0),
            ts(4),
            d("100000"),
            999,
            &make_hotkey_map(1), // only "miner-0" → uid 0
            None,
        )
        .unwrap();

        // Unknown hotkey should have zero payout
        assert_eq!(miner_payout(&result, "unknown-miner"), Decimal::ZERO);

        // Known miner should get full undiluted payout (unknown row not counted in supply)
        // supply = 8, budget = 8 × 4 × $10 = $320, per_cu = $40
        // effective = MIN($10, $40, $100) = $10
        // payout = 1.0 × 8 × $10 = $80
        assert_eq!(miner_payout(&result, "miner-0"), d("80"));
    }

    #[test]
    fn test_unknown_gpu_category_cu_excluded() {
        // CU row with gpu_category not in config.gpu_categories → filtered out
        let config = make_config(&[("H100", 1, "10")], 4, "100", None);

        let unknown_cat_row = cu_row((
            "miner-0",
            0,
            "node-0",
            "8",
            ts(0),
            "RTX4090", // not in config
            4,
            "10",
        ));
        let known_cat_row = cu_row(("miner-1", 1, "node-1", "8", ts(0), "H100", 4, "10"));

        let result = compute_incentive_pool(
            &config,
            &[unknown_cat_row, known_cat_row],
            &[],
            ts(0),
            ts(4),
            d("100000"),
            999,
            &make_hotkey_map(2),
            None,
        )
        .unwrap();

        // RTX4090 row should produce zero payout (filtered by unknown category)
        assert_eq!(miner_payout(&result, "miner-0"), Decimal::ZERO);

        // Known category row gets full undiluted payout
        // supply = 8 (RTX4090 row excluded), budget = 8×4×$10 = $320, per_cu=$40
        // effective = MIN($10, $40, $100) = $10, payout = 8 × $10 = $80
        assert_eq!(miner_payout(&result, "miner-1"), d("80"));
    }

    #[test]
    fn test_ru_unknown_gpu_category_still_included() {
        // RU row with gpu_category not in config.gpu_categories
        // Should STILL be included (RU filtering only checks is_slashed and hotkey)
        let config = make_config(&[("H100", 1, "10")], 4, "100", None);

        let ru_unknown_cat = ru_row((
            "miner-0",
            0,
            "node-0",
            "100",
            ts(0),
            "RTX4090", // not in config — but RU rows don't filter on gpu_category
            4,
            30,
        ));

        let result = compute_incentive_pool(
            &config,
            &[],
            &[ru_unknown_cat],
            ts(0),
            ts(4),
            d("100000"),
            999,
            &make_hotkey_map(1),
            None,
        )
        .unwrap();

        // RU row should still get payout despite unknown category
        // payout = 1.0 × $100 × 30/100 = $30
        assert_eq!(miner_payout(&result, "miner-0"), d("30"));
    }

    // ==================== Category 9: Weight Normalization ====================

    #[test]
    fn test_weights_sum_to_u16_max_various_scenarios() {
        let sum_weights = |result: &IncentivePoolResult| -> u64 {
            result
                .distribution
                .weights
                .iter()
                .map(|w| w.weight as u64)
                .sum::<u64>()
        };

        // Scenario 1: 1 miner
        {
            let config = make_config(&[("H100", 1, "10")], 4, "100", None);
            let result = compute_incentive_pool(
                &config,
                &[cu_row((
                    "miner-0",
                    0,
                    "node-0",
                    "8",
                    ts(0),
                    "H100",
                    4,
                    "10",
                ))],
                &[],
                ts(0),
                ts(4),
                d("100000"),
                999,
                &make_hotkey_map(1),
                None,
            )
            .unwrap();
            assert_eq!(sum_weights(&result), u16::MAX as u64, "1 miner scenario");
        }

        // Scenario 2: 3 miners
        {
            let config = make_config(&[("H100", 1, "10")], 4, "100", None);
            let cu_rows: Vec<CuLedgerRowResponse> = (0..3)
                .map(|i| {
                    let hotkey = format!("miner-{i}");
                    let node_id = format!("node-{i}");
                    cu_row((&hotkey, i as u32, &node_id, "8", ts(0), "H100", 4, "10"))
                })
                .collect();
            let result = compute_incentive_pool(
                &config,
                &cu_rows,
                &[],
                ts(0),
                ts(4),
                d("100000"),
                999,
                &make_hotkey_map(3),
                None,
            )
            .unwrap();
            assert_eq!(sum_weights(&result), u16::MAX as u64, "3 miners scenario");
        }

        // Scenario 3: 10 miners
        {
            let config = make_config(&[("H100", 1, "10")], 4, "100", None);
            let cu_rows: Vec<CuLedgerRowResponse> = (0..10)
                .map(|i| {
                    let hotkey = format!("miner-{i}");
                    let node_id = format!("node-{i}");
                    cu_row((&hotkey, i as u32, &node_id, "8", ts(0), "H100", 4, "10"))
                })
                .collect();
            let result = compute_incentive_pool(
                &config,
                &cu_rows,
                &[],
                ts(0),
                ts(4),
                d("100000"),
                999,
                &make_hotkey_map(10),
                None,
            )
            .unwrap();
            assert_eq!(sum_weights(&result), u16::MAX as u64, "10 miners scenario");
        }

        // Scenario 4: high burn (tiny payout relative to capacity)
        {
            let config = make_config(&[("H100", 1, "10")], 4, "100", None);
            let result = compute_incentive_pool(
                &config,
                &[cu_row((
                    "miner-0",
                    0,
                    "node-0",
                    "0.01",
                    ts(0),
                    "H100",
                    4,
                    "10",
                ))],
                &[],
                ts(0),
                ts(4),
                d("100000"),
                999,
                &make_hotkey_map(1),
                None,
            )
            .unwrap();
            assert_eq!(sum_weights(&result), u16::MAX as u64, "high burn scenario");
        }

        // Scenario 5: low burn (payouts near capacity)
        {
            let config = make_config(&[("H100", 1, "10")], 4, "100", None);
            // payout = 1.0 × 8 × $10 = $80, capacity = $81 → burn ≈ 1.2%
            let result = compute_incentive_pool(
                &config,
                &[cu_row((
                    "miner-0",
                    0,
                    "node-0",
                    "8",
                    ts(0),
                    "H100",
                    4,
                    "10",
                ))],
                &[],
                ts(0),
                ts(4),
                d("81"),
                999,
                &make_hotkey_map(1),
                None,
            )
            .unwrap();
            assert_eq!(sum_weights(&result), u16::MAX as u64, "low burn scenario");
        }
    }

    #[test]
    fn test_single_miner_weight_share() {
        // 1 miner, payout=$50, capacity=$100
        // Config: H100 target_count=1 (→ 8 GPUs), max_cu=$100
        let config = make_config(&[("H100", 1, "10")], 4, "100", None);
        // cu_amount=5 → capacity_budget = 8×4×$10=$320, supply=5, per_cu=$64
        // effective = MIN($10, $64, $100) = $10, payout = 5 × $10 = $50
        let result = compute_incentive_pool(
            &config,
            &[cu_row((
                "miner-0",
                0,
                "node-0",
                "5",
                ts(0),
                "H100",
                4,
                "10",
            ))],
            &[],
            ts(0),
            ts(4),
            d("100"),
            999,
            &make_hotkey_map(1),
            None,
        )
        .unwrap();

        assert_eq!(miner_payout(&result, "miner-0"), d("50"));

        // Miner share = 50/100 = 0.5, burn share = 0.5
        // Raw weights: each = 0.5 × 65535 = 32767.5, floor=32767, remainder=0.5
        // Leftover = 65535 - 2×32767 = 1, awarded to lowest uid with highest remainder
        // Both remainders equal (0.5), so uid 0 (miner) wins the tiebreak over uid 999 (burn)
        let miner_w = miner_weight(&result, 0);
        let burn_w = miner_weight(&result, 999);
        assert_eq!(miner_w, 32768, "miner gets half + rounding bonus");
        assert_eq!(burn_w, 32767, "burn gets half");
        assert_eq!(
            miner_w as u64 + burn_w as u64,
            u16::MAX as u64,
            "weights must sum to u16::MAX"
        );
    }

    #[test]
    fn test_many_miners_greedy_rounding() {
        // 100 miners each with tiny CU → each gets a tiny weight
        let config = make_config(&[("H100", 1, "10")], 4, "100", None);
        let n = 100usize;
        let cu_rows: Vec<CuLedgerRowResponse> = (0..n)
            .map(|i| {
                let hotkey = format!("miner-{i}");
                let node_id = format!("node-{i}");
                cu_row((&hotkey, i as u32, &node_id, "1", ts(0), "H100", 4, "10"))
            })
            .collect();

        let result = compute_incentive_pool(
            &config,
            &cu_rows,
            &[],
            ts(0),
            ts(4),
            d("100000"),
            999,
            &make_hotkey_map(n),
            None,
        )
        .unwrap();

        // Verify weights sum to u16::MAX
        let total: u64 = result
            .distribution
            .weights
            .iter()
            .map(|w| w.weight as u64)
            .sum();
        assert_eq!(
            total,
            u16::MAX as u64,
            "100 miners: weights must sum to u16::MAX"
        );

        // Verify no miner with nonzero payout has weight=0
        for i in 0..n {
            let hotkey = format!("miner-{i}");
            let payout = miner_payout(&result, &hotkey);
            if payout > Decimal::ZERO {
                let uid = i as u16;
                let w = miner_weight(&result, uid);
                assert!(
                    w > 0,
                    "miner-{i} has nonzero payout ({payout}) but zero weight"
                );
            }
        }
    }

    // ==================== Category 10: Potential Bug-Finding Edge Cases ====================

    #[test]
    fn test_category_supply_includes_non_vesting_rows() {
        // Potential bug: category_cu_supply sums ALL active CU rows, even those
        // with zero vesting overlap in the current epoch. This dilutes vesting rows.
        let config = make_config(&[("H100", 1, "10")], 4, "100", None);
        let hotkeys = make_hotkey_map(2);

        // With only Row A: supply=80, per_cu_budget=320/80=$4, payout = 80*$4 = $320
        let result_alone = compute_incentive_pool(
            &config,
            &[cu_row((
                "miner-0",
                0,
                "node-0",
                "80",
                ts(0),
                "H100",
                4,
                "10",
            ))],
            &[],
            ts(0),
            ts(4),
            d("100000"),
            999,
            &hotkeys,
            None,
        )
        .unwrap();
        assert_eq!(miner_payout(&result_alone, "miner-0"), d("320"));

        // With both rows: supply=160, Row A per_cu_budget=320/160=$2, payout = 80*$2 = $160
        // Row B vests in [T10,T14], NOT in epoch [T0,T4]
        let result_both = compute_incentive_pool(
            &config,
            &[
                cu_row(("miner-0", 0, "node-0", "80", ts(0), "H100", 4, "10")),
                cu_row(("miner-1", 1, "node-1", "80", ts(10), "H100", 4, "10")),
            ],
            &[],
            ts(0),
            ts(4),
            d("100000"),
            999,
            &hotkeys,
            None,
        )
        .unwrap();

        // Row B doesn't vest in this epoch → zero payout
        assert_eq!(miner_payout(&result_both, "miner-1"), Decimal::ZERO);
        // Row A is diluted by non-vesting Row B's supply contribution
        // This is the current behavior — non-vesting rows inflate category_cu_supply
        assert_eq!(miner_payout(&result_both, "miner-0"), d("160"));
    }

    #[test]
    fn test_very_small_cu_amount_no_precision_loss() {
        let config = make_config(&[("H100", 1, "10")], 4, "100", None);
        let hotkeys = make_hotkey_map(1);

        let result = compute_incentive_pool(
            &config,
            &[cu_row((
                "miner-0",
                0,
                "node-0",
                "0.000001",
                ts(0),
                "H100",
                4,
                "10",
            ))],
            &[],
            ts(0),
            ts(4),
            d("100000"),
            999,
            &hotkeys,
            None,
        )
        .unwrap();

        // supply=0.000001, budget=8*4*$10=$320, per_cu=$320/0.000001=$320000000
        // effective = MIN($10, $320000000, $100) = $10
        // payout = 1.0 * 0.000001 * $10 = $0.00001
        let payout = miner_payout(&result, "miner-0");
        assert_eq!(payout, d("0.00001"));
        assert!(payout > Decimal::ZERO);
        // Weight is 0 because the miner's share of capacity (1e-10) is too small
        // to represent in u16 weight space — this is expected behavior.
        assert_eq!(miner_weight(&result, 0), 0);
    }

    #[test]
    fn test_negative_emission_capacity() {
        let config = make_config(&[("H100", 1, "10")], 4, "100", None);
        let hotkeys = make_hotkey_map(1);

        let result = compute_incentive_pool(
            &config,
            &[cu_row((
                "miner-0",
                0,
                "node-0",
                "8",
                ts(0),
                "H100",
                4,
                "10",
            ))],
            &[],
            ts(0),
            ts(4),
            d("-1"),
            999,
            &hotkeys,
            None,
        )
        .unwrap();

        // Should go to all-burn path (effective_capacity ≤ 0)
        assert_eq!(result.distribution.weights.len(), 1);
        assert_eq!(result.distribution.weights[0].uid, 999);
        assert_eq!(result.distribution.weights[0].weight, u16::MAX);
        assert_eq!(result.burn_rate, Decimal::ONE);
        assert_eq!(result.burn_percentage, 100.0);
    }

    #[test]
    fn test_zero_cu_amount_rows_ignored() {
        let config = make_config(&[("H100", 1, "10")], 4, "100", None);
        let hotkeys = make_hotkey_map(2);

        let result = compute_incentive_pool(
            &config,
            &[
                cu_row(("miner-0", 0, "node-0", "0", ts(0), "H100", 4, "10")),
                cu_row(("miner-1", 1, "node-1", "8", ts(0), "H100", 4, "10")),
            ],
            &[],
            ts(0),
            ts(4),
            d("100000"),
            999,
            &hotkeys,
            None,
        )
        .unwrap();

        // Zero-CU miner gets zero payout and zero weight
        assert_eq!(miner_payout(&result, "miner-0"), Decimal::ZERO);
        assert_eq!(miner_weight(&result, 0), 0);

        // Non-zero CU miner gets expected payout
        // supply = 0 + 8 = 8, per_cu_budget = (8*4*$10)/8 = $40
        // effective = MIN($10, $40, $100) = $10
        // payout = 1.0 * 8 * $10 = $80
        assert_eq!(miner_payout(&result, "miner-1"), d("80"));
        assert!(miner_weight(&result, 1) > 0);
    }

    #[test]
    fn test_different_window_hours_per_row() {
        // Two CU rows in same category with different snapshotted window_hours (4h and 8h)
        let config = make_config(&[("H100", 1, "10")], 4, "100", None);
        let hotkeys = make_hotkey_map(2);

        let result = compute_incentive_pool(
            &config,
            &[
                cu_row(("miner-0", 0, "node-0", "8", ts(0), "H100", 4, "10")),
                cu_row(("miner-1", 1, "node-1", "8", ts(0), "H100", 8, "10")),
            ],
            &[],
            ts(0),
            ts(4),
            d("100000"),
            999,
            &hotkeys,
            None,
        )
        .unwrap();

        // category_supply = 8 + 8 = 16
        // Row 1 (4h window): vested=1.0, budget=8*4*$10=$320, per_cu=$320/16=$20
        //   effective = MIN($10, $20, $100) = $10, payout = 1.0 * 8 * $10 = $80
        // Row 2 (8h window): vested=4/8=0.5, budget=8*8*$10=$640, per_cu=$640/16=$40
        //   effective = MIN($10, $40, $100) = $10, payout = 0.5 * 8 * $10 = $40
        assert_eq!(miner_payout(&result, "miner-0"), d("80"));
        assert_eq!(miner_payout(&result, "miner-1"), d("40"));
    }

    #[test]
    fn test_different_price_usd_per_row() {
        // Two CU rows in same category with different snapshotted price_usd ($10 and $5)
        let config = make_config(&[("H100", 1, "10")], 4, "100", None);
        let hotkeys = make_hotkey_map(2);

        let result = compute_incentive_pool(
            &config,
            &[
                cu_row(("miner-0", 0, "node-0", "8", ts(0), "H100", 4, "10")),
                cu_row(("miner-1", 1, "node-1", "8", ts(0), "H100", 4, "5")),
            ],
            &[],
            ts(0),
            ts(4),
            d("100000"),
            999,
            &hotkeys,
            None,
        )
        .unwrap();

        // category_supply = 8 + 8 = 16
        // Row 1 ($10): budget=8*4*$10=$320, per_cu=$320/16=$20
        //   effective = MIN($10, $20, $100) = $10, payout = 1.0 * 8 * $10 = $80
        // Row 2 ($5): budget=8*4*$5=$160, per_cu=$160/16=$10
        //   effective = MIN($5, $10, $100) = $5, payout = 1.0 * 8 * $5 = $40
        assert_eq!(miner_payout(&result, "miner-0"), d("80"));
        assert_eq!(miner_payout(&result, "miner-1"), d("40"));
    }
}
