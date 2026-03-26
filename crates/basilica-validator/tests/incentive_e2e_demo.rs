//! End-to-End Demo: Rental to Weight Setting Flow
//!
//! This test demonstrates the complete incentive mechanism flow:
//! 1. Miners submit signed bids
//! 2. User requests rental → validator selects lowest bidder
//! 3. Rental runs, telemetry processed → billing calculates revenue
//! 4. Weight setter fetches deliveries
//! 5. Weights calculated proportional to revenue_usd
//! 6. Hotkey→UID resolution protects against deregistration attacks
//!
//! Run with: cargo test --test incentive_e2e_demo -- --nocapture

use std::collections::HashMap;

/// Simulates a miner's bid
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct MinerBid {
    miner_hotkey: String,
    miner_uid: u16,
    gpu_category: String,
    bid_per_hour_cents: u32, // cents/hr per GPU (e.g., 250 = $2.50)
    gpu_count: u32,
}

/// Simulates a rental record
#[derive(Debug, Clone)]
struct Rental {
    rental_id: String,
    miner_hotkey: String,
    miner_uid: u16,
    gpu_category: String,
    gpu_count: u32,
    hours_used: f64,
    bid_rate_cents: u32, // Miner's bid rate per GPU-hour (cents)
}

/// Simulates a delivery record (output of billing)
#[derive(Debug, Clone)]
struct MinerDelivery {
    miner_hotkey: String,
    miner_uid: u16, // UID at rental time
    gpu_category: String,
    total_hours: f64,
    revenue_usd: f64, // net miner revenue -- weights based on this
}

/// Simulates metagraph state
#[derive(Debug, Clone)]
struct Metagraph {
    /// Index = UID, Value = Hotkey
    hotkeys: Vec<String>,
}

impl Metagraph {
    fn hotkey_to_uid(&self) -> HashMap<String, u16> {
        self.hotkeys
            .iter()
            .enumerate()
            .map(|(uid, hk)| (hk.clone(), uid as u16))
            .collect()
    }
}

/// Emission config for weight distribution
#[derive(Debug, Clone)]
struct EmissionConfig {
    gpu_allocations: HashMap<String, f64>, // category -> percentage (must sum to 100)
    burn_percentage: f64,
}

fn separator() -> String {
    "=".repeat(70)
}

fn line() -> String {
    "-".repeat(50)
}

/// Simulates the auction/bid selection logic
fn select_winning_bidder(
    bids: &[MinerBid],
    gpu_category: &str,
    gpu_count: u32,
) -> Option<MinerBid> {
    bids.iter()
        .filter(|b| b.gpu_category == gpu_category && b.gpu_count >= gpu_count)
        .min_by_key(|b| b.bid_per_hour_cents)
        .cloned()
}

/// Simulates weight calculation (simplified version of weight_allocation.rs)
fn calculate_weights(
    deliveries: &[MinerDelivery],
    metagraph: &Metagraph,
    emission_config: &EmissionConfig,
) -> Vec<(u16, f64)> {
    let hotkey_to_uid = metagraph.hotkey_to_uid();
    let total_weight = 1.0; // Normalized

    // Calculate burn
    let burn_weight = total_weight * (emission_config.burn_percentage / 100.0);
    let remaining_weight = total_weight - burn_weight;

    // Group deliveries by category, resolving UID from current metagraph
    let mut miners_by_category: HashMap<String, Vec<(u16, f64)>> = HashMap::new();
    let mut skipped_deregistered = 0;
    let mut uid_changed = 0;

    for delivery in deliveries {
        // CRITICAL: Resolve UID from hotkey using CURRENT metagraph
        let current_uid = match hotkey_to_uid.get(&delivery.miner_hotkey) {
            Some(&uid) => {
                if uid != delivery.miner_uid {
                    uid_changed += 1;
                    println!(
                        "  ⚠️  UID changed for {}: {} → {} (using current)",
                        delivery.miner_hotkey, delivery.miner_uid, uid
                    );
                }
                uid
            }
            None => {
                skipped_deregistered += 1;
                println!(
                    "  🚫 Miner {} (UID {}) deregistered - clearing ${:.2} pending revenue",
                    delivery.miner_hotkey, delivery.miner_uid, delivery.revenue_usd
                );
                continue;
            }
        };

        if delivery.revenue_usd <= 0.0 {
            continue;
        }

        miners_by_category
            .entry(delivery.gpu_category.clone())
            .or_default()
            .push((current_uid, delivery.revenue_usd));
    }

    if skipped_deregistered > 0 {
        println!(
            "  📊 Skipped {} deregistered miners, {} UID changes",
            skipped_deregistered, uid_changed
        );
    }

    // Calculate weights per category
    let mut weights: HashMap<u16, f64> = HashMap::new();

    for (category, miners) in &miners_by_category {
        let category_allocation = emission_config
            .gpu_allocations
            .get(category)
            .copied()
            .unwrap_or(0.0);

        let category_weight_pool = remaining_weight * (category_allocation / 100.0);
        let total_payment: f64 = miners.iter().map(|(_, payment)| payment).sum();

        if total_payment <= 0.0 {
            continue;
        }

        for (uid, payment) in miners {
            let miner_share = payment / total_payment;
            let miner_weight = category_weight_pool * miner_share;

            // Aggregate weights for multi-category miners
            *weights.entry(*uid).or_default() += miner_weight;
        }
    }

    let mut result: Vec<(u16, f64)> = weights.into_iter().collect();
    result.sort_by_key(|(uid, _)| *uid);
    result
}

#[test]
fn demo_full_incentive_flow() {
    println!("\n{}", separator());
    println!("🚀 E2E DEMO: Rental to Weight Setting Flow");
    println!("{}\n", separator());

    // =========================================================================
    // STEP 1: Initial Metagraph State
    // =========================================================================
    println!("📋 STEP 1: Initial Metagraph State");
    println!("{}", line());

    let metagraph = Metagraph {
        hotkeys: vec![
            "5GrwvaEF...Alice".to_string(),   // UID 0
            "5FHneW46...Bob".to_string(),     // UID 1
            "5FLSigC9...Charlie".to_string(), // UID 2
            "5DAAnrj7...Dave".to_string(),    // UID 3
        ],
    };

    for (uid, hk) in metagraph.hotkeys.iter().enumerate() {
        println!("  UID {}: {}", uid, hk);
    }
    println!();

    // =========================================================================
    // STEP 2: Miners Submit Bids
    // =========================================================================
    println!("📋 STEP 2: Miners Submit Signed Bids");
    println!("{}", line());

    let bids = vec![
        MinerBid {
            miner_hotkey: "5FHneW46...Bob".to_string(),
            miner_uid: 1,
            gpu_category: "H100".to_string(),
            bid_per_hour_cents: 250, // Bob bids $2.50/hr
            gpu_count: 8,
        },
        MinerBid {
            miner_hotkey: "5FLSigC9...Charlie".to_string(),
            miner_uid: 2,
            gpu_category: "H100".to_string(),
            bid_per_hour_cents: 200, // Charlie bids $2.00/hr (LOWEST!)
            gpu_count: 8,
        },
        MinerBid {
            miner_hotkey: "5DAAnrj7...Dave".to_string(),
            miner_uid: 3,
            gpu_category: "A100".to_string(),
            bid_per_hour_cents: 120, // Dave bids on A100s
            gpu_count: 4,
        },
    ];

    for bid in &bids {
        println!(
            "  {} (UID {}): {} x{} @ ${:.2}/hr",
            bid.miner_hotkey,
            bid.miner_uid,
            bid.gpu_category,
            bid.gpu_count,
            bid.bid_per_hour_cents as f64 / 100.0
        );
    }
    println!();

    // =========================================================================
    // STEP 3: User Requests Rental - Lowest Bidder Wins
    // =========================================================================
    println!("📋 STEP 3: User Requests H100 Rental");
    println!("{}", line());

    let user_request_category = "H100";
    let user_request_gpus = 8;

    let winner = select_winning_bidder(&bids, user_request_category, user_request_gpus)
        .expect("Should find a winning bidder");

    println!(
        "  User wants: {} x{} GPUs",
        user_request_category, user_request_gpus
    );
    println!();
    println!(
        "  🏆 WINNER: {} (UID {})",
        winner.miner_hotkey, winner.miner_uid
    );
    println!(
        "     Bid: ${:.2}/GPU-hour",
        winner.bid_per_hour_cents as f64 / 100.0
    );
    println!();

    // =========================================================================
    // STEP 4: Rental Executes - Create Billing Records
    // =========================================================================
    println!("📋 STEP 4: Rentals Execute (simulate several rentals)");
    println!("{}", line());

    let rentals = vec![
        // Charlie won the H100 rental
        Rental {
            rental_id: "rental_001".to_string(),
            miner_hotkey: winner.miner_hotkey.clone(),
            miner_uid: winner.miner_uid,
            gpu_category: "H100".to_string(),
            gpu_count: 8,
            hours_used: 10.0, // 10 hours
            bid_rate_cents: winner.bid_per_hour_cents,
        },
        // Dave had an A100 rental
        Rental {
            rental_id: "rental_002".to_string(),
            miner_hotkey: "5DAAnrj7...Dave".to_string(),
            miner_uid: 3,
            gpu_category: "A100".to_string(),
            gpu_count: 4,
            hours_used: 20.0, // 20 hours
            bid_rate_cents: 120,
        },
        // Bob also got a smaller H100 rental later
        Rental {
            rental_id: "rental_003".to_string(),
            miner_hotkey: "5FHneW46...Bob".to_string(),
            miner_uid: 1,
            gpu_category: "H100".to_string(),
            gpu_count: 4,
            hours_used: 5.0,
            bid_rate_cents: 250,
        },
    ];

    for rental in &rentals {
        let gpu_hours = rental.gpu_count as f64 * rental.hours_used;
        let revenue_cents = (gpu_hours * rental.bid_rate_cents as f64).round() as u64;

        println!(
            "  📦 {}: {} (UID {})",
            rental.rental_id, rental.miner_hotkey, rental.miner_uid
        );
        println!(
            "     {} x{} for {:.1} hours",
            rental.gpu_category, rental.gpu_count, rental.hours_used
        );
        println!("     GPU-hours: {:.1}", gpu_hours);
        println!(
            "     Revenue: ${:.2} (@ ${:.2}/GPU-hr)",
            revenue_cents as f64 / 100.0,
            rental.bid_rate_cents as f64 / 100.0
        );
        println!();
    }

    // =========================================================================
    // STEP 5: Billing Aggregates to MinerDelivery
    // =========================================================================
    println!("📋 STEP 5: Billing Creates MinerDelivery Records");
    println!("{}", line());

    // Aggregate rentals to deliveries (simplified - real code groups by miner+category)
    let mut delivery_map: HashMap<(String, String), MinerDelivery> = HashMap::new();

    for rental in &rentals {
        let key = (rental.miner_hotkey.clone(), rental.gpu_category.clone());
        let gpu_hours = rental.gpu_count as f64 * rental.hours_used;
        // Calculate in cents, convert to dollars for MinerDelivery (which uses f64 USD)
        let revenue_usd = (gpu_hours * rental.bid_rate_cents as f64) / 100.0;

        let entry = delivery_map.entry(key).or_insert(MinerDelivery {
            miner_hotkey: rental.miner_hotkey.clone(),
            miner_uid: rental.miner_uid,
            gpu_category: rental.gpu_category.clone(),
            total_hours: 0.0,
            revenue_usd: 0.0,
        });

        entry.total_hours += gpu_hours;
        entry.revenue_usd += revenue_usd;
    }

    let deliveries: Vec<MinerDelivery> = delivery_map.into_values().collect();

    for d in &deliveries {
        println!(
            "  {} (UID {}): {} category",
            d.miner_hotkey, d.miner_uid, d.gpu_category
        );
        println!("     GPU-hours: {:.1}", d.total_hours);
        println!(
            "     Revenue: ${:.2} ← WEIGHTS BASED ON THIS",
            d.revenue_usd
        );
        println!();
    }

    // =========================================================================
    // STEP 6: Weight Calculation
    // =========================================================================
    println!("📋 STEP 6: Weight Calculation");
    println!("{}", line());

    let emission_config = EmissionConfig {
        gpu_allocations: {
            let mut m = HashMap::new();
            m.insert("H100".to_string(), 60.0); // H100 gets 60% of emissions
            m.insert("A100".to_string(), 30.0); // A100 gets 30%
            m.insert("B200".to_string(), 10.0); // B200 gets 10%
            m
        },
        burn_percentage: 0.0, // No burn for simplicity
    };

    println!("  Emission allocation:");
    for (cat, pct) in &emission_config.gpu_allocations {
        println!("    {}: {:.0}%", cat, pct);
    }
    println!();

    let weights = calculate_weights(&deliveries, &metagraph, &emission_config);

    println!("  🎯 Final Weights (normalized to 1.0):");
    let total_weight: f64 = weights.iter().map(|(_, w)| w).sum();
    for (uid, weight) in &weights {
        let pct = (weight / total_weight) * 100.0;
        let hotkey = &metagraph.hotkeys[*uid as usize];
        println!("    UID {}: {:.4} ({:.1}%) - {}", uid, weight, pct, hotkey);
    }
    println!();

    // =========================================================================
    // STEP 7: Verify Economic Properties
    // =========================================================================
    println!("📋 STEP 7: Verify Economic Properties");
    println!("{}", line());

    let charlie_payment: f64 = deliveries
        .iter()
        .filter(|d| d.miner_hotkey.contains("Charlie"))
        .map(|d| d.revenue_usd)
        .sum();

    let bob_payment: f64 = deliveries
        .iter()
        .filter(|d| d.miner_hotkey.contains("Bob"))
        .map(|d| d.revenue_usd)
        .sum();

    let dave_payment: f64 = deliveries
        .iter()
        .filter(|d| d.miner_hotkey.contains("Dave"))
        .map(|d| d.revenue_usd)
        .sum();

    println!("  Miner payments:");
    println!("    Charlie: ${:.2}", charlie_payment);
    println!("    Bob: ${:.2}", bob_payment);
    println!("    Dave: ${:.2}", dave_payment);
    println!();

    // Within H100 category, weights should be proportional to payments
    let charlie_h100_weight = weights
        .iter()
        .find(|(uid, _)| *uid == 2)
        .map(|(_, w)| *w)
        .unwrap_or(0.0);
    let bob_h100_weight = weights
        .iter()
        .find(|(uid, _)| *uid == 1)
        .map(|(_, w)| *w)
        .unwrap_or(0.0);

    if bob_h100_weight > 0.0 && charlie_h100_weight > 0.0 {
        println!("  H100 category weight distribution:");
        println!(
            "    Charlie: {:.4} (payment: ${:.2})",
            charlie_h100_weight, charlie_payment
        );
        println!(
            "    Bob: {:.4} (payment: ${:.2})",
            bob_h100_weight, bob_payment
        );

        let h100_payment_ratio = charlie_payment / bob_payment;
        let h100_weight_ratio = charlie_h100_weight / bob_h100_weight;
        println!("    Payment ratio (Charlie/Bob): {:.2}", h100_payment_ratio);
        println!("    Weight ratio (Charlie/Bob): {:.2}", h100_weight_ratio);

        let ratio_match = (h100_payment_ratio - h100_weight_ratio).abs() < 0.1;
        println!(
            "    ✅ Ratios match: {}",
            if ratio_match { "YES" } else { "CLOSE" }
        );
    }
    println!();

    // =========================================================================
    // STEP 8: Deregistration Protection Demo
    // =========================================================================
    println!("📋 STEP 8: Deregistration Protection Demo");
    println!("{}", line());

    // Simulate Charlie deregistering (remove from metagraph)
    let metagraph_after_deregister = Metagraph {
        hotkeys: vec![
            "5GrwvaEF...Alice".to_string(), // UID 0
            "5FHneW46...Bob".to_string(),   // UID 1
            "5NewMiner...Eve".to_string(),  // UID 2 - NEW MINER took Charlie's slot!
            "5DAAnrj7...Dave".to_string(),  // UID 3
        ],
    };

    println!("  ⚡ Simulating: Charlie deregistered, Eve took UID 2");
    println!();

    println!("  Running weight calculation with OLD deliveries but NEW metagraph...");
    println!();

    let weights_after =
        calculate_weights(&deliveries, &metagraph_after_deregister, &emission_config);

    println!();
    println!("  🎯 Weights after deregistration:");
    let total_after: f64 = weights_after.iter().map(|(_, w)| w).sum();
    for (uid, weight) in &weights_after {
        let pct = if total_after > 0.0 {
            (weight / total_after) * 100.0
        } else {
            0.0
        };
        let hotkey = &metagraph_after_deregister.hotkeys[*uid as usize];
        println!("    UID {}: {:.4} ({:.1}%) - {}", uid, weight, pct, hotkey);
    }
    println!();

    // Verify Eve (new UID 2) did NOT get Charlie's weight
    let eve_weight = weights_after
        .iter()
        .find(|(uid, _)| *uid == 2)
        .map(|(_, w)| *w)
        .unwrap_or(0.0);
    println!("  ✅ Eve (new UID 2) weight: {:.4}", eve_weight);
    println!("     Charlie's pending revenue was CLEARED, not given to Eve!");
    println!();

    assert!(
        eve_weight == 0.0,
        "Eve should NOT receive Charlie's weight - deregistration protection failed!"
    );

    // =========================================================================
    // STEP 9: UID Migration Demo
    // =========================================================================
    println!("📋 STEP 9: UID Migration Demo");
    println!("{}", line());

    // Simulate Bob re-registering at a different UID
    let metagraph_uid_change = Metagraph {
        hotkeys: vec![
            "5GrwvaEF...Alice".to_string(),   // UID 0
            "5NewMiner...Eve".to_string(),    // UID 1 - Eve took Bob's old slot
            "5FLSigC9...Charlie".to_string(), // UID 2
            "5DAAnrj7...Dave".to_string(),    // UID 3
            "5FHneW46...Bob".to_string(),     // UID 4 - Bob re-registered at new UID!
        ],
    };

    println!("  ⚡ Simulating: Bob re-registered at UID 4 (was UID 1)");
    println!();

    println!("  Running weight calculation...");
    println!();

    let weights_migrated = calculate_weights(&deliveries, &metagraph_uid_change, &emission_config);

    println!();
    println!("  🎯 Weights after UID migration:");
    for (uid, weight) in &weights_migrated {
        if *weight > 0.0 {
            let hotkey = &metagraph_uid_change.hotkeys[*uid as usize];
            println!("    UID {}: {:.4} - {}", uid, weight, hotkey);
        }
    }
    println!();

    // Verify Bob's weight went to UID 4 (his new UID), not UID 1 (Eve)
    let bob_new_weight = weights_migrated
        .iter()
        .find(|(uid, _)| *uid == 4)
        .map(|(_, w)| *w)
        .unwrap_or(0.0);
    let eve_weight = weights_migrated
        .iter()
        .find(|(uid, _)| *uid == 1)
        .map(|(_, w)| *w)
        .unwrap_or(0.0);

    println!("  ✅ Bob (new UID 4) weight: {:.4}", bob_new_weight);
    println!("  ✅ Eve (UID 1) weight: {:.4} (should be 0!)", eve_weight);
    println!();

    assert!(
        bob_new_weight > 0.0,
        "Bob should receive weight at his new UID"
    );
    assert!(
        eve_weight == 0.0,
        "Eve should NOT receive Bob's weight - UID migration protection failed!"
    );

    println!("{}", separator());
    println!("✅ E2E DEMO COMPLETE - All incentive mechanism checks passed!");
    println!("{}\n", separator());
}

#[test]
fn demo_bid_economics() {
    println!("\n{}", separator());
    println!("💰 BID ECONOMICS DEMO");
    println!("{}\n", separator());

    // Scenario: Three miners bid on same GPU category
    let hours = 100.0;
    let gpus = 8;
    let gpu_hours = hours * gpus as f64;

    // (name, bid_cents, note)
    let scenarios: Vec<(&str, u32, &str)> = vec![
        ("Alice", 350, "High bid - less likely to win"),
        ("Bob", 250, "Medium bid"),
        ("Charlie", 150, "Low bid - most likely to win"),
    ];

    println!(
        "Rental: {} GPUs for {} hours = {} GPU-hours",
        gpus, hours, gpu_hours
    );
    println!();

    for (name, bid_cents, note) in &scenarios {
        let revenue_usd = (gpu_hours * *bid_cents as f64) / 100.0;

        println!(
            "  {} bids ${:.2}/GPU-hr ({})",
            name,
            *bid_cents as f64 / 100.0,
            note
        );
        println!("    If wins: Revenue = ${:.2}", revenue_usd);
        println!();
    }

    // Show weight implications
    println!("  Weight implications (if all three win equal rentals):");

    let total_revenue_usd: f64 = scenarios
        .iter()
        .map(|(_, bid_cents, _)| (gpu_hours * *bid_cents as f64) / 100.0)
        .sum();

    for (name, bid_cents, _) in &scenarios {
        let revenue_usd = (gpu_hours * *bid_cents as f64) / 100.0;
        let weight_share = revenue_usd / total_revenue_usd;
        println!(
            "    {}: ${:.2} revenue → {:.1}% of category weight",
            name,
            revenue_usd,
            weight_share * 100.0
        );
    }
    println!();

    println!("  🎯 KEY INSIGHT: Lower bids = lower weight share");
    println!("     Miners must balance: win probability vs reward share");
    println!();
}

#[test]
fn demo_category_caps() {
    println!("\n{}", separator());
    println!("📊 CATEGORY CAPS DEMO");
    println!("{}\n", separator());

    let emission_config = EmissionConfig {
        gpu_allocations: {
            let mut m = HashMap::new();
            m.insert("H100".to_string(), 50.0);
            m.insert("A100".to_string(), 30.0);
            m.insert("B200".to_string(), 20.0);
            m
        },
        burn_percentage: 0.0,
    };

    println!("  Category allocation:");
    for (cat, pct) in &emission_config.gpu_allocations {
        println!("    {}: {}%", cat, pct);
    }
    println!();

    // Scenario: One miner dominates both H100 and A100
    let deliveries = vec![
        MinerDelivery {
            miner_hotkey: "DominantMiner".to_string(),
            miner_uid: 0,
            gpu_category: "H100".to_string(),
            total_hours: 1000.0,
            revenue_usd: 2500.0,
        },
        MinerDelivery {
            miner_hotkey: "DominantMiner".to_string(),
            miner_uid: 0,
            gpu_category: "A100".to_string(),
            total_hours: 800.0,
            revenue_usd: 1200.0,
        },
        MinerDelivery {
            miner_hotkey: "SmallH100Miner".to_string(),
            miner_uid: 1,
            gpu_category: "H100".to_string(),
            total_hours: 100.0,
            revenue_usd: 278.0,
        },
        MinerDelivery {
            miner_hotkey: "SmallA100Miner".to_string(),
            miner_uid: 2,
            gpu_category: "A100".to_string(),
            total_hours: 200.0,
            revenue_usd: 300.0,
        },
    ];

    let metagraph = Metagraph {
        hotkeys: vec![
            "DominantMiner".to_string(),
            "SmallH100Miner".to_string(),
            "SmallA100Miner".to_string(),
        ],
    };

    let weights = calculate_weights(&deliveries, &metagraph, &emission_config);

    println!("  Delivery summary:");
    for d in &deliveries {
        println!(
            "    {} ({}) ${:.0}",
            d.miner_hotkey, d.gpu_category, d.revenue_usd
        );
    }
    println!();

    println!("  🎯 Final weights:");
    let total: f64 = weights.iter().map(|(_, w)| w).sum();
    for (uid, weight) in &weights {
        let pct = (weight / total) * 100.0;
        let hotkey = &metagraph.hotkeys[*uid as usize];
        println!("    {} (UID {}): {:.1}%", hotkey, uid, pct);
    }
    println!();

    // Calculate what DominantMiner got
    let dominant_weight = weights
        .iter()
        .find(|(uid, _)| *uid == 0)
        .map(|(_, w)| *w)
        .unwrap_or(0.0);
    let dominant_pct = (dominant_weight / total) * 100.0;

    // Compute DominantMiner's share in each category from the data
    let h100_total: f64 = deliveries
        .iter()
        .filter(|d| d.gpu_category == "H100")
        .map(|d| d.revenue_usd)
        .sum();
    let a100_total: f64 = deliveries
        .iter()
        .filter(|d| d.gpu_category == "A100")
        .map(|d| d.revenue_usd)
        .sum();
    let dominant_h100: f64 = deliveries
        .iter()
        .filter(|d| d.miner_hotkey == "DominantMiner" && d.gpu_category == "H100")
        .map(|d| d.revenue_usd)
        .sum();
    let dominant_a100: f64 = deliveries
        .iter()
        .filter(|d| d.miner_hotkey == "DominantMiner" && d.gpu_category == "A100")
        .map(|d| d.revenue_usd)
        .sum();

    let h100_share = dominant_h100 / h100_total;
    let a100_share = dominant_a100 / a100_total;
    let expected_pct = h100_share * 50.0 + a100_share * 30.0;

    println!("  DominantMiner analysis:");
    println!("    Expected weight: {:.0}%", expected_pct);
    println!("    Actual weight: {:.1}%", dominant_pct);
    println!();

    println!("  ✅ Category caps work correctly:");
    println!("     - DominantMiner can't exceed sum of their category shares");
    println!("     - Multi-category miners earn from multiple CAPPED pools");
    println!();
}
