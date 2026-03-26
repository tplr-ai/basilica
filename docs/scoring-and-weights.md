# Scoring and Weight Setting in Basilica

This document explains how the Basilica validator scores miners and sets weights on the Bittensor network.

## Overview

The validator uses a **delivery-based weight model**: miners only receive emissions when their GPU nodes generate rental revenue. Weights are proportional to the `revenue_usd` recorded in delivery records from the billing API. GPUs that are not actively rented do not earn emissions.

## GPU Categories

The system recognizes the following GPU categories:

- **A100** - NVIDIA A100 GPUs
- **H100** - NVIDIA H100 GPUs
- **H200** - NVIDIA H200 GPUs
- **B200** - NVIDIA B200 GPUs
- **Other** - All other GPU types (not eligible for weight allocation)

**NOTE**: Only GPUs with a configured allocation in the validator's emission config receive weight. The "Other" category is never rewarded.

**NOTE**: Currently we only validate GPUs with CUDA version 12.8 or higher.

### Current Default Allocation

Based on `validator.toml.example`, the default weight pool distribution is:

- **Burn**: 95% of total weight allocation is sent to a burn address
- **A100**: 45% of remaining (post-burn) weight allocation
- **H100**: 55% of remaining (post-burn) weight allocation

These values are configurable per validator. Additional GPU categories (H200, B200) can be added to the `[emission.gpu_allocations]` config as needed.

## Delivery-Based Weight Model

### How It Works

1. **Rental Revenue Tracking**: When a miner's GPU node is rented, the billing system records a delivery with the `revenue_usd` generated during the rental period.
2. **Delivery Sync**: At each weight-setting epoch, the validator fetches delivery records from the billing API for the current period.
3. **Category Grouping**: Deliveries are grouped by GPU category. Each delivery contributes its `revenue_usd` to the miner's total within that category.
4. **Weight Allocation**: Within each category, weights are distributed proportionally to each miner's share of total revenue in that category.

### Weight Calculation

#### Step 1: Fetch Deliveries

The weight setter syncs delivery records from the billing API for the current epoch window:

```text
deliveries = billing_api.get_miner_delivery(period_start, period_end)
```

Each delivery contains:
- `miner_hotkey` — the miner's identity
- `node_id` — the specific GPU node
- `gpu_category` — the GPU type (e.g., "H100")
- `revenue_usd` — rental revenue generated

#### Step 2: Group by Category

Deliveries are grouped by GPU category. Only deliveries with `revenue_usd > 0` and a recognized GPU category contribute to weight allocation:

```text
For each delivery:
  category = delivery.gpu_category
  miners_by_category[category].push(miner_uid, revenue_usd)
```

#### Step 3: Allocate Category Weights

Each configured GPU category receives a share of the post-burn emission pool:

```text
burn_weight = total_emissions × burn_percentage
remaining = total_emissions - burn_weight

category_pool = remaining × (category_weight / 100.0)
```

If a category has no deliveries (no active rentals), its allocation is added to the burn.

#### Step 4: Distribute Within Category

Within each category, miner weight is proportional to their revenue share:

```text
miner_revenue = SUM(revenue_usd) for all deliveries of this miner in this category
total_category_revenue = SUM(revenue_usd) for all miners in this category

miner_weight = (miner_revenue / total_category_revenue) × category_pool
```

#### Step 5: Final Weight

The final weight for each miner is the sum of their weights across all categories:

```text
final_weight = SUM(miner_weight_in_category) across all categories
```

### Example

```
Configuration:
  burn_percentage = 95%
  A100 allocation = 45%
  H100 allocation = 55%

Total emissions: 65535 (u16::MAX)
Burn (95%): 62258 → burn_uid
Remaining: 3277

A100 pool (45% of remaining): 1475
  Miner 1: $500 revenue → 500/800 = 62.5% → weight 922
  Miner 2: $300 revenue → 300/800 = 37.5% → weight 553

H100 pool (55% of remaining): 1802
  Miner 3: $1000 revenue → 1000/1000 = 100% → weight 1802

If no miners had H100 rentals, the H100 pool (1802) would be added to burn.
```

## Multi-Category Support

A single miner can appear in multiple categories if they operate different GPU types:

- Miners with both A100 and H100 rentals earn weight in both categories
- Revenue is attributed to the specific GPU category of each rental
- Final weight is the sum of weights earned in each category

## Implementation Details

### Delivery Flow

```text
Billing API → WeightSetter.sync_deliveries_for_epoch()
  → Store deliveries locally
  → Group by GPU category
  → WeightAllocationEngine.allocate_weights()
  → Submit to Bittensor chain
```

### Weight Setting Frequency

Weights are set periodically based on the configured `blocks_per_weight_set` parameter. The weight setter:

- Checks the blockchain block every 12 seconds
- Sets weights when the configured number of blocks have passed (default: 360 blocks ≈ 72 minutes)
- Only includes miners whose hotkeys are active on the Bittensor metagraph

### Filtering Criteria

Deliveries are excluded from weight calculation when:

- The miner's hotkey is no longer in the metagraph (deregistered)
- The GPU category is unrecognized (maps to `Other`)
- The `revenue_usd` is zero, negative, or non-finite
- The node is excluded due to collateral requirements
