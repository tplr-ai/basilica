# Scoring and Weight Setting in Basilica

This document explains how the Basilica validator scores miners and sets weights on the Bittensor network.

## Overview

The validator uses a GPU-based scoring system that evaluates miners based on their hardware capabilities and validation success rates. Only H100 and H200 GPUs are eligible for rewards, while other GPU types are excluded from weight distribution.

## GPU Categories

The system recognizes three GPU categories, but only two are eligible for rewards:

- **H100** - NVIDIA H100 GPUs
- **H200** - NVIDIA H200 GPUs
- **OTHER** - All other GPU types

**NOTE**: The "OTHER" category is not rewarded. Only miners with H100 or H200 GPUs are considered for weight allocation.

**NOTE**: Currently we only validate GPUs with CUDA version 12.8 or higher.

### Current Allocation

Based on the emission configuration, the weight pool is distributed as follows:

- **Burn**: 95% of the total weight allocation is sent to a burn address to maintain network economics.
- **H100**: 40% of total **available** weight allocation
- **H200**: 60% of total **available** weight allocation

## Scoring Formula

### For Each Miner

The base score is calculated as:

```text
validation_ratio = successful_validations / total_validations
```

This ratio represents the miner's availability and reliability. High availability makes miners rank higher.

### For Each Miner in a Category

Within each GPU category, the miner's score is weighted by their GPU count:

```text
category_score = validation_ratio × gpu_count
```

The GPU count is aggregated across all machines the miner operates within that category.

### Category Competition

For each category C, the total score is:

```text
total_category_score = SUM(validation_ratio_i × gpu_count_i) for all miners i in C
```

Miners compete locally within their category. The more populated a category is, the more competition exists for that category's weight pool.

### Weight Distribution Within Category

Each miner's weight within their category is proportional to their contribution:

```text
miner_weight_in_category = (category_score / total_category_score) × category_weight_pool
```

### Final Miner Weight

The final weight for each miner is the sum of their weights across all categories:

```text
final_weight = SUM(miner_weight_in_category) across all categories
```

## Implementation Details

### Validation Process

1. **Miner Discovery**: Validators discover miners from the Bittensor metagraph
2. **SSH Endpoint Discovery**: Validators query miners via gRPC for GPU node SSH endpoints
3. **Direct SSH Verification**: Validators SSH directly to GPU nodes for hardware verification
4. **Score Calculation**: Based on validation success and GPU specifications

### GPU Profile Updates

The system maintains GPU profiles for each miner that track:

- Primary GPU model
- GPU count distribution across models
- Total validation score
- Verification count
- Last update timestamp

### Weight Setting Frequency

Weights are set periodically based on the configured `blocks_per_weight_set` parameter. The weight setter:

- Sets weights when 360 blocks have passed
- Only includes miners with active axons on the chain and GPU nodes that passed verification

### Filtering Criteria

Miners must meet several criteria to receive weights:

- Have GPU nodes that passed validation within the cutoff time (default: 3 hours)
- Have active axons on the Bittensor network (non-zero IP and port)
- Own H100 or H200 GPUs (OTHER category GPUs are excluded)

## Multi-Category Support

A single miner can appear in multiple categories if they operate different GPU types:

- Miners with both H100 and H200 GPUs compete in both categories
- Scores are calculated proportionally based on GPU distribution
- Final weight is the sum of weights earned in each category
