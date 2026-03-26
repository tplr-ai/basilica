# Miner Bidding Strategy

## Overview

Miners register nodes with validators and attach a per-GPU hourly price. Validators store those registrations and select the lowest-priced eligible node for rentals. This document outlines how miners configure and submit those prices.

## Current State

### ✅ Implemented

1. **Registration client** (`registration_client.rs`):
   - `RegisterBid` submits node registrations (SSH details + `hourly_rate_cents`)
   - `HealthCheck` keeps registrations active
   - `UpdateBid` / `RemoveBid` RPCs exist for price updates and removal

2. **Bid manager** (`bidding.rs`):
   - Validates every node’s `gpu_category` has a configured static price
   - Builds `NodeRegistration` entries from node config + static prices
   - Registers nodes once and sends periodic health checks

3. **Configuration** (`config.rs`):
   - `BiddingConfig` with a single `static` strategy
   - Prices are set in dollars and converted to cents on load
   - `NodeConfig` includes `gpu_category` and `gpu_count` for each node

### ❌ Not Yet Implemented

- Dynamic pricing strategies (cost-plus, utilization, time-of-day, hybrid)
- Automatic price updates via `UpdateBid`
- Market intelligence (watch competitor bids)

## Bid Lifecycle

```
┌─────────────────────────────────────────────────────────────────────┐
│                        MINER BIDDING FLOW                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐    ┌──────────────────────┐    ┌────────────────┐  │
│  │ config.toml  │───▶│ BidManager           │───▶│ RegisterBid     │  │
│  │ static prices│    │ - validate prices    │    │ nodes + prices  │  │
│  └──────────────┘    │ - build registrations│    └──────┬─────────┘  │
│                      └──────────────────────┘           │            │
│                                                        ▼            │
│                                          ┌──────────────────────┐  │
│                                          │ Validator stores     │  │
│                                          │ registered nodes     │  │
│                                          └──────────┬───────────┘  │
│                                                     │              │
│                                                     ▼              │
│                                          ┌──────────────────────┐  │
│                                          │ HealthCheck loop     │  │
│                                          │ keeps nodes active   │  │
│                                          └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

## Bidding Strategy Options

### Option 1: Static Config (Simple)

Miner sets fixed prices per GPU category in config file.

```toml
[bidding.strategy.static.static_prices]
# Fixed prices per GPU-hour by category
H100 = 2.50
A100 = 1.20
RTX4090 = 0.80
```

**Pros:**
- Simple to understand and configure
- Predictable revenue
- No risk of algorithmic errors

**Cons:**
- Can't adapt to market conditions
- May leave money on table or never win bids

**Best for:** Miners who know their costs and want predictable operation.

---

### Option 2: Cost-Plus Pricing (Future)

Not implemented yet. Planned to derive bids from electricity, depreciation, and overhead costs plus a margin.

---

### Option 3: Utilization-Based (Future)

Not implemented yet. Planned to adjust bids based on current capacity utilization.

---

### Option 4: Time-of-Day Pricing (Future)

Not implemented yet. Planned to adjust prices by time-of-day to reflect energy costs.

---

### Option 5: Hybrid Strategy (Future)

Not implemented yet. Planned to combine multiple signals (utilization, cost basis, time-of-day) into a single price.

---

## Implementation Roadmap

### Phase 1: Static Pricing + Registration ✅ IMPLEMENTED

- BidManager validates static prices for all nodes
- Registers nodes with validator via `RegisterBid`

### Phase 2: Health Checks ✅ IMPLEMENTED

- Miner sends periodic `HealthCheck` at validator-provided intervals
- Validator uses freshness to keep nodes eligible

### Phase 3: Price Updates (Future)

- Use `UpdateBid` to change `hourly_rate_cents` when strategies change

### Phase 4: Dynamic Strategies (Future)

- Cost-plus, utilization, time-of-day, hybrid

---

## Economic Considerations

### The Bidding Tradeoff

```
                    WIN PROBABILITY
                          ▲
                          │
            High ─────────┼─────────● Low bid
                          │        /
                          │       /
                          │      /
                          │     /
                          │    /
                          │   /
            Low ──────────┼──● High bid
                          │
                          └───────────────────▶
                              REWARD PER WIN
```

- **Low bids**: High probability of winning, but lower reward per rental
- **High bids**: Low probability of winning, but higher reward when you do win

**Optimal strategy**: Find the bid that maximizes expected revenue:
```
Expected Revenue = P(win) × Revenue(bid)
```

### Race to Bottom Risk

If all miners use aggressive utilization-based bidding, prices can spiral down. Mitigations:

1. **Floor price (future strategy)**: Never bid below cost
2. **Collateral**: Miners with stake have skin in the game
3. **Reputation**: Long-term miners optimize for sustainable pricing
4. **Category caps**: Emission caps prevent single miner domination

### New Miner Strategy

New miners face a cold-start problem: no track record → validators may prefer established miners.

**Suggested approach:**
1. Start with slightly below-market bids to win initial work
2. Build reputation through reliable delivery
3. Gradually increase prices as reputation grows

---

## Configuration Examples

### Basic Static Bidding (Currently Supported)

```toml
# config.toml - Add this section to enable auto-bidding

# Set your prices per GPU-hour for each category
[bidding.strategy.static.static_prices]
H100 = 2.50
A100 = 1.20
RTX4090 = 0.80
```

### Node Configuration with GPU Info

```toml
# Each node needs gpu_category and gpu_count

[[node_management.nodes]]
host = "192.168.1.100"
port = 22
username = "basilica"
gpu_category = "H100"        # Must match bidding.strategy.static.static_prices key
gpu_count = 8

[[node_management.nodes]]
host = "192.168.1.101"
port = 22
username = "basilica"
gpu_category = "A100"
gpu_count = 4
```

### Future: Cost-Plus Strategy (Not Yet Implemented)

Not implemented yet. Placeholder for a cost-based pricing strategy.

### Future: Hybrid Strategy (Not Yet Implemented)

Not implemented yet. Placeholder for combining multiple pricing signals.

---

## Monitoring & Alerts

Track bidding effectiveness:

```rust
// Metrics to expose
struct BiddingMetrics {
    registrations_submitted: Counter,
    registrations_accepted: Counter,
    registrations_rejected: Counter,
    current_bid_price: Gauge,        // by category
    nodes_active: Gauge,             // by category
    rentals_won: Counter,
    revenue_earned: Counter,
}
```

**Alerts to configure:**
- Nodes marked inactive → check health check connectivity
- Win rate dropping → competitors undercutting
- Registrations being rejected → check GPU categories and pricing config

---

## Summary

| Strategy | Complexity | Adaptability | Risk | Best For |
|----------|------------|--------------|------|----------|
| Static | Low | None | Over/under pricing | Simple operations |
| Cost-Plus (future) | - | - | Not implemented | - |
| Utilization (future) | - | - | Not implemented | - |
| Time-of-Day (future) | - | - | Not implemented | - |
| Hybrid (future) | - | - | Not implemented | - |

**Recommended starting point**: Static config with clear per-GPU pricing, then revisit dynamic strategies when they land.
