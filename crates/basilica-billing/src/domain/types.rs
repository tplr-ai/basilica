use chrono::{DateTime, Duration, Utc};
use rust_decimal::prelude::FromPrimitive;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;
use uuid::Uuid;

/// User identifier (from Auth0)
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct UserId(String);

impl UserId {
    pub fn new(id: String) -> Self {
        Self(id)
    }

    pub fn from_uuid(uuid: Uuid) -> Self {
        Self(uuid.to_string())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn as_uuid(&self) -> Result<Uuid, uuid::Error> {
        Uuid::parse_str(&self.0)
    }
}

impl fmt::Display for UserId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Rental identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RentalId(Uuid);

impl RentalId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    pub fn from_uuid(uuid: Uuid) -> Self {
        Self(uuid)
    }

    pub fn as_uuid(&self) -> Uuid {
        self.0
    }
}

impl Default for RentalId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for RentalId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl FromStr for RentalId {
    type Err = uuid::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let uuid_str = s.strip_prefix("rental-").unwrap_or(s);
        Ok(Self(Uuid::parse_str(uuid_str)?))
    }
}

/// Credit reservation identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ReservationId(Uuid);

impl ReservationId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    pub fn from_uuid(uuid: Uuid) -> Self {
        Self(uuid)
    }

    pub fn as_uuid(&self) -> Uuid {
        self.0
    }
}

impl Default for ReservationId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for ReservationId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Billing package identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PackageId(String);

impl PackageId {
    pub fn new(id: String) -> Self {
        Self(id)
    }

    pub fn a100() -> Self {
        Self("a100".to_string())
    }

    pub fn h100() -> Self {
        Self("h100".to_string())
    }

    pub fn h200() -> Self {
        Self("h200".to_string())
    }

    pub fn h800() -> Self {
        Self("h800".to_string())
    }

    pub fn b200() -> Self {
        Self("b200".to_string())
    }

    pub fn rtx_5090() -> Self {
        Self("rtx_5090".to_string())
    }

    pub fn rtx_4090() -> Self {
        Self("rtx_4090".to_string())
    }

    pub fn rtx_ada_6000() -> Self {
        Self("rtx_ada_6000".to_string())
    }

    pub fn rtx_ada_4000() -> Self {
        Self("rtx_ada_4000".to_string())
    }

    pub fn rtx_ada_2000() -> Self {
        Self("rtx_ada_2000".to_string())
    }

    pub fn l40s() -> Self {
        Self("l40s".to_string())
    }

    pub fn l40() -> Self {
        Self("l40".to_string())
    }

    pub fn l4() -> Self {
        Self("l4".to_string())
    }

    pub fn rtx_a6000() -> Self {
        Self("rtx_a6000".to_string())
    }

    pub fn rtx_a5000() -> Self {
        Self("rtx_a5000".to_string())
    }

    pub fn rtx_a4500() -> Self {
        Self("rtx_a4500".to_string())
    }

    pub fn rtx_a4000() -> Self {
        Self("rtx_a4000".to_string())
    }

    pub fn a40() -> Self {
        Self("a40".to_string())
    }

    pub fn a30() -> Self {
        Self("a30".to_string())
    }

    pub fn rtx_3090() -> Self {
        Self("rtx_3090".to_string())
    }

    pub fn custom() -> Self {
        Self("custom".to_string())
    }

    // Legacy aliases for backward compatibility
    pub fn standard() -> Self {
        Self::h100()
    }

    pub fn premium() -> Self {
        Self::h200()
    }

    pub fn enterprise() -> Self {
        Self::custom()
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn from_gpu_model(gpu_model: &str) -> Self {
        let model_lower = gpu_model.to_lowercase();

        if model_lower.contains("b200") {
            Self::b200()
        } else if model_lower.contains("h200") {
            Self::h200()
        } else if model_lower.contains("h800") {
            Self::h800()
        } else if model_lower.contains("h100") {
            Self::h100()
        } else if model_lower.contains("a100") {
            Self::a100()
        } else if model_lower.contains("a40") {
            Self::a40()
        } else if model_lower.contains("a30") {
            Self::a30()
        } else if model_lower.contains("rtx 5090") || model_lower.contains("geforce rtx 5090") {
            Self::rtx_5090()
        } else if model_lower.contains("rtx 4090") || model_lower.contains("geforce rtx 4090") {
            Self::rtx_4090()
        } else if model_lower.contains("rtx 6000 ada") {
            Self::rtx_ada_6000()
        } else if model_lower.contains("rtx 4000 ada") {
            Self::rtx_ada_4000()
        } else if model_lower.contains("rtx 2000 ada") {
            Self::rtx_ada_2000()
        } else if model_lower.contains("l40s") {
            Self::l40s()
        } else if model_lower.contains("l40") {
            Self::l40()
        } else if model_lower.contains("l4") {
            Self::l4()
        } else if model_lower.contains("rtx a6000") {
            Self::rtx_a6000()
        } else if model_lower.contains("rtx a5000") {
            Self::rtx_a5000()
        } else if model_lower.contains("rtx a4500") {
            Self::rtx_a4500()
        } else if model_lower.contains("rtx a4000") {
            Self::rtx_a4000()
        } else if model_lower.contains("rtx 3090") || model_lower.contains("geforce rtx 3090") {
            Self::rtx_3090()
        } else {
            Self::custom()
        }
    }
}

impl fmt::Display for PackageId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Credit balance with precision handling
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct CreditBalance(Decimal);

impl CreditBalance {
    pub fn zero() -> Self {
        Self(Decimal::ZERO)
    }

    pub fn from_decimal(amount: Decimal) -> Self {
        Self(amount.round_dp(6)) // 6 decimal places for micro-credits
    }

    pub fn from_f64(amount: f64) -> Option<Self> {
        Decimal::from_f64(amount).map(|d| Self(d.round_dp(6)))
    }

    pub fn as_decimal(&self) -> Decimal {
        self.0
    }

    pub fn add(&self, other: CreditBalance) -> Self {
        Self::from_decimal(self.0 + other.0)
    }

    pub fn subtract(&self, other: CreditBalance) -> Option<Self> {
        if self.0 >= other.0 {
            Some(Self::from_decimal(self.0 - other.0))
        } else {
            None
        }
    }

    pub fn multiply(&self, factor: Decimal) -> Self {
        Self::from_decimal(self.0 * factor)
    }

    pub fn is_sufficient(&self, required: CreditBalance) -> bool {
        self.0 >= required.0
    }
}

impl fmt::Display for CreditBalance {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Rental lifecycle states
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RentalState {
    Pending,
    Active,
    Suspended,
    Terminating,
    Completed,
    Failed,
}

impl RentalState {
    pub fn is_active(&self) -> bool {
        matches!(
            self,
            RentalState::Active | RentalState::Suspended | RentalState::Pending
        )
    }

    pub fn is_terminal(&self) -> bool {
        matches!(self, RentalState::Completed | RentalState::Failed)
    }

    pub fn can_transition_to(&self, next: RentalState) -> bool {
        matches!(
            (self, next),
            (RentalState::Pending, RentalState::Active)
                | (RentalState::Pending, RentalState::Failed)
                | (RentalState::Active, RentalState::Suspended)
                | (RentalState::Active, RentalState::Terminating)
                | (RentalState::Suspended, RentalState::Active)
                | (RentalState::Suspended, RentalState::Terminating)
                | (RentalState::Terminating, RentalState::Completed)
                | (RentalState::Terminating, RentalState::Failed)
        )
    }
}

impl fmt::Display for RentalState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RentalState::Pending => write!(f, "pending"),
            RentalState::Active => write!(f, "active"),
            RentalState::Suspended => write!(f, "suspended"),
            RentalState::Terminating => write!(f, "terminating"),
            RentalState::Completed => write!(f, "completed"),
            RentalState::Failed => write!(f, "failed"),
        }
    }
}

/// Billing period for usage calculations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BillingPeriod {
    Hourly,
    Daily,
    Weekly,
    Monthly,
}

impl BillingPeriod {
    pub fn duration(&self) -> Duration {
        match self {
            BillingPeriod::Hourly => Duration::hours(1),
            BillingPeriod::Daily => Duration::days(1),
            BillingPeriod::Weekly => Duration::weeks(1),
            BillingPeriod::Monthly => Duration::days(30), // Approximate
        }
    }

    pub fn calculate_periods(&self, start: DateTime<Utc>, end: DateTime<Utc>) -> u64 {
        let duration = end - start;
        let period_duration = self.duration();
        ((duration.num_seconds() as f64 / period_duration.num_seconds() as f64).ceil()) as u64
    }
}

/// GPU specification details
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GpuSpec {
    pub model: String,
    pub memory_mb: u64,
    pub count: u32,
}

/// Resource specifications for rental
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResourceSpec {
    pub gpu_specs: Vec<GpuSpec>,
    pub cpu_cores: u32,
    pub memory_gb: u32,
    pub storage_gb: u32,
    pub disk_iops: u64,
    pub network_bandwidth_mbps: u64,
}

/// Usage metrics for billing calculations
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct UsageMetrics {
    pub gpu_hours: Decimal,
    pub gpu_count: u32,
    pub cpu_hours: Decimal,
    pub memory_gb_hours: Decimal,
    pub storage_gb_hours: Decimal,
    pub network_gb: Decimal,
    pub disk_io_gb: Decimal,
}

impl UsageMetrics {
    pub fn zero() -> Self {
        Self {
            gpu_hours: Decimal::ZERO,
            gpu_count: 0,
            cpu_hours: Decimal::ZERO,
            memory_gb_hours: Decimal::ZERO,
            storage_gb_hours: Decimal::ZERO,
            network_gb: Decimal::ZERO,
            disk_io_gb: Decimal::ZERO,
        }
    }

    pub fn add(&self, other: &UsageMetrics) -> Self {
        Self {
            gpu_hours: self.gpu_hours + other.gpu_hours,
            gpu_count: self.gpu_count.max(other.gpu_count),
            cpu_hours: self.cpu_hours + other.cpu_hours,
            memory_gb_hours: self.memory_gb_hours + other.memory_gb_hours,
            storage_gb_hours: self.storage_gb_hours + other.storage_gb_hours,
            network_gb: self.network_gb + other.network_gb,
            disk_io_gb: self.disk_io_gb + other.disk_io_gb,
        }
    }
}

/// Cost breakdown for transparency
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CostBreakdown {
    pub base_cost: CreditBalance,
    pub usage_cost: CreditBalance,
    pub volume_discount: CreditBalance,
    pub discounts: CreditBalance,
    pub overage_charges: CreditBalance,
    pub total_cost: CreditBalance,
}

impl CostBreakdown {
    pub fn calculate_total(&self) -> CreditBalance {
        let subtotal = self
            .base_cost
            .add(self.usage_cost)
            .add(self.overage_charges);
        subtotal
            .subtract(self.volume_discount)
            .and_then(|after_volume| after_volume.subtract(self.discounts))
            .unwrap_or(CreditBalance::zero())
    }
}

/// User tier for discount eligibility
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UserTier {
    Standard,
    Student,
    Enterprise,
    Custom,
}

impl UserTier {
    pub fn default_discount_percentage(&self) -> Option<Decimal> {
        match self {
            UserTier::Standard => None,
            UserTier::Student => Decimal::from_str("0.20").ok(),
            UserTier::Enterprise => Decimal::from_str("0.15").ok(),
            UserTier::Custom => None,
        }
    }
}

impl fmt::Display for UserTier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UserTier::Standard => write!(f, "standard"),
            UserTier::Student => write!(f, "student"),
            UserTier::Enterprise => write!(f, "enterprise"),
            UserTier::Custom => write!(f, "custom"),
        }
    }
}

impl FromStr for UserTier {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "standard" => Ok(UserTier::Standard),
            "student" => Ok(UserTier::Student),
            "enterprise" => Ok(UserTier::Enterprise),
            "custom" => Ok(UserTier::Custom),
            _ => Err(format!("Invalid user tier: {}", s)),
        }
    }
}

/// User metadata for pricing and discounts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserMetadata {
    pub user_id: UserId,
    pub user_tier: UserTier,
    pub discount_percentage: Option<Decimal>,
    pub promo_codes: Vec<String>,
    pub tier_updated_at: DateTime<Utc>,
    pub custom_attributes: std::collections::HashMap<String, String>,
}

impl UserMetadata {
    pub fn effective_discount_percentage(&self) -> Decimal {
        self.discount_percentage
            .or_else(|| self.user_tier.default_discount_percentage())
            .unwrap_or(Decimal::ZERO)
    }
}

/// Discount type for promotional codes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiscountType {
    Percentage,
    FixedAmount,
}

impl fmt::Display for DiscountType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DiscountType::Percentage => write!(f, "percentage"),
            DiscountType::FixedAmount => write!(f, "fixed_amount"),
        }
    }
}

impl FromStr for DiscountType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "percentage" => Ok(DiscountType::Percentage),
            "fixed_amount" | "fixedamount" => Ok(DiscountType::FixedAmount),
            _ => Err(format!("Invalid discount type: {}", s)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_credit_balance_arithmetic() {
        let balance1 = CreditBalance::from_f64(100.5).unwrap();
        let balance2 = CreditBalance::from_f64(50.25).unwrap();

        let sum = balance1.add(balance2);
        assert_eq!(sum.as_decimal(), Decimal::from_str("150.75").unwrap());

        let diff = balance1.subtract(balance2).unwrap();
        assert_eq!(diff.as_decimal(), Decimal::from_str("50.25").unwrap());

        assert!(balance2.subtract(balance1).is_none());
    }

    #[test]
    fn test_rental_state_transitions() {
        assert!(RentalState::Pending.can_transition_to(RentalState::Active));
        assert!(RentalState::Active.can_transition_to(RentalState::Suspended));
        assert!(RentalState::Active.can_transition_to(RentalState::Terminating));
        assert!(!RentalState::Completed.can_transition_to(RentalState::Active));
        assert!(!RentalState::Active.can_transition_to(RentalState::Pending));
    }

    #[test]
    fn test_billing_period_calculations() {
        let start = Utc::now();
        let end = start + Duration::hours(25);

        assert_eq!(BillingPeriod::Hourly.calculate_periods(start, end), 25);
        assert_eq!(BillingPeriod::Daily.calculate_periods(start, end), 2);
    }

    #[test]
    fn test_rental_id_from_str_plain_uuid() {
        let uuid_str = "550e8400-e29b-41d4-a716-446655440000";
        let rental_id = RentalId::from_str(uuid_str).unwrap();
        assert_eq!(rental_id.to_string(), uuid_str);
    }

    #[test]
    fn test_rental_id_from_str_with_prefix() {
        let prefixed_str = "rental-550e8400-e29b-41d4-a716-446655440000";
        let expected_uuid = "550e8400-e29b-41d4-a716-446655440000";
        let rental_id = RentalId::from_str(prefixed_str).unwrap();
        assert_eq!(rental_id.to_string(), expected_uuid);
    }

    #[test]
    fn test_rental_id_from_str_invalid() {
        let invalid_str = "not-a-uuid";
        assert!(RentalId::from_str(invalid_str).is_err());
    }

    #[test]
    fn test_rental_id_from_str_invalid_with_prefix() {
        let invalid_str = "rental-not-a-uuid";
        assert!(RentalId::from_str(invalid_str).is_err());
    }
}
