use crate::domain::types::{BillingPeriod, CostBreakdown, CreditBalance, PackageId, UsageMetrics};
use crate::error::Result;
use basilica_protocol::billing::{
    BillingPackage as ProtoBillingPackage, IncludedResources as ProtoIncludedResources,
    PackageRates as ProtoPackageRates,
};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::str::FromStr;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BillingPackage {
    pub id: PackageId,
    pub name: String,
    pub description: String,
    pub hourly_rate: CreditBalance,
    pub gpu_model: String,
    pub billing_period: BillingPeriod,
    pub priority: u32,
    pub active: bool,
    pub metadata: HashMap<String, String>,

    pub storage_rate_per_gb_hour: CreditBalance,
    pub network_rate_per_gb: CreditBalance,
    pub disk_io_rate_per_gb: CreditBalance,
    pub cpu_rate_per_core_hour: CreditBalance,
    pub memory_rate_per_gb_hour: CreditBalance,

    pub included_storage_gb_hours: Decimal,
    pub included_network_gb: Decimal,
    pub included_disk_io_gb: Decimal,
    pub included_cpu_core_hours: Decimal,
    pub included_memory_gb_hours: Decimal,
}

impl BillingPackage {
    pub fn new(
        id: PackageId,
        name: String,
        description: String,
        hourly_rate: CreditBalance,
        gpu_model: String,
    ) -> Self {
        Self {
            id,
            name,
            description,
            hourly_rate,
            gpu_model,
            billing_period: BillingPeriod::Hourly,
            priority: 100,
            active: true,
            metadata: HashMap::new(),

            storage_rate_per_gb_hour: CreditBalance::zero(),
            network_rate_per_gb: CreditBalance::zero(),
            disk_io_rate_per_gb: CreditBalance::zero(),
            cpu_rate_per_core_hour: CreditBalance::zero(),
            memory_rate_per_gb_hour: CreditBalance::zero(),

            included_storage_gb_hours: Decimal::ZERO,
            included_network_gb: Decimal::ZERO,
            included_disk_io_gb: Decimal::ZERO,
            included_cpu_core_hours: Decimal::ZERO,
            included_memory_gb_hours: Decimal::ZERO,
        }
    }

    /// Calculate cost with GPU count multiplier and volume discount
    ///
    /// Business rule: final_cost = (hourly_rate × gpu_hours × gpu_count × volume_discount_multiplier)
    /// where: volume_discount_multiplier = 0.9 (10% discount) if gpu_count > 1, else 1.0
    ///
    /// NOTE: We only charge for GPU hours. All other resource usage (CPU, memory, network, disk, storage) is NOT billed.
    pub fn calculate_cost_with_gpu_count(
        &self,
        usage: &UsageMetrics,
        gpu_count: u32,
    ) -> CostBreakdown {
        let effective_gpu_count = gpu_count.max(1);
        let gpu_hours = usage.gpu_hours;

        let raw_gpu_cost = self
            .hourly_rate
            .multiply(gpu_hours)
            .multiply(Decimal::from(effective_gpu_count));

        let volume_discount = if gpu_count > 1 {
            raw_gpu_cost.multiply(Decimal::from_str("0.10").unwrap())
        } else {
            CreditBalance::zero()
        };

        let base_cost_after_volume = raw_gpu_cost
            .subtract(volume_discount)
            .unwrap_or(raw_gpu_cost);

        CostBreakdown {
            base_cost: raw_gpu_cost,
            usage_cost: CreditBalance::zero(),
            volume_discount,
            discounts: CreditBalance::zero(),
            overage_charges: CreditBalance::zero(),
            total_cost: base_cost_after_volume,
        }
    }

    /// Calculate cost for given usage (backward compatibility)
    ///
    /// DEPRECATED: Use calculate_cost_with_gpu_count instead.
    /// This method assumes gpu_count from usage metrics or defaults to 1.
    pub fn calculate_cost(&self, usage: &UsageMetrics) -> CostBreakdown {
        self.calculate_cost_with_gpu_count(usage, usage.gpu_count.max(1))
    }

    /// Convert to protobuf format for gRPC
    pub fn to_proto(&self) -> ProtoBillingPackage {
        ProtoBillingPackage {
            package_id: self.id.to_string(),
            name: self.name.clone(),
            description: self.description.clone(),
            rates: Some(ProtoPackageRates {
                cpu_rate_per_hour: "0".to_string(),
                memory_rate_per_gb_hour: "0".to_string(),
                gpu_rates: HashMap::from([(self.gpu_model.clone(), self.hourly_rate.to_string())]),
                network_rate_per_gb: "0".to_string(),
                disk_iops_rate: "0".to_string(),
                base_rate_per_hour: self.hourly_rate.to_string(),
            }),
            included_resources: Some(ProtoIncludedResources {
                cpu_core_hours: 0,
                memory_gb_hours: 0,
                gpu_hours: HashMap::new(),
                network_gb: 0,
                disk_iops: 0,
            }),
            overage_rates: None,
            priority: self.priority,
            is_active: self.active,
        }
    }
}

// Pricing business rules - currently empty as all pricing comes from database
// Package assignment happens automatically via GPU model detection in
// PackageId::from_gpu_model() and find_package_for_gpu_model() repository method

use async_trait::async_trait;

/// Package service for business logic operations
#[async_trait]
pub trait PackageService: Send + Sync {
    /// Evaluate the cost for a package given usage metrics
    async fn evaluate_cost(
        &self,
        package_id: &PackageId,
        usage: &UsageMetrics,
    ) -> Result<CostBreakdown>;

    async fn recommend_package(&self, gpu_model: &str) -> Result<BillingPackage>;

    async fn validate_package_requirements(
        &self,
        package: &BillingPackage,
        gpu_model: &str,
    ) -> Result<bool>;
}

use crate::storage::PackageRepository;
use std::sync::Arc;

pub struct RepositoryPackageService {
    repository: Arc<dyn PackageRepository + Send + Sync>,
}

impl RepositoryPackageService {
    pub fn new(repository: Arc<dyn PackageRepository + Send + Sync>) -> Self {
        Self { repository }
    }
}

#[async_trait]
impl PackageService for RepositoryPackageService {
    async fn evaluate_cost(
        &self,
        package_id: &PackageId,
        usage: &UsageMetrics,
    ) -> Result<CostBreakdown> {
        let package = self.repository.get_package(package_id).await?;
        Ok(package.calculate_cost(usage))
    }

    async fn recommend_package(&self, gpu_model: &str) -> Result<BillingPackage> {
        self.repository.find_package_for_gpu_model(gpu_model).await
    }

    async fn validate_package_requirements(
        &self,
        package: &BillingPackage,
        gpu_model: &str,
    ) -> Result<bool> {
        self.repository
            .is_package_compatible_with_gpu(&package.id, gpu_model)
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_package_creation() {
        let package = BillingPackage::new(
            PackageId::h100(),
            "Test Package".to_string(),
            "Test Description".to_string(),
            CreditBalance::from_f64(10.0).unwrap(),
            "H100".to_string(),
        );

        assert_eq!(package.id, PackageId::h100());
        assert_eq!(package.hourly_rate, CreditBalance::from_f64(10.0).unwrap());
        assert!(package.active);
    }

    #[test]
    fn test_cost_calculation() {
        let package = BillingPackage::new(
            PackageId::h100(),
            "H100 GPU".to_string(),
            "NVIDIA H100 GPU instances".to_string(),
            CreditBalance::from_f64(1.11).unwrap(),
            "H100".to_string(),
        );
        let usage = UsageMetrics {
            gpu_hours: Decimal::from(10),
            gpu_count: 1,
            cpu_hours: Decimal::ZERO,
            memory_gb_hours: Decimal::ZERO,
            storage_gb_hours: Decimal::ZERO,
            network_gb: Decimal::ZERO,
            disk_io_gb: Decimal::ZERO,
        };

        let cost = package.calculate_cost(&usage);
        assert_eq!(cost.total_cost, CreditBalance::from_f64(11.1).unwrap());
    }

    #[test]
    fn test_extras_calculation_with_overages() {
        let mut package = BillingPackage::new(
            PackageId::h100(),
            "H100 GPU".to_string(),
            "NVIDIA H100 GPU instances".to_string(),
            CreditBalance::from_f64(1.11).unwrap(),
            "H100".to_string(),
        );

        package.storage_rate_per_gb_hour = CreditBalance::from_f64(0.10).unwrap();
        package.network_rate_per_gb = CreditBalance::from_f64(0.05).unwrap();
        package.disk_io_rate_per_gb = CreditBalance::from_f64(0.03).unwrap();
        package.cpu_rate_per_core_hour = CreditBalance::from_f64(0.02).unwrap();
        package.memory_rate_per_gb_hour = CreditBalance::from_f64(0.01).unwrap();

        package.included_storage_gb_hours = Decimal::from(100);
        package.included_network_gb = Decimal::from(50);
        package.included_disk_io_gb = Decimal::from(50);
        package.included_cpu_core_hours = Decimal::from(80);
        package.included_memory_gb_hours = Decimal::from(320);

        let usage = UsageMetrics {
            gpu_hours: Decimal::from(10),
            gpu_count: 1,
            cpu_hours: Decimal::from(100),
            memory_gb_hours: Decimal::from(400),
            storage_gb_hours: Decimal::from(150),
            network_gb: Decimal::from(75),
            disk_io_gb: Decimal::from(60),
        };

        let cost = package.calculate_cost(&usage);

        let expected_base = CreditBalance::from_f64(11.1).unwrap();

        assert_eq!(cost.base_cost, expected_base);
        assert_eq!(cost.usage_cost, CreditBalance::zero());
        assert_eq!(cost.total_cost, expected_base);
    }

    #[test]
    fn test_extras_calculation_within_included_allowances() {
        let mut package = BillingPackage::new(
            PackageId::h100(),
            "H100 GPU".to_string(),
            "NVIDIA H100 GPU instances".to_string(),
            CreditBalance::from_f64(1.11).unwrap(),
            "H100".to_string(),
        );

        package.storage_rate_per_gb_hour = CreditBalance::from_f64(0.10).unwrap();
        package.network_rate_per_gb = CreditBalance::from_f64(0.05).unwrap();
        package.disk_io_rate_per_gb = CreditBalance::from_f64(0.03).unwrap();
        package.cpu_rate_per_core_hour = CreditBalance::from_f64(0.02).unwrap();
        package.memory_rate_per_gb_hour = CreditBalance::from_f64(0.01).unwrap();

        package.included_storage_gb_hours = Decimal::from(200);
        package.included_network_gb = Decimal::from(100);
        package.included_disk_io_gb = Decimal::from(100);
        package.included_cpu_core_hours = Decimal::from(100);
        package.included_memory_gb_hours = Decimal::from(500);

        let usage = UsageMetrics {
            gpu_hours: Decimal::from(10),
            gpu_count: 1,
            cpu_hours: Decimal::from(50),
            memory_gb_hours: Decimal::from(300),
            storage_gb_hours: Decimal::from(100),
            network_gb: Decimal::from(50),
            disk_io_gb: Decimal::from(50),
        };

        let cost = package.calculate_cost(&usage);

        let expected_base = CreditBalance::from_f64(11.1).unwrap();
        assert_eq!(cost.base_cost, expected_base);
        assert_eq!(cost.usage_cost, CreditBalance::zero());
        assert_eq!(cost.total_cost, expected_base);
    }

    #[test]
    fn test_extras_calculation_zero_rates() {
        let package = BillingPackage::new(
            PackageId::h100(),
            "H100 GPU".to_string(),
            "NVIDIA H100 GPU instances".to_string(),
            CreditBalance::from_f64(1.11).unwrap(),
            "H100".to_string(),
        );

        let usage = UsageMetrics {
            gpu_hours: Decimal::from(10),
            gpu_count: 1,
            cpu_hours: Decimal::from(100),
            memory_gb_hours: Decimal::from(400),
            storage_gb_hours: Decimal::from(150),
            network_gb: Decimal::from(75),
            disk_io_gb: Decimal::from(60),
        };

        let cost = package.calculate_cost(&usage);

        let expected_base = CreditBalance::from_f64(11.1).unwrap();
        assert_eq!(cost.base_cost, expected_base);
        assert_eq!(cost.usage_cost, CreditBalance::zero());
        assert_eq!(cost.total_cost, expected_base);
    }

    #[test]
    fn test_extras_calculation_negative_prevention() {
        let mut package = BillingPackage::new(
            PackageId::h100(),
            "H100 GPU".to_string(),
            "NVIDIA H100 GPU instances".to_string(),
            CreditBalance::from_f64(1.11).unwrap(),
            "H100".to_string(),
        );

        package.storage_rate_per_gb_hour = CreditBalance::from_f64(0.10).unwrap();
        package.included_storage_gb_hours = Decimal::from(200);

        let usage = UsageMetrics {
            gpu_hours: Decimal::from(10),
            gpu_count: 1,
            cpu_hours: Decimal::ZERO,
            memory_gb_hours: Decimal::ZERO,
            storage_gb_hours: Decimal::from(50),
            network_gb: Decimal::ZERO,
            disk_io_gb: Decimal::ZERO,
        };

        let cost = package.calculate_cost(&usage);

        assert_eq!(cost.usage_cost, CreditBalance::zero());
    }

    #[test]
    fn test_extras_calculation_partial_overages() {
        let mut package = BillingPackage::new(
            PackageId::h100(),
            "H100 GPU".to_string(),
            "NVIDIA H100 GPU instances".to_string(),
            CreditBalance::from_f64(1.11).unwrap(),
            "H100".to_string(),
        );

        package.storage_rate_per_gb_hour = CreditBalance::from_f64(0.10).unwrap();
        package.network_rate_per_gb = CreditBalance::from_f64(0.05).unwrap();
        package.included_storage_gb_hours = Decimal::from(100);
        package.included_network_gb = Decimal::from(50);

        let usage = UsageMetrics {
            gpu_hours: Decimal::from(10),
            gpu_count: 1,
            cpu_hours: Decimal::ZERO,
            memory_gb_hours: Decimal::ZERO,
            storage_gb_hours: Decimal::from(110),
            network_gb: Decimal::from(40),
            disk_io_gb: Decimal::ZERO,
        };

        let cost = package.calculate_cost(&usage);

        let expected_base = CreditBalance::from_f64(11.1).unwrap();

        assert_eq!(cost.base_cost, expected_base);
        assert_eq!(cost.usage_cost, CreditBalance::zero());
        assert_eq!(cost.total_cost, expected_base);
    }

    #[test]
    fn test_extras_calculation_fractional_gpu_hours() {
        use rust_decimal::prelude::FromStr;

        let mut package = BillingPackage::new(
            PackageId::h100(),
            "H100 GPU".to_string(),
            "NVIDIA H100 GPU instances".to_string(),
            CreditBalance::from_f64(1.11).unwrap(),
            "H100".to_string(),
        );

        package.storage_rate_per_gb_hour = CreditBalance::from_f64(0.10).unwrap();
        package.included_storage_gb_hours = Decimal::from(100);

        let usage = UsageMetrics {
            gpu_hours: Decimal::from_str("0.5").unwrap(),
            gpu_count: 1,
            cpu_hours: Decimal::ZERO,
            memory_gb_hours: Decimal::ZERO,
            storage_gb_hours: Decimal::from(110),
            network_gb: Decimal::ZERO,
            disk_io_gb: Decimal::ZERO,
        };

        let cost = package.calculate_cost(&usage);

        let expected_base = CreditBalance::from_f64(0.555).unwrap();
        assert_eq!(cost.base_cost, expected_base);
    }

    #[test]
    fn test_single_gpu_no_volume_discount() {
        let package = BillingPackage::new(
            PackageId::h100(),
            "H100 GPU".to_string(),
            "NVIDIA H100".to_string(),
            CreditBalance::from_f64(1.11).unwrap(),
            "H100".to_string(),
        );

        let mut usage = UsageMetrics::zero();
        usage.gpu_hours = Decimal::from(1);
        usage.gpu_count = 1;

        let breakdown = package.calculate_cost_with_gpu_count(&usage, 1);

        assert_eq!(breakdown.volume_discount, CreditBalance::zero());
        assert_eq!(breakdown.base_cost, CreditBalance::from_f64(1.11).unwrap());
        assert_eq!(breakdown.total_cost, CreditBalance::from_f64(1.11).unwrap());
    }

    #[test]
    fn test_two_gpu_volume_discount() {
        let package = BillingPackage::new(
            PackageId::h100(),
            "H100 GPU".to_string(),
            "NVIDIA H100".to_string(),
            CreditBalance::from_f64(1.11).unwrap(),
            "H100".to_string(),
        );

        let mut usage = UsageMetrics::zero();
        usage.gpu_hours = Decimal::from(1);

        let breakdown = package.calculate_cost_with_gpu_count(&usage, 2);

        let expected_raw = CreditBalance::from_f64(2.22).unwrap();
        let expected_discount = CreditBalance::from_f64(0.222).unwrap();

        assert_eq!(breakdown.base_cost, expected_raw);
        assert_eq!(breakdown.volume_discount, expected_discount);
        assert_eq!(
            breakdown.total_cost,
            CreditBalance::from_f64(1.998).unwrap()
        );
    }

    #[test]
    fn test_four_gpu_volume_discount() {
        let package = BillingPackage::new(
            PackageId::h100(),
            "H100 GPU".to_string(),
            "NVIDIA H100".to_string(),
            CreditBalance::from_f64(1.11).unwrap(),
            "H100".to_string(),
        );

        let mut usage = UsageMetrics::zero();
        usage.gpu_hours = Decimal::from(1);

        let breakdown = package.calculate_cost_with_gpu_count(&usage, 4);

        let expected_raw = CreditBalance::from_f64(4.44).unwrap();
        let expected_discount = CreditBalance::from_f64(0.444).unwrap();

        assert_eq!(breakdown.base_cost, expected_raw);
        assert_eq!(breakdown.volume_discount, expected_discount);
        assert_eq!(
            breakdown.total_cost,
            CreditBalance::from_f64(3.996).unwrap()
        );
    }

    #[test]
    fn test_eight_gpu_volume_discount() {
        let package = BillingPackage::new(
            PackageId::h100(),
            "H100 GPU".to_string(),
            "NVIDIA H100".to_string(),
            CreditBalance::from_f64(1.11).unwrap(),
            "H100".to_string(),
        );

        let mut usage = UsageMetrics::zero();
        usage.gpu_hours = Decimal::from(1);

        let breakdown = package.calculate_cost_with_gpu_count(&usage, 8);

        let expected_raw = CreditBalance::from_f64(8.88).unwrap();
        let expected_discount = CreditBalance::from_f64(0.888).unwrap();

        assert_eq!(breakdown.base_cost, expected_raw);
        assert_eq!(breakdown.volume_discount, expected_discount);
        assert_eq!(
            breakdown.total_cost,
            CreditBalance::from_f64(7.992).unwrap()
        );
    }

    #[test]
    fn test_multi_gpu_with_extras_cost() {
        let mut package = BillingPackage::new(
            PackageId::h100(),
            "H100 GPU".to_string(),
            "NVIDIA H100".to_string(),
            CreditBalance::from_f64(1.11).unwrap(),
            "H100".to_string(),
        );

        package.storage_rate_per_gb_hour = CreditBalance::from_f64(0.10).unwrap();
        package.included_storage_gb_hours = Decimal::from(100);

        let mut usage = UsageMetrics::zero();
        usage.gpu_hours = Decimal::from(1);
        usage.storage_gb_hours = Decimal::from(150);

        let breakdown = package.calculate_cost_with_gpu_count(&usage, 4);

        let expected_raw = CreditBalance::from_f64(4.44).unwrap();
        let expected_discount = CreditBalance::from_f64(0.444).unwrap();
        let expected_total = CreditBalance::from_f64(3.996).unwrap();

        assert_eq!(breakdown.base_cost, expected_raw);
        assert_eq!(breakdown.volume_discount, expected_discount);
        assert_eq!(breakdown.usage_cost, CreditBalance::zero());
        assert_eq!(breakdown.total_cost, expected_total);
    }

    #[test]
    fn test_zero_gpu_count() {
        let package = BillingPackage::new(
            PackageId::h100(),
            "H100 GPU".to_string(),
            "NVIDIA H100".to_string(),
            CreditBalance::from_f64(1.11).unwrap(),
            "H100".to_string(),
        );

        let mut usage = UsageMetrics::zero();
        usage.gpu_hours = Decimal::from(1);

        let breakdown = package.calculate_cost_with_gpu_count(&usage, 0);

        assert_eq!(breakdown.volume_discount, CreditBalance::zero());
        assert_eq!(breakdown.base_cost, CreditBalance::from_f64(1.11).unwrap());
    }

    #[test]
    fn test_no_double_counting() {
        let package = BillingPackage::new(
            PackageId::h100(),
            "H100 GPU".to_string(),
            "NVIDIA H100".to_string(),
            CreditBalance::from_f64(1.11).unwrap(),
            "H100".to_string(),
        );

        let mut usage = UsageMetrics::zero();
        usage.gpu_hours = Decimal::from(1);

        let breakdown_8gpu = package.calculate_cost_with_gpu_count(&usage, 8);
        let breakdown_1gpu = package.calculate_cost_with_gpu_count(&usage, 1);

        let expected_raw_8gpu = CreditBalance::from_f64(8.88).unwrap();
        let expected_discount_8gpu = CreditBalance::from_f64(0.888).unwrap();

        assert_eq!(breakdown_8gpu.base_cost, expected_raw_8gpu);
        assert_eq!(breakdown_8gpu.volume_discount, expected_discount_8gpu);
        assert_eq!(breakdown_1gpu.volume_discount, CreditBalance::zero());
    }

    #[test]
    fn test_extras_calculation_realistic_scenario() {
        let mut package = BillingPackage::new(
            PackageId::h100(),
            "H100 GPU".to_string(),
            "NVIDIA H100 GPU instances with extras".to_string(),
            CreditBalance::from_f64(1.11).unwrap(),
            "H100".to_string(),
        );

        package.storage_rate_per_gb_hour = CreditBalance::from_f64(0.10).unwrap();
        package.network_rate_per_gb = CreditBalance::from_f64(0.05).unwrap();
        package.disk_io_rate_per_gb = CreditBalance::from_f64(0.03).unwrap();
        package.cpu_rate_per_core_hour = CreditBalance::from_f64(0.02).unwrap();
        package.memory_rate_per_gb_hour = CreditBalance::from_f64(0.01).unwrap();

        package.included_storage_gb_hours = Decimal::from(1000);
        package.included_network_gb = Decimal::from(500);
        package.included_disk_io_gb = Decimal::from(500);
        package.included_cpu_core_hours = Decimal::from(800);
        package.included_memory_gb_hours = Decimal::from(3200);

        let usage = UsageMetrics {
            gpu_hours: Decimal::from(100),
            gpu_count: 1,
            cpu_hours: Decimal::from(850),
            memory_gb_hours: Decimal::from(3500),
            storage_gb_hours: Decimal::from(1200),
            network_gb: Decimal::from(600),
            disk_io_gb: Decimal::from(550),
        };

        let cost = package.calculate_cost(&usage);

        let expected_base = CreditBalance::from_f64(111.0).unwrap();

        assert_eq!(cost.base_cost, expected_base);
        assert_eq!(cost.usage_cost, CreditBalance::zero());
        assert_eq!(cost.total_cost, expected_base);
    }
}
