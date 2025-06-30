//! Virtual machine and container detection
//!
//! This module detects various virtualization technologies to ensure
//! attestation is running on bare metal hardware.

use anyhow::Result;
use std::fs;
use std::path::Path;
use std::process::Command;

#[derive(Debug, Clone)]
pub struct VirtualizationDetector;

impl VirtualizationDetector {
    pub fn new() -> Self {
        Self
    }

    /// Comprehensive virtualization detection
    pub fn detect_virtualization(&self) -> Result<VirtualizationStatus> {
        let hypervisor = self.detect_hypervisor()?;
        let container = self.detect_container()?;
        let cpu_flags = self.check_cpu_virtualization_flags()?;
        let dmi_vendor = self.check_dmi_vendor()?;
        let kernel_modules = self.check_virtualization_modules()?;
        let network_interfaces = self.check_virtual_network_interfaces()?;
        let disk_devices = self.check_virtual_disk_devices()?;

        let is_virtualized = hypervisor.is_some()
            || container.is_some()
            || !cpu_flags.is_empty()
            || dmi_vendor.is_some()
            || !kernel_modules.is_empty()
            || !network_interfaces.is_empty()
            || !disk_devices.is_empty();

        let confidence = self.calculate_confidence(&hypervisor, &container, &cpu_flags);

        Ok(VirtualizationStatus {
            is_virtualized,
            hypervisor,
            container,
            cpu_flags,
            dmi_vendor,
            kernel_modules,
            network_interfaces,
            disk_devices,
            confidence,
        })
    }

    /// Detect hypervisor using multiple methods
    fn detect_hypervisor(&self) -> Result<Option<HypervisorType>> {
        // Method 1: Check systemd-detect-virt
        if let Ok(output) = Command::new("systemd-detect-virt").arg("-v").output() {
            let virt_type = String::from_utf8_lossy(&output.stdout).trim().to_string();
            match virt_type.as_str() {
                "kvm" => return Ok(Some(HypervisorType::Kvm)),
                "qemu" => return Ok(Some(HypervisorType::Qemu)),
                "vmware" => return Ok(Some(HypervisorType::VmWare)),
                "microsoft" | "hyperv" => return Ok(Some(HypervisorType::HyperV)),
                "xen" => return Ok(Some(HypervisorType::Xen)),
                "virtualbox" | "oracle" => return Ok(Some(HypervisorType::VirtualBox)),
                "none" => {}
                _ => return Ok(Some(HypervisorType::Unknown(virt_type))),
            }
        }

        // Method 2: Check /proc/cpuinfo for hypervisor flag
        if let Ok(cpuinfo) = fs::read_to_string("/proc/cpuinfo") {
            if cpuinfo.contains("hypervisor") {
                return Ok(Some(HypervisorType::Unknown("generic".to_string())));
            }
        }

        // Method 3: Check /sys/hypervisor/type
        if let Ok(hyp_type) = fs::read_to_string("/sys/hypervisor/type") {
            let hyp_type = hyp_type.trim();
            match hyp_type {
                "xen" => return Ok(Some(HypervisorType::Xen)),
                _ => return Ok(Some(HypervisorType::Unknown(hyp_type.to_string()))),
            }
        }

        // Method 4: Check DMI information
        if let Ok(product_name) = fs::read_to_string("/sys/class/dmi/id/product_name") {
            let product = product_name.trim().to_lowercase();
            if product.contains("vmware") {
                return Ok(Some(HypervisorType::VmWare));
            } else if product.contains("virtualbox") {
                return Ok(Some(HypervisorType::VirtualBox));
            } else if product.contains("kvm") {
                return Ok(Some(HypervisorType::Kvm));
            }
        }

        Ok(None)
    }

    /// Detect container runtime
    fn detect_container(&self) -> Result<Option<ContainerType>> {
        // Check for Docker
        if Path::new("/.dockerenv").exists() {
            return Ok(Some(ContainerType::Docker));
        }

        // Check for Kubernetes
        if std::env::var("KUBERNETES_SERVICE_HOST").is_ok() {
            return Ok(Some(ContainerType::Kubernetes));
        }

        // Check cgroup for container signatures
        if let Ok(cgroup) = fs::read_to_string("/proc/self/cgroup") {
            if cgroup.contains("/docker/") || cgroup.contains("/docker-") {
                return Ok(Some(ContainerType::Docker));
            } else if cgroup.contains("/kubepods/") {
                return Ok(Some(ContainerType::Kubernetes));
            } else if cgroup.contains("/lxc/") {
                return Ok(Some(ContainerType::Lxc));
            } else if cgroup.contains("/machine.slice/") && cgroup.contains("systemd") {
                return Ok(Some(ContainerType::SystemdNspawn));
            }
        }

        // Check for containerd
        if let Ok(mountinfo) = fs::read_to_string("/proc/self/mountinfo") {
            if mountinfo.contains("containerd") {
                return Ok(Some(ContainerType::Containerd));
            }
        }

        Ok(None)
    }

    /// Check CPU flags that indicate virtualization
    fn check_cpu_virtualization_flags(&self) -> Result<Vec<String>> {
        let mut flags = Vec::new();

        if let Ok(cpuinfo) = fs::read_to_string("/proc/cpuinfo") {
            for line in cpuinfo.lines() {
                if line.starts_with("flags") {
                    let flag_list = line.split(':').nth(1).unwrap_or("");

                    // Check for virtualization-related flags
                    if flag_list.contains("hypervisor") {
                        flags.push("hypervisor".to_string());
                    }
                    if flag_list.contains("vmx") && !flag_list.contains("svm") {
                        // VMX without SVM might indicate nested virtualization
                        flags.push("vmx_only".to_string());
                    }
                    break;
                }
            }
        }

        Ok(flags)
    }

    /// Check DMI vendor information
    fn check_dmi_vendor(&self) -> Result<Option<String>> {
        let paths = [
            "/sys/class/dmi/id/sys_vendor",
            "/sys/class/dmi/id/board_vendor",
            "/sys/class/dmi/id/bios_vendor",
        ];

        for path in &paths {
            if let Ok(vendor) = fs::read_to_string(path) {
                let vendor = vendor.trim().to_lowercase();
                if vendor.contains("vmware")
                    || vendor.contains("virtualbox")
                    || vendor.contains("qemu")
                    || vendor.contains("kvm")
                    || vendor.contains("xen")
                    || vendor.contains("microsoft corporation")
                    || vendor.contains("innotek gmbh")
                {
                    return Ok(Some(vendor));
                }
            }
        }

        Ok(None)
    }

    /// Check for virtualization-related kernel modules
    fn check_virtualization_modules(&self) -> Result<Vec<String>> {
        let mut modules = Vec::new();

        let virt_modules = [
            "virtio",
            "virtio_pci",
            "virtio_net",
            "virtio_blk",
            "vmw_balloon",
            "vmw_vmci",
            "vmwgfx",
            "vboxguest",
            "vboxsf",
            "hv_vmbus",
            "hv_storvsc",
            "hv_netvsc",
            "xen_blkfront",
            "xen_netfront",
        ];

        if let Ok(loaded_modules) = fs::read_to_string("/proc/modules") {
            for module in &virt_modules {
                if loaded_modules.contains(module) {
                    modules.push(module.to_string());
                }
            }
        }

        Ok(modules)
    }

    /// Check for virtual network interfaces
    fn check_virtual_network_interfaces(&self) -> Result<Vec<String>> {
        let mut interfaces = Vec::new();

        if let Ok(entries) = fs::read_dir("/sys/class/net") {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();

                // Check for common virtual interface patterns
                if name.starts_with("veth")      // Docker/container virtual ethernet
                    || name.starts_with("virbr") // libvirt bridge
                    || name.starts_with("docker")
                    || name.starts_with("vboxnet")
                    || name.starts_with("vmnet")
                {
                    interfaces.push(name);
                }
            }
        }

        Ok(interfaces)
    }

    /// Check for virtual disk devices
    fn check_virtual_disk_devices(&self) -> Result<Vec<String>> {
        let mut devices = Vec::new();

        if let Ok(entries) = fs::read_dir("/sys/block") {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();

                // Check device model/vendor
                let model_path = format!("/sys/block/{name}/device/model");
                let vendor_path = format!("/sys/block/{name}/device/vendor");

                if let Ok(model) = fs::read_to_string(&model_path) {
                    let model = model.trim().to_lowercase();
                    if model.contains("virtual")
                        || model.contains("vmware")
                        || model.contains("qemu")
                        || model.contains("vbox")
                    {
                        devices.push(format!("{} (model: {})", name, model.trim()));
                    }
                }

                if let Ok(vendor) = fs::read_to_string(&vendor_path) {
                    let vendor = vendor.trim().to_lowercase();
                    if vendor.contains("vmware")
                        || vendor.contains("qemu")
                        || vendor.contains("vbox")
                    {
                        devices.push(format!("{} (vendor: {})", name, vendor.trim()));
                    }
                }
            }
        }

        Ok(devices)
    }

    fn calculate_confidence(
        &self,
        hypervisor: &Option<HypervisorType>,
        container: &Option<ContainerType>,
        cpu_flags: &[String],
    ) -> f64 {
        let mut confidence: f64 = 0.0;

        if hypervisor.is_some() {
            confidence += 0.9;
        }
        if container.is_some() {
            confidence += 0.9;
        }
        if !cpu_flags.is_empty() {
            confidence += 0.5;
        }

        confidence.min(1.0)
    }
}

#[derive(Debug, Clone)]
pub struct VirtualizationStatus {
    pub is_virtualized: bool,
    pub hypervisor: Option<HypervisorType>,
    pub container: Option<ContainerType>,
    pub cpu_flags: Vec<String>,
    pub dmi_vendor: Option<String>,
    pub kernel_modules: Vec<String>,
    pub network_interfaces: Vec<String>,
    pub disk_devices: Vec<String>,
    pub confidence: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum HypervisorType {
    Kvm,
    Qemu,
    VmWare,
    VirtualBox,
    HyperV,
    Xen,
    Unknown(String),
}

#[derive(Debug, Clone, PartialEq)]
pub enum ContainerType {
    Docker,
    Kubernetes,
    Containerd,
    Lxc,
    SystemdNspawn,
    Unknown(String),
}

impl Default for VirtualizationDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_virtualization_detector_creation() {
        let detector = VirtualizationDetector::new();
        // Should create without errors
        let _ = detector;
    }

    #[test]
    fn test_virtualization_detection() {
        let detector = VirtualizationDetector::new();
        let result = detector.detect_virtualization();

        // Should complete without errors
        assert!(result.is_ok());

        let status = result.unwrap();
        // The result will vary depending on the test environment
        println!("Virtualization detected: {}", status.is_virtualized);
        println!("Confidence: {}", status.confidence);
    }
}
