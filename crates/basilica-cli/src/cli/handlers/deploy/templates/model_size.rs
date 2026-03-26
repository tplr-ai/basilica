//! Model size estimation and GPU requirement calculation
//!
//! Provides sensible defaults based on model name patterns.
//! Users can override with explicit CLI flags (--gpu, --memory, --gpu-model).

use regex::Regex;
use std::sync::LazyLock;

/// GPU requirements for a model
#[derive(Debug, Clone)]
pub struct GpuRequirements {
    /// Number of GPUs required
    pub gpu_count: u32,
    /// Estimated GPU memory required in GB
    pub memory_gb: u32,
    /// Recommended GPU model
    pub recommended_gpu: String,
}

/// Default GPU memory per device (80GB for A100/H100)
const DEFAULT_GPU_MEMORY_GB: u32 = 80;

/// Default memory estimate when parameter count cannot be extracted
const DEFAULT_MEMORY_GB: u32 = 16;

/// Regex pattern for extracting parameter count from model names
/// Matches patterns like "7b", "70b", "0.5b", "1.5b", "7B", "235B", etc.
static PARAM_REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(\d+\.?\d*)[bB]").expect("Invalid regex pattern"));

/// Estimate GPU requirements based on model name
///
/// Extracts parameter count from model name and estimates memory needs.
/// Returns sensible defaults that users can override with CLI flags.
pub fn estimate_gpu_requirements(model: &str) -> GpuRequirements {
    let memory_gb = extract_param_count(model)
        .map(estimate_memory_from_params)
        .unwrap_or(DEFAULT_MEMORY_GB);

    let gpu_count = calculate_gpu_count(memory_gb, DEFAULT_GPU_MEMORY_GB);
    let recommended_gpu = recommend_gpu(memory_gb);

    GpuRequirements {
        gpu_count,
        memory_gb,
        recommended_gpu,
    }
}

/// Extract billion parameter count from model name
fn extract_param_count(model: &str) -> Option<f32> {
    PARAM_REGEX
        .captures(model)
        .and_then(|cap| cap.get(1))
        .and_then(|m| m.as_str().parse::<f32>().ok())
}

/// Estimate VRAM needed in GB from parameter count
///
/// Rule of thumb: ~2GB per billion parameters for FP16
/// Add 20% overhead for KV cache and runtime
fn estimate_memory_from_params(params_billions: f32) -> u32 {
    let base = params_billions * 2.0 * 1.2;
    // Round up to nearest 8GB
    (base as u32).div_ceil(8) * 8
}

/// Calculate number of GPUs needed
fn calculate_gpu_count(required_memory_gb: u32, gpu_memory_gb: u32) -> u32 {
    let count = (required_memory_gb as f32 / gpu_memory_gb as f32).ceil() as u32;
    count.clamp(1, 8)
}

/// Recommend GPU based on memory requirements
fn recommend_gpu(memory_gb: u32) -> String {
    if memory_gb <= 80 {
        "A100".to_string()
    } else {
        "H100".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_param_count() {
        assert!((extract_param_count("llama-2-7b").unwrap() - 7.0).abs() < 0.001);
        assert!((extract_param_count("Qwen3-0.6B").unwrap() - 0.6).abs() < 0.001);
        assert!((extract_param_count("Llama-2-70b").unwrap() - 70.0).abs() < 0.001);
        assert!((extract_param_count("Qwen3-235B").unwrap() - 235.0).abs() < 0.001);
        assert!(extract_param_count("phi-2").is_none()); // No 'b' suffix
    }

    #[test]
    fn test_estimate_memory_from_params() {
        // 7B: 7 * 2 * 1.2 = 16.8 -> 16GB (truncated then rounded up to 8)
        assert_eq!(estimate_memory_from_params(7.0), 16);
        // 0.5B: 0.5 * 2 * 1.2 = 1.2 -> 8GB
        assert_eq!(estimate_memory_from_params(0.5), 8);
        // 70B: 70 * 2 * 1.2 = 168 -> 168GB
        assert_eq!(estimate_memory_from_params(70.0), 168);
        // 235B: 235 * 2 * 1.2 = 564 -> 568GB
        assert_eq!(estimate_memory_from_params(235.0), 568);
    }

    #[test]
    fn test_calculate_gpu_count() {
        assert_eq!(calculate_gpu_count(16, 80), 1);
        assert_eq!(calculate_gpu_count(80, 80), 1);
        assert_eq!(calculate_gpu_count(160, 80), 2);
        assert_eq!(calculate_gpu_count(320, 80), 4);
        assert_eq!(calculate_gpu_count(640, 80), 8);
        assert_eq!(calculate_gpu_count(1000, 80), 8); // Capped at 8
    }

    #[test]
    fn test_estimate_gpu_requirements() {
        // Small model
        let reqs = estimate_gpu_requirements("Qwen/Qwen2.5-0.5B-Instruct");
        assert_eq!(reqs.gpu_count, 1);
        assert_eq!(reqs.memory_gb, 8);
        assert_eq!(reqs.recommended_gpu, "A100");

        // 7B model
        let reqs = estimate_gpu_requirements("meta-llama/Llama-2-7b");
        assert_eq!(reqs.gpu_count, 1);
        assert_eq!(reqs.recommended_gpu, "A100");

        // 70B model
        let reqs = estimate_gpu_requirements("meta-llama/Llama-2-70b");
        assert!(reqs.gpu_count >= 2);
        assert_eq!(reqs.recommended_gpu, "H100");

        // 235B model (extracts from name)
        let reqs = estimate_gpu_requirements("Qwen/Qwen3-235B-A22B");
        assert!(reqs.gpu_count >= 4);
        assert_eq!(reqs.recommended_gpu, "H100");
    }

    #[test]
    fn test_unknown_model_uses_default() {
        let reqs = estimate_gpu_requirements("some-unknown-model");
        assert_eq!(reqs.memory_gb, DEFAULT_MEMORY_GB);
        assert_eq!(reqs.gpu_count, 1);
    }
}
