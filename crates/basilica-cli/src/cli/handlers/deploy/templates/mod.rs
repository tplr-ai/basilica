//! Deployment templates for common ML inference frameworks
//!
//! This module provides pre-configured deployment shortcuts for:
//! - vLLM: OpenAI-compatible LLM inference server
//! - SGLang: Fast LLM inference with RadixAttention
//! - OpenClaw: OpenClaw gateway
//! - Tau: Telegram agent with Cursor CLI

pub mod common;
pub mod model_size;
pub mod openclaw;
pub mod sglang;
pub mod tau;
pub mod vllm;

pub use model_size::{estimate_gpu_requirements, GpuRequirements};
pub use openclaw::handle_openclaw_deploy;
pub use sglang::handle_sglang_deploy;
pub use tau::handle_tau_deploy;
pub use vllm::handle_vllm_deploy;
