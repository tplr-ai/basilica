//! Basilica sharded inference deployment template.

use crate::cli::commands::{ShardedOptions, TemplateCommonOptions};
use crate::error::{CliError, DeployError};
use crate::output::{print_info, print_success};
use crate::progress::{complete_spinner_and_clear, create_spinner};
use basilica_sdk::types::{
    CreateInferencePipelineRequest, InferenceCoordinatorRequest, InferenceModelRequest,
    InferencePipelineResponse, InferenceStageGpuRequest, InferenceStageSetRequest,
};
use basilica_sdk::BasilicaClient;

use super::model_size::estimate_gpu_requirements;

const DEFAULT_MODEL: &str = "sshleifer/tiny-gpt2";

pub async fn handle_sharded_deploy(
    client: &BasilicaClient,
    model: Option<String>,
    common: TemplateCommonOptions,
    opts: ShardedOptions,
) -> Result<(), CliError> {
    let model = model.unwrap_or_else(|| DEFAULT_MODEL.to_string());
    if opts.stages == 0 {
        return Err(CliError::Deploy(DeployError::Validation {
            message: "--stages must be greater than zero".to_string(),
        }));
    }

    let estimated = estimate_gpu_requirements(&model);
    let per_stage_memory = ((estimated.memory_gb as f32) / (opts.stages as f32)).ceil() as u32;
    let gpu_models = if common.gpu_model.is_empty() {
        vec![estimated.recommended_gpu.clone()]
    } else {
        common.gpu_model.clone()
    };
    print_info(&format!(
        "Sharding model across {} stage(s), ~{} GB VRAM per stage ({})",
        opts.stages,
        per_stage_memory,
        gpu_models.join(", ")
    ));

    let name = common
        .name
        .clone()
        .unwrap_or_else(|| generate_sharded_name(&model));
    let request = CreateInferencePipelineRequest {
        instance_name: Some(name.clone()),
        model: InferenceModelRequest {
            model_ref: model.clone(),
            revision: "main".to_string(),
            trust_remote_code: opts.trust_remote_code,
        },
        stages: InferenceStageSetRequest {
            count: opts.stages,
            gpu: InferenceStageGpuRequest {
                count: common.gpu.unwrap_or(1),
                model: gpu_models,
                min_gpu_memory_gb: Some(per_stage_memory.max(1)),
            },
        },
        coordinator: InferenceCoordinatorRequest {
            draft_model_ref: opts.draft_model.clone(),
            ..Default::default()
        },
        public: true,
    };

    let spinner = create_spinner(&format!(
        "Creating sharded inference summons '{}' with model '{}'...",
        name, model
    ));
    let response = client
        .create_inference_pipeline(request)
        .await
        .map_err(CliError::Api)?;
    complete_spinner_and_clear(spinner);

    if common.json {
        crate::output::json_output(&response)?;
    } else {
        print_sharded_success(&response, &model);
    }

    Ok(())
}

fn generate_sharded_name(model: &str) -> String {
    let model_part = model
        .split('/')
        .next_back()
        .unwrap_or(model)
        .to_lowercase()
        .chars()
        .filter_map(|c| {
            if c.is_ascii_alphanumeric() {
                Some(c)
            } else if c == '-' || c == '_' || c == '.' {
                Some('-')
            } else {
                None
            }
        })
        .collect::<String>();
    let sanitized = model_part.trim_matches('-');
    let prefix = if sanitized.is_empty() {
        "sharded"
    } else if sanitized.len() > 36 {
        &sanitized[..36]
    } else {
        sanitized
    };
    format!(
        "sharded-{}-{}",
        prefix,
        &uuid::Uuid::new_v4().to_string()[..8]
    )
}

fn print_sharded_success(response: &InferencePipelineResponse, model: &str) {
    print_success(&format!(
        "Sharded inference pipeline '{}' created",
        response.instance_name
    ));
    println!("  Model: {}", model);
    println!("  Status: {}", response.state);
    println!(
        "  OpenAI-compatible endpoint: {}/v1/chat/completions",
        response.url
    );
    println!(
        "  Receipt verification: {}/inference/receipts/verify",
        response.url.trim_end_matches('/')
    );
}
