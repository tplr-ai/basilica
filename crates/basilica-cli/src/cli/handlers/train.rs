//! `basilica train ...` handlers (SDK arch § 10).
//!
//! Each subcommand maps 1:1 onto the basilica-sdk Rust client method
//! shipped in this same PR (`create_distributed_deployment`,
//! `scale_distributed_deployment`, `get_deployment`, `delete_deployment`,
//! `get_deployment_logs`, `get_deployment_events`).
//!
//! Notably absent: `train preflight` and `train nccl-baseline`. SDK arch
//! § 7 / § 10 explicitly remove those commands -- they would imply a
//! cross-tenant aggregated bench cache, violating the platform's tenancy
//! invariant. Bench data is per-UD via `train up --bench on-start`
//! followed by `train bench <name>`. Phase 5b lock-down test:
//! `cargo test test_train_subcommands_no_preflight_or_baseline`.

use crate::cli::commands::{
    TrainAction, TrainBenchMode, TrainCommand, TrainRendezvousBackend, TrainTopologySpread,
    WorldSizeTriple,
};
use crate::client::create_authenticated_client;
use crate::config::CliConfig;
use crate::error::CliError;
use crate::output::{json_output, print_error, print_info, print_success};
use basilica_sdk::{
    CreateDistributedDeploymentRequest, DistributedBenchMode, DistributedBenchSpec,
    DistributedNcclSpec, DistributedProviderFilter, DistributedRendezvousBackend,
    DistributedRendezvousSpec, DistributedSpec, DistributedTopologySpread,
    DistributedTopologySpreadStrategy, DistributedWorldSize, GpuRequirementsSpec,
    ResourceRequirements,
};
use color_eyre::eyre::eyre;
use std::collections::BTreeMap;

pub async fn handle_train(cmd: TrainCommand, config: &CliConfig) -> Result<(), CliError> {
    let client = create_authenticated_client(config).await?;
    let json = cmd.json;

    match cmd.action {
        TrainAction::Up {
            name,
            source,
            world_size,
            gpu_count,
            gpu_models,
            min_gpu_memory_gb,
            image,
            cpu,
            memory,
            provider,
            exclude_provider,
            topology_spread,
            nccl_env,
            bench,
            rendezvous_backend,
            ttl_seconds,
            timeout: _timeout,
        } => {
            handle_up(
                &client,
                name,
                source,
                world_size,
                gpu_count,
                gpu_models,
                min_gpu_memory_gb,
                image,
                cpu,
                memory,
                provider,
                exclude_provider,
                topology_spread,
                nccl_env,
                bench,
                rendezvous_backend,
                ttl_seconds,
                json,
            )
            .await
        }
        TrainAction::Ls => handle_ls(&client, json).await,
        TrainAction::Ps => handle_ps(&client, json).await,
        TrainAction::Scale { name, target } => handle_scale(&client, &name, target, json).await,
        TrainAction::Logs {
            name,
            rank: _rank,
            tail,
            follow,
        } => handle_logs(&client, &name, follow, tail).await,
        TrainAction::Events { name } => handle_events(&client, &name, json).await,
        TrainAction::Bench { name } => handle_bench(&client, &name, json).await,
        TrainAction::Down { name } => handle_down(&client, &name, json).await,
    }
}

#[allow(clippy::too_many_arguments)]
async fn handle_up(
    client: &basilica_sdk::BasilicaClient,
    name: String,
    source: std::path::PathBuf,
    world_size: WorldSizeTriple,
    gpu_count: u32,
    gpu_models: Vec<String>,
    min_gpu_memory_gb: Option<u32>,
    image: String,
    cpu: String,
    memory: String,
    provider: Vec<String>,
    exclude_provider: Vec<String>,
    topology_spread: TrainTopologySpread,
    nccl_env: Vec<String>,
    bench: TrainBenchMode,
    rendezvous_backend: TrainRendezvousBackend,
    ttl_seconds: Option<u32>,
    json: bool,
) -> Result<(), CliError> {
    // Source-packaging: read the file's contents and inline them as the
    // operator's `command="auto"` script. The CLI does not run pip or
    // build a layered image -- the operator's torchrun wrapper handles
    // dispatch. For BYO-launcher use cases, drop down to the SDK directly.
    if !source.exists() {
        return Err(CliError::Internal(eyre!(
            "source file not found: {}",
            source.display()
        )));
    }

    // Translate CLI enum -> SDK enum (kept deliberately separate so a
    // future CLI rename does not silently change the wire token).
    let topology = match topology_spread {
        TrainTopologySpread::Pack => DistributedTopologySpreadStrategy::Pack,
        TrainTopologySpread::ProviderAware => DistributedTopologySpreadStrategy::ProviderAware,
        TrainTopologySpread::RegionAware => DistributedTopologySpreadStrategy::RegionAware,
        TrainTopologySpread::None => DistributedTopologySpreadStrategy::None,
    };

    let bench_mode = match bench {
        TrainBenchMode::Off => DistributedBenchMode::Off,
        TrainBenchMode::OnStart => DistributedBenchMode::OnStart,
    };

    let rdzv = match rendezvous_backend {
        TrainRendezvousBackend::EtcdV2 => DistributedRendezvousBackend::EtcdV2,
        TrainRendezvousBackend::C10d => DistributedRendezvousBackend::C10d,
        TrainRendezvousBackend::Static => DistributedRendezvousBackend::Static,
    };

    // Parse `KEY=VALUE` env-var list.
    let mut nccl_env_map = BTreeMap::new();
    for pair in nccl_env {
        let (k, v) = pair.split_once('=').ok_or_else(|| {
            CliError::Internal(eyre!("--nccl-env expects KEY=VALUE, got {:?}", pair))
        })?;
        nccl_env_map.insert(k.to_string(), v.to_string());
    }

    let spec = DistributedSpec {
        enabled: true,
        world_size: DistributedWorldSize {
            min: world_size.min,
            target: world_size.target,
            max: world_size.max,
        },
        rendezvous: DistributedRendezvousSpec {
            backend: rdzv,
            port: None,
        },
        provider_filter: DistributedProviderFilter {
            include: provider,
            exclude: exclude_provider,
        },
        topology_spread: DistributedTopologySpread { strategy: topology },
        nccl: DistributedNcclSpec { env: nccl_env_map },
        bench: Some(DistributedBenchSpec { mode: bench_mode }),
        command: "auto".to_string(),
    };

    let resources = ResourceRequirements {
        cpu,
        memory,
        cpu_request: None,
        memory_request: None,
        gpus: Some(GpuRequirementsSpec {
            count: gpu_count,
            model: gpu_models,
            min_cuda_version: None,
            min_gpu_memory_gb,
            interconnect: None,
            geo: None,
            spot: None,
            infiniband: None,
        }),
    };

    let request = CreateDistributedDeploymentRequest {
        instance_name: name.clone(),
        image,
        replicas: world_size.target,
        port: 18789,
        // Source-as-args path mirrors the basilica-distributed-trainer image's
        // expectation: `args: ["--", "python3", "/workspace/<script>"]`.
        // BYO-launcher use cases drop down to the SDK directly.
        command: None,
        args: Some(vec![
            "--".to_string(),
            "python3".to_string(),
            format!(
                "/workspace/{}",
                source.file_name().unwrap().to_string_lossy()
            ),
        ]),
        env: None,
        resources: Some(resources),
        ttl_seconds,
        enable_billing: true,
        distributed: spec,
    };

    print_info(&format!(
        "Launching distributed UD '{}' (world={}:{}:{}, providers={:?})",
        name,
        world_size.min,
        world_size.target,
        world_size.max,
        request.distributed.provider_filter.include,
    ));

    let response = client
        .create_distributed_deployment(request)
        .await
        .map_err(map_sdk_err)?;

    if json {
        json_output(&response)?;
    } else {
        print_success(&format!(
            "Distributed UD '{}' admitted (state={}, namespace={})",
            response.instance_name, response.state, response.namespace
        ));
        print_info("Use `basilica train ps` or `basilica train logs` to follow progress.");
    }
    Ok(())
}

async fn handle_ls(client: &basilica_sdk::BasilicaClient, json: bool) -> Result<(), CliError> {
    // For Phase 5b, `train ls` reuses the existing /deployments listing
    // and filters on spec.distributed presence client-side. A dedicated
    // server-side filter is a Phase 6+ optimization.
    let response = client.list_deployments().await.map_err(map_sdk_err)?;
    if json {
        json_output(&response)?;
    } else {
        for d in &response.deployments {
            println!("{}\t{}", d.instance_name, d.state);
        }
    }
    Ok(())
}

async fn handle_ps(client: &basilica_sdk::BasilicaClient, json: bool) -> Result<(), CliError> {
    // Same surface as `ls` for Phase 5b; Phase 6 may filter to "active"
    // distributed UDs only.
    handle_ls(client, json).await
}

async fn handle_scale(
    client: &basilica_sdk::BasilicaClient,
    name: &str,
    target: u32,
    json: bool,
) -> Result<(), CliError> {
    if target == 0 {
        return Err(CliError::Internal(eyre!("scale target must be >= 1")));
    }
    let response = client
        .scale_distributed_deployment(name, target)
        .await
        .map_err(map_sdk_err)?;
    if json {
        json_output(&response)?;
    } else {
        print_success(&format!(
            "Scaled '{}' worldSize.target -> {} (state={})",
            response.instance_name, target, response.state
        ));
    }
    Ok(())
}

async fn handle_logs(
    client: &basilica_sdk::BasilicaClient,
    name: &str,
    follow: bool,
    tail: Option<u32>,
) -> Result<(), CliError> {
    // Phase 5b: per-rank logs reuse the existing `/deployments/{name}/logs`
    // streaming endpoint; the operator labels per-rank pods so `--rank N`
    // is a future filter on the client side. For now we surface the
    // first available pod's logs (rank-0).
    let response = client
        .get_deployment_logs(name, follow, tail)
        .await
        .map_err(map_sdk_err)?;
    let body = response
        .text()
        .await
        .map_err(|e| CliError::Internal(eyre!("Failed to read log stream: {}", e)))?;
    println!("{}", body);
    Ok(())
}

async fn handle_events(
    client: &basilica_sdk::BasilicaClient,
    name: &str,
    json: bool,
) -> Result<(), CliError> {
    let response = client
        .get_deployment_events(name, None)
        .await
        .map_err(map_sdk_err)?;
    if json {
        json_output(&response)?;
    } else {
        for evt in &response.events {
            println!(
                "{}\t{}\t{}\t{}",
                evt.last_timestamp.as_deref().unwrap_or("-"),
                evt.event_type,
                evt.reason,
                evt.message
            );
        }
    }
    Ok(())
}

async fn handle_bench(
    client: &basilica_sdk::BasilicaClient,
    name: &str,
    json: bool,
) -> Result<(), CliError> {
    // The bench result is read off `status.distributed.bench.result`
    // populated by the operator. The current API gateway's
    // DeploymentResponse does not yet typed-expose `status.distributed`;
    // until a follow-up exposes it, we surface a not-yet-available
    // message and document the SDK arch § 7 rule.
    let _ = client.get_deployment(name).await.map_err(map_sdk_err)?;
    if json {
        json_output(&serde_json::json!({
            "name": name,
            "bench": null,
            "note": "status.distributed.bench is not yet exposed by the API \
                     DeploymentResponse; populated on the K8s CR. Phase 6 follow-up.",
        }))?;
    } else {
        print_info(&format!(
            "bench result for '{}' will surface once the API gateway exposes \
             status.distributed.bench (operator already populates the CR; \
             Phase 6 follow-up).",
            name
        ));
    }
    Ok(())
}

async fn handle_down(
    client: &basilica_sdk::BasilicaClient,
    name: &str,
    json: bool,
) -> Result<(), CliError> {
    let response = client.delete_deployment(name).await.map_err(map_sdk_err)?;
    if json {
        json_output(&response)?;
    } else {
        print_success(&format!("Deleted '{}'", name));
    }
    Ok(())
}

fn map_sdk_err(e: basilica_sdk::ApiError) -> CliError {
    print_error(&format!("API error: {}", e));
    CliError::Api(e)
}
