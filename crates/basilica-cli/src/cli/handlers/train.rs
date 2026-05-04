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
            command,
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
        } => {
            handle_up(
                &client,
                name,
                command,
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
        TrainAction::Logs { name, tail, follow } => handle_logs(&client, &name, follow, tail).await,
        TrainAction::Events { name } => handle_events(&client, &name, json).await,
        TrainAction::Bench { name } => handle_bench(&client, &name, json).await,
        TrainAction::Down { name } => handle_down(&client, &name, json).await,
    }
}

#[allow(clippy::too_many_arguments)]
async fn handle_up(
    client: &basilica_sdk::BasilicaClient,
    name: String,
    command: String,
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
    ttl_seconds: u32,
    json: bool,
) -> Result<(), CliError> {
    if command.trim().is_empty() {
        return Err(CliError::Internal(eyre!(
            "--command must not be empty (BYO-launcher mode is required \
             for `basilica train up`; for source-shipping use the Python \
             SDK or @basilica.distributed)"
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

    // Stash the user's BYO command on `spec.distributed.command`. The
    // operator wraps it as `sh -c <command>` (see operator
    // distributed.rs::build_worker_command BYO branch). Empty
    // `spec.command` and `spec.args` mean the operator's `$@`
    // positional-arg list ends up just `["--"]`, harmless for the
    // user's launcher.
    let mut spec_with_command = spec;
    spec_with_command.command = command.clone();

    let request = CreateDistributedDeploymentRequest {
        instance_name: name.clone(),
        image,
        replicas: world_size.target,
        port: 18789,
        command: None,
        args: None,
        env: None,
        resources: Some(resources),
        // 0 means "no auto-delete" (escape hatch); otherwise convert.
        ttl_seconds: if ttl_seconds == 0 {
            None
        } else {
            Some(ttl_seconds)
        },
        enable_billing: true,
        distributed: spec_with_command,
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

/// Hard cap on per-call N+1 distributed-filter queries. Phase 5b: the
/// API does not yet expose a server-side `?distributed=true` filter, so
/// `train ls`/`ps` walk the full deployments list and check each via
/// `get_deployment` for `spec.distributed.enabled`. Above this cap we
/// fall back to "show all" with a warning rather than burst the API.
/// Phase 6 follow-up: add the server-side filter and remove the cap.
const TRAIN_LS_FILTER_CAP: usize = 50;

async fn handle_ls(client: &basilica_sdk::BasilicaClient, json: bool) -> Result<(), CliError> {
    let response = client.list_deployments().await.map_err(map_sdk_err)?;

    if response.deployments.len() > TRAIN_LS_FILTER_CAP {
        print_info(&format!(
            "More than {} deployments; skipping per-UD distributed filter \
             (Phase 6 server-side filter is a follow-up). Showing all.",
            TRAIN_LS_FILTER_CAP
        ));
        if json {
            json_output(&response)?;
        } else {
            for d in &response.deployments {
                println!("{}\t{}", d.instance_name, d.state);
            }
        }
        return Ok(());
    }

    // N+1 distributed-filter for the small-namespace case. The API
    // gateway returns the full DeploymentResponse (including the
    // `distributed` block when set). For Phase 5b this is acceptable
    // (most users have a handful of UDs); the cap above guards
    // against surprise fan-out.
    let mut distributed_rows: Vec<(String, String)> = Vec::new();
    for summary in &response.deployments {
        match client.get_deployment(&summary.instance_name).await {
            Ok(full) => {
                // The API's DeploymentResponse does not yet typed-expose
                // `spec.distributed`. Phase 5b heuristic: a UD whose name
                // shows up in `train ls` either was created via the
                // distributed path (we can't tell from the response
                // alone today) -- so default to showing all and document
                // the gap. Phase 6 follow-up: surface a `distributed`
                // boolean on DeploymentResponse.
                let _ = full;
                distributed_rows.push((summary.instance_name.clone(), summary.state.clone()));
            }
            Err(_) => continue,
        }
    }

    if json {
        // Emit the same response shape as the non-filtered path.
        json_output(&response)?;
    } else {
        if distributed_rows.is_empty() {
            print_info("No distributed deployments found in this namespace.");
        }
        for (name, state) in &distributed_rows {
            println!("{}\t{}", name, state);
        }
        // Honest note: Phase 5b cannot strictly distinguish distributed
        // UDs from non-distributed without per-UD GET inspection of
        // `spec.distributed`, which the current `DeploymentResponse`
        // does not expose. This is shown to the user so they don't
        // assume the filter is exact.
        print_info(
            "Note: `train ls` currently shows all deployments. Phase 6 will \
             add a `distributed` flag on DeploymentResponse for strict \
             filtering.",
        );
    }
    Ok(())
}

async fn handle_ps(client: &basilica_sdk::BasilicaClient, json: bool) -> Result<(), CliError> {
    // Same surface as `ls` for Phase 5b.
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
    // Phase 5b gap: the operator populates `status.distributed.bench`
    // on the CR (architecture doc § 11.1, PR #389) but the basilica-api
    // gateway's `DeploymentResponse` does not yet typed-expose it.
    // `basilica train bench` returns a non-zero exit code so users
    // notice -- placeholder JSON with `bench: null` would fail silently
    // in scripts. The Python facade (`training.bench`) has the same
    // limitation today; both clear once a Phase 6 backend follow-up
    // exposes the field.
    let _ = client.get_deployment(name).await.map_err(map_sdk_err)?;
    let note = "status.distributed.bench is populated on the K8s CR by the \
                operator (PR #389) but is not yet exposed via the basilica-api \
                gateway's DeploymentResponse. Phase 6 backend follow-up will \
                surface it; until then `basilica train bench` is unavailable. \
                Cluster operators can read the value directly via \
                `kubectl get userdeployment <ud>-deployment -n <ns> -o \
                jsonpath='{.status.distributed.bench}'`.";
    if json {
        json_output(&serde_json::json!({
            "name": name,
            "bench": null,
            "available": false,
            "note": note,
        }))?;
    } else {
        print_error(&format!(
            "bench result for '{}' is not yet available.",
            name
        ));
        print_info(note);
    }
    // Non-zero exit so CI / scripts catch the gap rather than treat
    // it as a silently-empty success.
    Err(CliError::Internal(eyre!(
        "train bench is staged but the API does not yet expose \
         status.distributed.bench (Phase 6 follow-up); see message above"
    )))
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
