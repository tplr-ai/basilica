use std::sync::Arc;
use std::time::Duration;

use anyhow::Result as AnyResult;
use futures::StreamExt;
use kube::{Api, Client, ResourceExt};
use kube_runtime::controller::{Action, Controller};
use kube_runtime::watcher;
use tracing::{error, info, warn};

use crate::billing::{BillingClient, HttpBillingClient, MockBillingClient};
use crate::controllers::job_controller::JobController;
use crate::controllers::node_profile_controller::NodeProfileController;
use crate::controllers::node_removal_controller::NodeRemovalController;
use crate::controllers::rental_controller::RentalController;
use crate::controllers::user_deployment_controller::UserDeploymentController;
use crate::crd::basilica_job::BasilicaJob;
use crate::crd::basilica_node_profile::BasilicaNodeProfile;
use crate::crd::gpu_rental::GpuRental;
use crate::crd::user_deployment::UserDeployment;
use crate::k8s_client::{K8sClient, KubeClient};
use crate::metrics_provider::K8sMetricsProvider;
use k8s_openapi::api::core::v1::Node;

#[derive(Clone)]
struct JobCtx<C: K8sClient + Clone + 'static> {
    ctrl: JobController<C>,
}

#[derive(Clone)]
struct RentalCtx<C: K8sClient + Clone + 'static> {
    ctrl: RentalController<C>,
}

#[derive(Clone)]
struct UserDeploymentCtx {
    ctrl: UserDeploymentController,
}

#[derive(Debug)]
struct ReconcileError(anyhow::Error);

impl std::fmt::Display for ReconcileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for ReconcileError {}

async fn reconcile_job<C: K8sClient + Clone + 'static>(
    obj: Arc<BasilicaJob>,
    ctx: Arc<JobCtx<C>>,
) -> std::result::Result<Action, ReconcileError> {
    let ns = obj.namespace().unwrap_or_else(|| "default".into());
    if let Err(e) = ctx.ctrl.reconcile(&ns, &obj).await {
        return Err(ReconcileError(e));
    }
    Ok(Action::requeue(Duration::from_secs(30)))
}

fn error_policy_job<C: K8sClient + Clone + 'static>(
    _obj: Arc<BasilicaJob>,
    err: &ReconcileError,
    _ctx: Arc<JobCtx<C>>,
) -> Action {
    warn!("job reconcile error: {}", err);
    Action::requeue(Duration::from_secs(10))
}

async fn reconcile_rental<C: K8sClient + Clone + 'static>(
    obj: Arc<GpuRental>,
    ctx: Arc<RentalCtx<C>>,
) -> std::result::Result<Action, ReconcileError> {
    let ns = obj.namespace().unwrap_or_else(|| "default".into());
    if let Err(e) = ctx.ctrl.reconcile(&ns, &obj).await {
        return Err(ReconcileError(e));
    }
    Ok(Action::requeue(Duration::from_secs(30)))
}

fn error_policy_rental<C: K8sClient + Clone + 'static>(
    _obj: Arc<GpuRental>,
    err: &ReconcileError,
    _ctx: Arc<RentalCtx<C>>,
) -> Action {
    warn!("rental reconcile error: {}", err);
    Action::requeue(Duration::from_secs(10))
}

async fn reconcile_user_deployment(
    obj: Arc<UserDeployment>,
    ctx: Arc<UserDeploymentCtx>,
) -> std::result::Result<Action, ReconcileError> {
    let ns = obj.namespace().unwrap_or_else(|| "default".into());
    if let Err(e) = ctx.ctrl.reconcile(&ns, &obj).await {
        return Err(ReconcileError(e));
    }
    Ok(Action::requeue(Duration::from_secs(30)))
}

fn error_policy_user_deployment(
    _obj: Arc<UserDeployment>,
    err: &ReconcileError,
    _ctx: Arc<UserDeploymentCtx>,
) -> Action {
    warn!("user deployment reconcile error: {}", err);
    Action::requeue(Duration::from_secs(10))
}

pub async fn run() -> AnyResult<()> {
    let client = Client::try_default().await?;
    let kube_client = KubeClient {
        client: client.clone(),
    };

    // Choose billing client based on env var BASILICA_BILLING_URL
    let billing_arc: std::sync::Arc<dyn BillingClient + Send + Sync> =
        match std::env::var("BASILICA_BILLING_URL") {
            Ok(url) if !url.is_empty() => std::sync::Arc::new(HttpBillingClient::new(url)),
            _ => std::sync::Arc::new(MockBillingClient::default()),
        };
    // Build controllers
    let mut job_ctrl = JobController::new_with_billing(kube_client.clone(), billing_arc.clone());
    let mut rent_ctrl = RentalController::new_with_arc(kube_client.clone(), billing_arc);
    let node_removal_ctrl = NodeRemovalController::new(kube_client.clone());
    let node_profile_ctrl = NodeProfileController::new(kube_client.clone());

    let public_ip =
        std::env::var("DEPLOYMENT_PUBLIC_IP").unwrap_or_else(|_| "localhost".to_string());
    let public_port = std::env::var("DEPLOYMENT_PUBLIC_PORT")
        .ok()
        .and_then(|s| s.parse::<u16>().ok())
        .unwrap_or(8080);
    let user_deploy_ctrl = UserDeploymentController::new(
        std::sync::Arc::new(kube_client.clone()),
        public_ip,
        public_port,
    );

    // Optionally enable K8s metrics provider when BASILICA_ENABLE_K8S_METRICS=true
    if std::env::var("BASILICA_ENABLE_K8S_METRICS").ok().as_deref() == Some("true") {
        let provider = std::sync::Arc::new(K8sMetricsProvider::new(client.clone()));
        job_ctrl = job_ctrl.with_metrics_provider(provider.clone());
        rent_ctrl = rent_ctrl.with_metrics_provider(provider);
    }

    // Set up APIs
    let bj_api: Api<BasilicaJob> = Api::all(client.clone());
    let gr_api: Api<GpuRental> = Api::all(client.clone());
    let np_api: Api<BasilicaNodeProfile> = Api::all(client.clone());
    let ud_api: Api<UserDeployment> = Api::all(client.clone());
    let node_api: Api<Node> = Api::all(client.clone());

    // Run controllers as streams with explicit watcher configuration
    let jobs = Controller::new(bj_api, watcher::Config::default().any_semantic())
        .run(
            reconcile_job,
            error_policy_job,
            Arc::new(JobCtx { ctrl: job_ctrl }),
        )
        .for_each(|res| async move {
            match res {
                Ok(_o) => {}
                Err(e) => {
                    error!("job controller stream error: {}", e);
                }
            }
        });

    let rentals = Controller::new(gr_api, watcher::Config::default().any_semantic())
        .run(
            reconcile_rental,
            error_policy_rental,
            Arc::new(RentalCtx { ctrl: rent_ctrl }),
        )
        .for_each(|res| async move {
            match res {
                Ok(_o) => {}
                Err(e) => {
                    error!("rental controller stream error: {}", e);
                }
            }
        });

    let node_removal = Controller::new(np_api, watcher::Config::default().any_semantic())
        .run(
            |obj, _| {
                let ctrl = node_removal_ctrl.clone();
                async move {
                    if let Err(e) = ctrl.reconcile(&obj).await {
                        return Err(ReconcileError(e));
                    }
                    Ok(Action::requeue(Duration::from_secs(30)))
                }
            },
            |_obj, err, _| {
                warn!("node removal reconcile error: {}", err);
                Action::requeue(Duration::from_secs(10))
            },
            Arc::new(()),
        )
        .for_each(|res| async move {
            match res {
                Ok(_o) => {}
                Err(e) => {
                    error!("node removal controller stream error: {}", e);
                }
            }
        });

    let node_profile = Controller::new(node_api, watcher::Config::default().any_semantic())
        .run(
            |obj, _| {
                let ctrl = node_profile_ctrl.clone();
                async move {
                    if let Err(e) = ctrl.reconcile(&obj).await {
                        return Err(ReconcileError(e));
                    }
                    Ok(Action::requeue(Duration::from_secs(60)))
                }
            },
            |_obj, err, _| {
                warn!("node profile reconcile error: {}", err);
                Action::requeue(Duration::from_secs(10))
            },
            Arc::new(()),
        )
        .for_each(|res| async move {
            match res {
                Ok(_o) => {}
                Err(e) => {
                    error!("node profile controller stream error: {}", e);
                }
            }
        });

    let user_deployments = Controller::new(ud_api, watcher::Config::default().any_semantic())
        .run(
            reconcile_user_deployment,
            error_policy_user_deployment,
            Arc::new(UserDeploymentCtx {
                ctrl: user_deploy_ctrl,
            }),
        )
        .for_each(|res| async move {
            match res {
                Ok(_o) => {}
                Err(e) => {
                    error!("user deployment controller stream error: {}", e);
                }
            }
        });

    info!("operator controllers started");
    futures::future::join5(jobs, rentals, node_removal, node_profile, user_deployments).await;
    Ok(())
}
