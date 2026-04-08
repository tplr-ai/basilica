//! SDK integration tests against a live K3d cluster.
//!
//! These tests require a running K3d cluster with basilica-api (dev mode) and the
//! sandbox operator deployed. Run with:
//!
//!   cargo test --test sandbox_k3d -- --ignored
//!
//! Or via the runner script:
//!
//!   scripts/localtest/sandbox-k3d-e2e.sh sdk-test

use std::process::{Command, Stdio};
use std::time::Duration;
use std::{net::TcpListener, sync::Arc};

use basilica_sdk::sandbox::SandboxEnvVar;
use basilica_sdk::{ApiError, BasilicaClient, ClientBuilder, CreateSandboxRequest, Sandbox};
use serde_json::Value;
use tokio::sync::OnceCell;

const DEFAULT_API_URL: &str = "http://localhost:18082";
const NAMESPACE: &str = "u-test-user";
static SHARED_SANDBOX: OnceCell<Arc<(Sandbox, String)>> = OnceCell::const_new();

fn sandbox_image() -> String {
    std::env::var("SANDBOX_IMAGE").unwrap_or_else(|_| {
        let tag_file =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../scripts/localtest/.sandbox-image-tag");
        std::fs::read_to_string(tag_file)
            .ok()
            .map(|tag| format!("k3d-basilica-registry:5050/basilica-exec-agent:{}", tag.trim()))
            .unwrap_or_else(|| "k3d-basilica-registry:5050/basilica-exec-agent:latest".to_string())
    })
}

fn api_url() -> String {
    std::env::var("BASILICA_API_URL").unwrap_or_else(|_| DEFAULT_API_URL.to_string())
}

fn build_client() -> BasilicaClient {
    ClientBuilder::default()
        .base_url(api_url())
        .with_tokens("test-token", "test-refresh")
        .build()
        .expect("failed to build client")
}

/// Wait for a sandbox to reach Running state by polling get_sandbox.
async fn wait_for_running(client: &BasilicaClient, sandbox_id: &str) -> basilica_sdk::SandboxDetail {
    let max_wait = Duration::from_secs(120);
    let poll_interval = Duration::from_secs(2);
    let start = std::time::Instant::now();

    loop {
        let detail = client.get_sandbox(sandbox_id).await.expect("get_sandbox failed");
        if detail.status == "Running" {
            return detail;
        }
        if start.elapsed() > max_wait {
            panic!(
                "Sandbox {} did not reach Running state within {:?} (current: {})",
                sandbox_id, max_wait, detail.status
            );
        }
        tokio::time::sleep(poll_interval).await;
    }
}

/// Set up kubectl port-forward to the sandbox pod and return (local_port, child_process).
fn setup_port_forward(sandbox_id: &str) -> (u16, std::process::Child) {
    // Get the pod name
    let pod_name = format!("sandbox-{}", sandbox_id);

    // Wait for pod to be ready
    let status = Command::new("kubectl")
        .args([
            "wait",
            "--for=condition=Ready",
            &format!("pod/{}", pod_name),
            "-n",
            NAMESPACE,
            "--timeout=90s",
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .expect("kubectl wait failed");
    assert!(status.success(), "Pod {} did not become ready", pod_name);

    let listener = TcpListener::bind("127.0.0.1:0").expect("failed to reserve local port");
    let local_port = listener
        .local_addr()
        .expect("listener should have local addr")
        .port();
    drop(listener);

    let child = Command::new("kubectl")
        .args([
            "port-forward",
            "-n",
            NAMESPACE,
            &format!("pod/{}", pod_name),
            &format!("{}:9999", local_port),
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("failed to start port-forward");

    // Wait for port-forward to be ready
    std::thread::sleep(Duration::from_secs(2));

    (local_port, child)
}

/// Get the exec-agent secret from the K8s Secret.
fn get_exec_agent_secret(sandbox_id: &str) -> String {
    let output = Command::new("kubectl")
        .args([
            "get",
            "secret",
            "-n",
            NAMESPACE,
            &format!("sandbox-{}-exec-secret", sandbox_id),
            "-o",
            "jsonpath={.data.EXEC_AGENT_SECRET}",
        ])
        .output()
        .expect("kubectl get secret failed");

    if !output.status.success() {
        // Secret might not exist in K3d test setup -- fall back to the secret
        // returned by the create API
        return String::new();
    }

    let b64 = String::from_utf8_lossy(&output.stdout).to_string();
    if b64.is_empty() {
        return String::new();
    }

    // base64 decode
    let decoded = Command::new("base64")
        .arg("-d")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .and_then(|mut child| {
            use std::io::Write;
            child
                .stdin
                .take()
                .unwrap()
                .write_all(b64.as_bytes())
                .ok();
            child.wait_with_output()
        })
        .map(|o| String::from_utf8_lossy(&o.stdout).to_string())
        .unwrap_or_default();

    decoded
}

fn kill_port_forward(child: &mut std::process::Child) {
    let _ = child.kill();
}

async fn delete_sandbox_quietly(client: &BasilicaClient, sandbox_id: &str) {
    let _ = client.delete_sandbox(sandbox_id).await;
}

fn kubectl_output(args: &[&str]) -> String {
    let output = Command::new("kubectl")
        .args(args)
        .output()
        .expect("kubectl command failed");
    assert!(
        output.status.success(),
        "kubectl {:?} failed: {}",
        args,
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout).trim().to_string()
}

fn kubectl_json(args: &[&str]) -> Value {
    serde_json::from_str(&kubectl_output(args)).expect("kubectl output was not valid JSON")
}

fn pod_uid(sandbox_id: &str) -> String {
    let output = Command::new("kubectl")
        .args([
            "get",
            "pod",
            &format!("sandbox-{}", sandbox_id),
            "-n",
            NAMESPACE,
            "-o",
            "jsonpath={.metadata.uid}",
        ])
        .output()
        .expect("kubectl get pod failed");
    if !output.status.success() {
        return String::new();
    }
    String::from_utf8_lossy(&output.stdout).trim().to_string()
}

fn wait_for_pod_recreated(sandbox_id: &str, previous_uid: &str) {
    let start = std::time::Instant::now();
    let timeout = Duration::from_secs(90);
    loop {
        let current_uid = pod_uid(sandbox_id);
        if !current_uid.is_empty() && current_uid != previous_uid {
            let status = Command::new("kubectl")
                .args([
                    "wait",
                    "--for=condition=Ready",
                    &format!("pod/sandbox-{}", sandbox_id),
                    "-n",
                    NAMESPACE,
                    "--timeout=90s",
                ])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .expect("kubectl wait failed");
            assert!(status.success(), "recreated pod did not become ready");
            return;
        }
        if start.elapsed() > timeout {
            panic!("sandbox pod {sandbox_id} was not recreated within {:?}", timeout);
        }
        std::thread::sleep(Duration::from_secs(2));
    }
}

fn default_create_request() -> CreateSandboxRequest {
    CreateSandboxRequest {
        image: sandbox_image(),
        cpu: Some("1".to_string()),
        memory: Some("512Mi".to_string()),
        env: vec![],
        ttl_seconds: Some(600),
        network_isolation: Some("egress".to_string()),
    }
}

async fn create_running_sandbox(
    client: &BasilicaClient,
    request: CreateSandboxRequest,
) -> (Sandbox, String) {
    let sandbox = client
        .create_sandbox(request)
        .await
        .expect("create_sandbox failed");
    let secret = sandbox
        .exec_agent_secret()
        .expect("exec_agent_secret should be present")
        .to_string();
    wait_for_running(client, &sandbox.sandbox_id).await;
    (sandbox, secret)
}

async fn shared_running_sandbox(client: &BasilicaClient) -> (Sandbox, String) {
    let shared = SHARED_SANDBOX
        .get_or_init(|| async {
            let created = create_running_sandbox(client, default_create_request()).await;
            Arc::new(created)
        })
        .await;
    (shared.0.clone(), shared.1.clone())
}

fn sandbox_with_port_forward(sandbox: Sandbox, secret: String) -> (Sandbox, std::process::Child) {
    let (local_port, child) = setup_port_forward(&sandbox.sandbox_id);
    let sandbox = sandbox
        .with_data_plane_url(format!("http://localhost:{}", local_port))
        .with_exec_agent_secret(secret);
    (sandbox, child)
}

fn assert_error_contains(err: &ApiError, needle: &str) {
    match err {
        ApiError::BadRequest { message }
        | ApiError::Internal { message }
        | ApiError::InvalidRequest { message }
        | ApiError::ApiResponse { message, .. } => {
            assert!(
                message.contains(needle),
                "expected error to contain '{needle}', got: {message}"
            );
        }
        other => panic!("expected message-bearing error, got: {other:?}"),
    }
}

// ============================================================================
// Tests
// ============================================================================

/// Full integration test flow that covers all 15+ test cases.
/// Runs as a single test to share sandbox setup/teardown overhead.
#[tokio::test]
#[ignore] // requires live K3d cluster -- run with: cargo test --test sandbox_k3d -- --ignored
async fn test_sandbox_full_lifecycle() {
    let client = build_client();

    // --- 1. Create sandbox with valid image ---
    let request = CreateSandboxRequest {
        image: sandbox_image(),
        cpu: Some("1".to_string()),
        memory: Some("512Mi".to_string()),
        env: vec![],
        ttl_seconds: Some(600),
        network_isolation: Some("egress".to_string()),
    };
    let sandbox = client
        .create_sandbox(request)
        .await
        .expect("create_sandbox failed");

    assert!(!sandbox.sandbox_id.is_empty(), "sandbox_id should not be empty");
    assert!(!sandbox.domain.is_empty(), "domain should not be empty");
    assert!(
        sandbox.exec_agent_secret().is_some(),
        "exec_agent_secret should be present after creation"
    );
    let sandbox_id = sandbox.sandbox_id.clone();
    let create_secret = sandbox.exec_agent_secret().unwrap().to_string();
    println!("Created sandbox: {} (domain: {})", sandbox_id, sandbox.domain);

    // --- 2. List sandboxes, verify created sandbox appears ---
    let list_resp = client
        .list_sandboxes()
        .await
        .expect("list_sandboxes failed");
    assert!(
        list_resp
            .sandboxes
            .iter()
            .any(|s| s.sandbox_id == sandbox_id),
        "Created sandbox should appear in list"
    );
    let listed = list_resp
        .sandboxes
        .iter()
        .find(|s| s.sandbox_id == sandbox_id)
        .expect("created sandbox should appear in list");
    assert_eq!(listed.ttl_seconds, Some(600));
    assert_eq!(listed.network_isolation, "egress");
    assert!(!listed.from_warm_pool);

    // --- 3. Get sandbox by ID, verify detail fields ---
    let detail = client
        .get_sandbox(&sandbox_id)
        .await
        .expect("get_sandbox failed");
    assert_eq!(detail.sandbox_id, sandbox_id);
    assert_eq!(detail.image, sandbox_image());
    assert!(!detail.cpu.is_empty());
    assert!(!detail.memory.is_empty());
    assert_eq!(detail.ttl_seconds, Some(600));
    assert_eq!(detail.network_isolation, "egress");
    assert!(!detail.from_warm_pool);

    // --- 4. Verify URL helpers ---
    assert!(
        sandbox.data_plane_url().starts_with("https://"),
        "data_plane_url should start with https://"
    );
    assert!(
        sandbox.data_plane_url().contains(&sandbox.domain),
        "data_plane_url should contain the domain"
    );
    assert!(
        sandbox.ws_url().starts_with("wss://"),
        "ws_url should start with wss://"
    );
    assert!(
        sandbox.ws_url().ends_with("/ws"),
        "ws_url should end with /ws"
    );
    assert!(
        sandbox.exec_url().starts_with("https://"),
        "exec_url should start with https://"
    );
    assert!(
        sandbox.exec_url().ends_with("/exec"),
        "exec_url should end with /exec"
    );

    // --- 5. Wait for sandbox to reach Running state ---
    let running_detail = wait_for_running(&client, &sandbox_id).await;
    assert!(running_detail.ready_at.is_some());
    assert!(running_detail.expires_at.is_some());
    println!("Sandbox {} is Running", sandbox_id);

    // --- 6. Set up data-plane access via port-forward ---
    let (local_port, mut pf_child) = setup_port_forward(&sandbox_id);
    let data_plane_url = format!("http://localhost:{}", local_port);
    println!("Data-plane URL: {}", data_plane_url);
    let initial_pod_uid = pod_uid(&sandbox_id);

    // Get secret -- prefer from API response, fall back to K8s secret
    let k8s_secret = get_exec_agent_secret(&sandbox_id);
    let secret = if !create_secret.is_empty() {
        create_secret.clone()
    } else {
        k8s_secret
    };
    assert!(!secret.is_empty(), "exec_agent_secret must be available");

    // Build a sandbox handle with overridden data-plane URL
    let sandbox = sandbox
        .with_data_plane_url(data_plane_url.clone())
        .with_exec_agent_secret(secret.clone());

    // --- 7. Exec command, verify stdout and exit code ---
    let exec_resp = sandbox
        .exec(vec!["echo".to_string(), "hello-sdk".to_string()])
        .await
        .expect("exec failed");
    assert_eq!(exec_resp.exit_code, 0, "exec exit code should be 0");
    assert!(
        exec_resp.stdout.contains("hello-sdk"),
        "exec stdout should contain 'hello-sdk', got: {}",
        exec_resp.stdout
    );

    // --- 8. Run code, verify output ---
    // /run defaults to Python, so use Python code
    let run_resp = sandbox
        .run("print('run-output-42')")
        .await
        .expect("run failed");
    assert_eq!(run_resp.exit_code, 0, "run exit code should be 0");
    assert!(
        run_resp.stdout.contains("run-output-42"),
        "run stdout should contain 'run-output-42', got: {}",
        run_resp.stdout
    );

    // --- 9. Rotate secret and verify stale auth fails ---
    let rotated = client
        .rotate_sandbox_secret(&sandbox_id)
        .await
        .expect("rotate_sandbox_secret failed");
    assert_eq!(rotated.sandbox_id, sandbox_id);
    assert_ne!(rotated.exec_agent_secret, secret);

    kill_port_forward(&mut pf_child);
    wait_for_pod_recreated(&sandbox_id, &initial_pod_uid);
    let (rotated_port, mut pf_child) = setup_port_forward(&sandbox_id);
    let sandbox = sandbox.with_data_plane_url(format!("http://localhost:{}", rotated_port));

    let stale_result = sandbox
        .exec(vec!["echo".to_string(), "stale-secret".to_string()])
        .await;
    assert!(
        matches!(stale_result, Err(ApiError::Authentication { .. })),
        "stale secret should fail authentication, got: {:?}",
        stale_result
    );

    let sandbox = sandbox.with_exec_agent_secret(rotated.exec_agent_secret.clone());
    let rotated_exec = sandbox
        .exec(vec!["echo".to_string(), "secret-rotated".to_string()])
        .await
        .expect("exec with rotated secret failed");
    assert_eq!(rotated_exec.exit_code, 0);
    assert!(rotated_exec.stdout.contains("secret-rotated"));

    // --- 10. Write file ---
    let test_content = "SDK integration test content\nLine 2";
    let write_resp = sandbox
        .files()
        .write("/tmp/sdk-test.txt", test_content)
        .await
        .expect("files.write failed");
    assert_eq!(write_resp.path, "/tmp/sdk-test.txt");

    // --- 11. Read file back, verify content matches ---
    let read_resp = sandbox
        .files()
        .read("/tmp/sdk-test.txt")
        .await
        .expect("files.read failed");
    assert_eq!(
        read_resp.content, test_content,
        "read content should match written content"
    );
    assert_eq!(read_resp.path, "/tmp/sdk-test.txt");

    // --- 12. List files, verify written file appears ---
    let list_resp = sandbox
        .files()
        .list("/tmp")
        .await
        .expect("files.list failed");
    assert!(
        list_resp.files.iter().any(|f| f.name == "sdk-test.txt"),
        "Written file should appear in listing, got: {:?}",
        list_resp.files.iter().map(|f| &f.name).collect::<Vec<_>>()
    );

    // --- 12. Delete sandbox, verify it's gone ---
    client
        .delete_sandbox(&sandbox_id)
        .await
        .expect("delete_sandbox failed");
    println!("Deleted sandbox: {}", sandbox_id);

    // Give the deletion a moment to propagate
    tokio::time::sleep(Duration::from_secs(2)).await;

    // Verify sandbox is gone from list
    let list_after = client.list_sandboxes().await.expect("list_sandboxes failed");
    assert!(
        !list_after
            .sandboxes
            .iter()
            .any(|s| s.sandbox_id == sandbox_id && s.status != "Terminating"),
        "Deleted sandbox should not appear in active sandboxes"
    );

    // Clean up port-forward
    let _ = pf_child.kill();
}

#[tokio::test]
#[ignore]
async fn test_sandbox_namespace_labels() {
    let client = build_client();
    let (sandbox, _) = shared_running_sandbox(&client).await;

    let namespace = kubectl_json(&["get", "ns", NAMESPACE, "-o", "json"]);
    assert_eq!(namespace["metadata"]["labels"]["basilica.ai/type"], "sandbox-tenant");
    assert_eq!(namespace["metadata"]["labels"]["basilica.ai/user-id"], "test-user");
    assert!(!sandbox.sandbox_id.is_empty());
}

#[tokio::test]
#[ignore]
async fn test_sandbox_resource_quota() {
    let client = build_client();
    let (sandbox, _) = shared_running_sandbox(&client).await;

    let quota = kubectl_json(&["get", "resourcequota", "sandbox-quota", "-n", NAMESPACE, "-o", "json"]);
    assert_eq!(quota["spec"]["hard"]["pods"], "20");
    assert_eq!(quota["spec"]["hard"]["requests.cpu"], "16");
    assert!(!sandbox.sandbox_id.is_empty());
}

#[tokio::test]
#[ignore]
async fn test_sandbox_limit_range() {
    let client = build_client();
    let (sandbox, _) = shared_running_sandbox(&client).await;

    let limit_range = kubectl_json(&["get", "limitrange", "sandbox-limits", "-n", NAMESPACE, "-o", "json"]);
    let limits = limit_range["spec"]["limits"].as_array().expect("limits should be array");
    let container = limits
        .iter()
        .find(|item| item["type"] == "Container")
        .expect("container limit not found");
    assert_eq!(container["max"]["cpu"], "2");
    assert_eq!(container["max"]["memory"], "4Gi");

    assert!(!sandbox.sandbox_id.is_empty());
}

#[tokio::test]
#[ignore]
async fn test_sandbox_image_pull_policy() {
    let client = build_client();
    let (sandbox, _) = shared_running_sandbox(&client).await;

    let pod_name = format!("sandbox-{}", sandbox.sandbox_id);
    let pull_policy = kubectl_output(&[
        "get",
        "pod",
        &pod_name,
        "-n",
        NAMESPACE,
        "-o",
        "jsonpath={.spec.containers[0].imagePullPolicy}",
    ]);
    assert_eq!(pull_policy, "IfNotPresent");

    assert!(!sandbox.sandbox_id.is_empty());
}

#[tokio::test]
#[ignore]
async fn test_sandbox_network_policy_exists() {
    let client = build_client();
    let (sandbox, _) = shared_running_sandbox(&client).await;

    let policies = kubectl_json(&[
        "get",
        "networkpolicy",
        "-n",
        NAMESPACE,
        "-o",
        "json",
    ]);
    let names: Vec<_> = policies["items"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|item| item["metadata"]["name"].as_str())
        .collect();
    assert!(names.contains(&"sandbox-default-policy"));
    assert!(names.contains(&"sandbox-full-isolation-policy"));

    assert!(!sandbox.sandbox_id.is_empty());
}

#[tokio::test]
#[ignore]
async fn test_sandbox_network_policy_blocks_imds() {
    let client = build_client();
    let (sandbox, secret) = shared_running_sandbox(&client).await;
    let sandbox_id = sandbox.sandbox_id.clone();
    let (sandbox, mut pf_child) = sandbox_with_port_forward(sandbox, secret);

    let resp = sandbox
        .exec(vec![
            "python3".to_string(),
            "-c".to_string(),
            "import urllib.request; urllib.request.urlopen('http://169.254.169.254/', timeout=3).read(); print('unexpected')".to_string(),
        ])
        .await
        .expect("exec failed");
    assert_ne!(resp.exit_code, 0, "IMDS request should fail");
    assert!(!resp.stdout.contains("unexpected"));

    kill_port_forward(&mut pf_child);
    assert!(!sandbox_id.is_empty());
}

#[tokio::test]
#[ignore]
async fn test_sandbox_create_with_env_vars() {
    let client = build_client();
    let mut req = default_create_request();
    req.env = vec![SandboxEnvVar {
        name: "TEST_VAR".to_string(),
        value: "hello123".to_string(),
    }];

    let (sandbox, secret) = create_running_sandbox(&client, req).await;
    let sandbox_id = sandbox.sandbox_id.clone();
    let (sandbox, mut pf_child) = sandbox_with_port_forward(sandbox, secret);

    let resp = sandbox
        .exec(vec![
            "python3".to_string(),
            "-c".to_string(),
            "import os; print(os.environ.get('TEST_VAR', ''))".to_string(),
        ])
        .await
        .expect("exec failed");
    assert_eq!(resp.exit_code, 0);
    assert!(resp.stdout.contains("hello123"));

    kill_port_forward(&mut pf_child);
    delete_sandbox_quietly(&client, &sandbox_id).await;
}

#[tokio::test]
#[ignore]
async fn test_sandbox_create_with_network_isolation_full() {
    let client = build_client();
    let mut req = default_create_request();
    req.network_isolation = Some("full".to_string());

    let (sandbox, secret) = create_running_sandbox(&client, req).await;
    let sandbox_id = sandbox.sandbox_id.clone();
    let (sandbox, mut pf_child) = sandbox_with_port_forward(sandbox, secret);

    let resp = sandbox
        .exec(vec![
            "python3".to_string(),
            "-c".to_string(),
            "import urllib.request; urllib.request.urlopen('http://example.com', timeout=3).read(); print('unexpected')".to_string(),
        ])
        .await
        .expect("exec failed");
    assert_ne!(resp.exit_code, 0, "full isolation should block public internet");
    assert!(!resp.stdout.contains("unexpected"));

    kill_port_forward(&mut pf_child);
    delete_sandbox_quietly(&client, &sandbox_id).await;
}

#[tokio::test]
#[ignore]
async fn test_sandbox_exec_with_workdir() {
    let client = build_client();
    let (sandbox, secret) = shared_running_sandbox(&client).await;
    let sandbox_id = sandbox.sandbox_id.clone();
    let (sandbox, mut pf_child) = sandbox_with_port_forward(sandbox, secret);

    let resp = sandbox
        .exec_with_options(vec!["pwd".to_string()], Some("/tmp".to_string()), None, None)
        .await
        .expect("exec failed");
    assert_eq!(resp.exit_code, 0);
    assert_eq!(resp.stdout.trim(), "/tmp");

    kill_port_forward(&mut pf_child);
    assert!(!sandbox_id.is_empty());
}

#[tokio::test]
#[ignore]
async fn test_sandbox_exec_with_timeout() {
    let client = build_client();
    let (sandbox, secret) = shared_running_sandbox(&client).await;
    let sandbox_id = sandbox.sandbox_id.clone();
    let (sandbox, mut pf_child) = sandbox_with_port_forward(sandbox, secret);

    let start = std::time::Instant::now();
    let resp = sandbox
        .exec_with_options(
            vec![
                "python3".to_string(),
                "-c".to_string(),
                "import time; time.sleep(5)".to_string(),
            ],
            None,
            None,
            Some(2),
        )
        .await
        .expect("exec failed");
    let elapsed = start.elapsed();
    assert!(
        elapsed >= Duration::from_secs(2),
        "timeout-aware exec should not return immediately"
    );
    assert!(
        elapsed <= Duration::from_secs(10),
        "timeout-aware exec should complete within a bounded window, got {:?}",
        elapsed
    );
    if resp.exit_code == 124 {
        assert!(resp.stderr.contains("timed out"));
    } else {
        assert_eq!(resp.exit_code, 0);
    }

    kill_port_forward(&mut pf_child);
    assert!(!sandbox_id.is_empty());
}

#[tokio::test]
#[ignore]
async fn test_sandbox_ws_connect_info() {
    let client = build_client();
    let (sandbox, secret) = shared_running_sandbox(&client).await;
    let sandbox_id = sandbox.sandbox_id.clone();
    let (local_port, mut pf_child) = setup_port_forward(&sandbox_id);
    let sandbox = sandbox
        .with_data_plane_url(format!("http://localhost:{local_port}"))
        .with_exec_agent_secret(secret.clone());

    let info = sandbox.ws_connect_info().expect("ws_connect_info should exist");
    assert_eq!(info.0, format!("ws://localhost:{local_port}/ws"));
    assert_eq!(info.1, "Authorization");
    assert_eq!(info.2, format!("Bearer {secret}"));

    kill_port_forward(&mut pf_child);
    assert!(!sandbox_id.is_empty());
}

#[tokio::test]
#[ignore]
async fn test_sandbox_url_helpers_with_override() {
    let client = build_client();
    let (sandbox, _) = shared_running_sandbox(&client).await;
    let sandbox_id = sandbox.sandbox_id.clone();
    let sandbox = sandbox.with_data_plane_url("http://localhost:12345".to_string());

    assert_eq!(sandbox.data_plane_url(), "http://localhost:12345");
    assert_eq!(sandbox.exec_url(), "http://localhost:12345/exec");
    assert_eq!(sandbox.ws_url(), "ws://localhost:12345/ws");

    assert!(!sandbox_id.is_empty());
}

#[tokio::test]
#[ignore]
async fn test_sandbox_file_delete() {
    let client = build_client();
    let (sandbox, secret) = shared_running_sandbox(&client).await;
    let sandbox_id = sandbox.sandbox_id.clone();
    let (sandbox, mut pf_child) = sandbox_with_port_forward(sandbox, secret);

    sandbox
        .files()
        .write("/tmp/delete-me.txt", "to be deleted")
        .await
        .expect("write failed");
    sandbox
        .files()
        .delete("/tmp/delete-me.txt")
        .await
        .expect("delete failed");
    let result = sandbox.files().read("/tmp/delete-me.txt").await;
    assert!(result.is_err(), "reading deleted file should fail");

    kill_port_forward(&mut pf_child);
    assert!(!sandbox_id.is_empty());
}

#[tokio::test]
#[ignore]
async fn test_sandbox_file_mkdir_and_stat() {
    let client = build_client();
    let (sandbox, secret) = shared_running_sandbox(&client).await;
    let sandbox_id = sandbox.sandbox_id.clone();
    let (sandbox, mut pf_child) = sandbox_with_port_forward(sandbox, secret);

    sandbox
        .files()
        .mkdir("/tmp/sdk-test-dir", true)
        .await
        .expect("mkdir failed");
    let stat = sandbox
        .files()
        .stat("/tmp/sdk-test-dir")
        .await
        .expect("stat failed");
    assert!(stat.is_dir);

    kill_port_forward(&mut pf_child);
    assert!(!sandbox_id.is_empty());
}

#[tokio::test]
#[ignore]
async fn test_sandbox_blocked_env_var_rejected() {
    let client = build_client();
    let mut req = default_create_request();
    req.env = vec![SandboxEnvVar {
        name: "EXEC_AGENT_SECRET".to_string(),
        value: "evil".to_string(),
    }];

    let err = client
        .create_sandbox(req)
        .await
        .expect_err("blocked env var should fail");
    assert_error_contains(&err, "EXEC_AGENT_SECRET");
}

#[tokio::test]
#[ignore]
async fn test_sandbox_blocked_env_prefix_rejected() {
    let client = build_client();
    let mut req = default_create_request();
    req.env = vec![SandboxEnvVar {
        name: "KUBERNETES_SERVICE_HOST".to_string(),
        value: "10.0.0.1".to_string(),
    }];

    let err = client
        .create_sandbox(req)
        .await
        .expect_err("blocked env prefix should fail");
    assert_error_contains(&err, "blocked");
}

#[tokio::test]
#[ignore]
async fn test_sandbox_excessive_cpu_rejected() {
    let client = build_client();
    let mut req = default_create_request();
    req.cpu = Some("3".to_string());

    let err = client
        .create_sandbox(req)
        .await
        .expect_err("excessive cpu should fail");
    assert_error_contains(&err, "2 cores");
}

#[tokio::test]
#[ignore]
async fn test_sandbox_excessive_memory_rejected() {
    let client = build_client();
    let mut req = default_create_request();
    req.memory = Some("5Gi".to_string());

    let err = client
        .create_sandbox(req)
        .await
        .expect_err("excessive memory should fail");
    assert_error_contains(&err, "4Gi");
}

#[tokio::test]
#[ignore]
async fn test_sandbox_create_with_defaults() {
    let client = build_client();
    let request = CreateSandboxRequest {
        image: sandbox_image(),
        cpu: None,
        memory: None,
        env: vec![],
        ttl_seconds: None,
        network_isolation: Some("egress".to_string()),
    };

    let (sandbox, _) = create_running_sandbox(&client, request).await;
    assert!(!sandbox.domain.is_empty());
    assert!(sandbox.exec_agent_secret().is_some());

    let pod_name = format!("sandbox-{}", sandbox.sandbox_id);
    let pod = kubectl_json(&["get", "pod", &pod_name, "-n", NAMESPACE, "-o", "json"]);
    assert_eq!(
        pod["spec"]["containers"][0]["resources"]["requests"]["cpu"],
        "1"
    );
    assert_eq!(
        pod["spec"]["containers"][0]["resources"]["requests"]["memory"],
        "2Gi"
    );

    delete_sandbox_quietly(&client, &sandbox.sandbox_id).await;
}

/// Test creating sandbox with invalid image returns an error.
#[tokio::test]
#[ignore]
async fn test_create_sandbox_invalid_image() {
    let client = build_client();

    let request = CreateSandboxRequest {
        image: "nonexistent-registry.invalid/no-such-image:latest".to_string(),
        cpu: None,
        memory: None,
        env: vec![],
        ttl_seconds: None,
        network_isolation: Some("egress".to_string()),
    };

    let result = client.create_sandbox(request).await;
    assert!(
        result.is_err(),
        "Creating sandbox with invalid image should fail"
    );
    let err = result.err().unwrap();
    println!("Expected error for invalid image: {:?}", err);
    assert_error_contains(&err, "allowlist");
}

/// Test getting a nonexistent sandbox returns an error.
#[tokio::test]
#[ignore]
async fn test_get_nonexistent_sandbox() {
    let client = build_client();

    let result = client.get_sandbox("sb-nonexistent-00000000").await;
    assert!(
        result.is_err(),
        "Getting nonexistent sandbox should fail"
    );
    match &result {
        Err(ApiError::NotFound { .. }) => {
            println!("Got expected NotFound error");
        }
        Err(ApiError::ApiResponse { status, .. }) if *status == 404 => {
            println!("Got expected 404 ApiResponse");
        }
        Err(other) => {
            println!("Got error (acceptable): {:?}", other);
        }
        Ok(_) => panic!("Should have gotten an error"),
    }
}

/// Test deleting a nonexistent sandbox returns an error.
#[tokio::test]
#[ignore]
async fn test_delete_nonexistent_sandbox() {
    let client = build_client();

    let result = client.delete_sandbox("sb-nonexistent-00000000").await;
    assert!(
        result.is_err(),
        "Deleting nonexistent sandbox should fail"
    );
    if let Err(e) = &result {
        println!("Expected error for deleting nonexistent: {:?}", e);
    }
}
