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

use basilica_sdk::{ApiError, BasilicaClient, ClientBuilder, CreateSandboxRequest};

const DEFAULT_API_URL: &str = "http://localhost:18082";
const SANDBOX_IMAGE: &str = "k3d-basilica-registry:5050/basilica-exec-agent:latest";
const NAMESPACE: &str = "u-test-user";

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

    // Start port-forward with port 0 (random local port)
    // Use :9999 to forward to exec-agent port
    let child = Command::new("kubectl")
        .args([
            "port-forward",
            "-n",
            NAMESPACE,
            &format!("pod/{}", pod_name),
            "0:9999",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("failed to start port-forward");

    // Give port-forward a moment to establish and read the assigned port
    std::thread::sleep(Duration::from_secs(2));

    // Find the actual local port by checking what port kubectl bound to
    // kubectl port-forward prints "Forwarding from 127.0.0.1:PORT -> 9999"
    // Since we can't easily read from the child's stdout without blocking,
    // use a fixed port range approach instead
    let local_port = 20000 + (std::process::id() as u16 % 10000);

    // Kill the random-port child and restart with a known port
    drop(child);
    std::thread::sleep(Duration::from_millis(500));

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
            "jsonpath={.data.exec-agent-secret}",
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
        image: SANDBOX_IMAGE.to_string(),
        cpu: Some("1".to_string()),
        memory: Some("512Mi".to_string()),
        env: vec![],
        ttl_seconds: Some(600),
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

    // --- 3. Get sandbox by ID, verify detail fields ---
    let detail = client
        .get_sandbox(&sandbox_id)
        .await
        .expect("get_sandbox failed");
    assert_eq!(detail.sandbox_id, sandbox_id);
    assert_eq!(detail.image, SANDBOX_IMAGE);
    assert!(!detail.cpu.is_empty());
    assert!(!detail.memory.is_empty());

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
    let _detail = wait_for_running(&client, &sandbox_id).await;
    println!("Sandbox {} is Running", sandbox_id);

    // --- 6. Set up data-plane access via port-forward ---
    let (local_port, mut pf_child) = setup_port_forward(&sandbox_id);
    let data_plane_url = format!("http://localhost:{}", local_port);
    println!("Data-plane URL: {}", data_plane_url);

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

    // --- 9. Write file ---
    let test_content = "SDK integration test content\nLine 2";
    let write_resp = sandbox
        .files()
        .write("/tmp/sdk-test.txt", test_content)
        .await
        .expect("files.write failed");
    assert_eq!(write_resp.path, "/tmp/sdk-test.txt");

    // --- 10. Read file back, verify content matches ---
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

    // --- 11. List files, verify written file appears ---
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
    };

    let result = client.create_sandbox(request).await;
    assert!(
        result.is_err(),
        "Creating sandbox with invalid image should fail"
    );
    if let Err(e) = &result {
        println!("Expected error for invalid image: {:?}", e);
    }
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
