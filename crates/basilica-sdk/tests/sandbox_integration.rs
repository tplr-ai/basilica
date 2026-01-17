use basilica_sdk::sandbox::{Sandbox, SandboxConfig};
use std::env;
use std::time::Duration;

#[tokio::test]
async fn sandbox_integration_smoke() -> basilica_sdk::Result<()> {
    let api_url = match env::var("BASILICA_API_URL") {
        Ok(url) => url,
        Err(_) => {
            eprintln!("BASILICA_API_URL not set, skipping sandbox integration test");
            return Ok(());
        }
    };
    let api_token = match env::var("BASILICA_API_TOKEN") {
        Ok(token) => token,
        Err(_) => {
            eprintln!("BASILICA_API_TOKEN not set, skipping sandbox integration test");
            return Ok(());
        }
    };

    let sandbox = Sandbox::create(
        api_url,
        Some(api_token),
        SandboxConfig::new("python").with_runtime("container"),
    )
    .await?;
    sandbox.wait_until_ready(Duration::from_secs(300)).await?;

    let run_result = sandbox.run("print('hello')").await?;
    assert_eq!(run_result.exit_code, 0);
    assert!(run_result.stdout.contains("hello"));

    let exec_result = sandbox.exec(&["bash", "-lc", "echo exec-ok"]).await?;
    assert_eq!(exec_result.exit_code, 0);
    assert!(exec_result.stdout.contains("exec-ok"));

    sandbox
        .write_file("/workspace/test.txt", "rust-sdk")
        .await?;
    let content = sandbox.read_file("/workspace/test.txt").await?;
    assert_eq!(content, "rust-sdk");

    // TODO: Add snapshot/restore verification once snapshot API is stable in local clusters.

    sandbox.delete().await?;
    Ok(())
}

#[tokio::test]
async fn sandbox_rust_smoke() -> basilica_sdk::Result<()> {
    let api_url = match env::var("BASILICA_API_URL") {
        Ok(url) => url,
        Err(_) => {
            eprintln!("BASILICA_API_URL not set, skipping sandbox integration test");
            return Ok(());
        }
    };
    let api_token = match env::var("BASILICA_API_TOKEN") {
        Ok(token) => token,
        Err(_) => {
            eprintln!("BASILICA_API_TOKEN not set, skipping sandbox integration test");
            return Ok(());
        }
    };

    let sandbox = Sandbox::create(
        api_url,
        Some(api_token),
        SandboxConfig::new("rust").with_runtime("container"),
    )
    .await?;
    sandbox.wait_until_ready(Duration::from_secs(300)).await?;

    let program = r#"
        fn main() {
            println!("hello-rust");
        }
    "#;
    sandbox.write_file("/workspace/main.rs", program).await?;
    let exec_result = sandbox
        .exec(&[
            "bash",
            "-lc",
            "rustc /workspace/main.rs -o /workspace/app && /workspace/app",
        ])
        .await?;
    assert_eq!(exec_result.exit_code, 0);
    assert!(exec_result.stdout.contains("hello-rust"));

    // TODO: Add cargo-based build flow once sandbox images include a standard cargo cache volume.

    sandbox.delete().await?;
    Ok(())
}
