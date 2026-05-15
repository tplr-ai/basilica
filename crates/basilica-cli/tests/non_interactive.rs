//! Integration tests: every cost-bearing or state-mutating command in
//! `BASILICA_NON_INTERACTIVE` mode must surface a structured `MissingInput` /
//! `MissingPrerequisite` JSON error on stderr rather than blocking on stdin.

use assert_cmd::Command;

fn parse_stderr_json(bytes: &[u8]) -> serde_json::Value {
    let s = std::str::from_utf8(bytes).expect("stderr is utf-8");
    // The renderer writes one JSON object per line, possibly preceded by
    // unrelated log output. Find the first line that parses as JSON.
    for line in s.lines() {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(line) {
            return v;
        }
    }
    panic!("no JSON object found in stderr:\n{s}");
}

#[test]
fn ssh_keys_add_without_file_in_non_interactive_returns_missing_input() {
    let assert = Command::cargo_bin("basilica")
        .unwrap()
        .env("BASILICA_NON_INTERACTIVE", "1")
        .env_remove("RUST_LOG")
        .args(["--json", "ssh-keys", "add"])
        .assert()
        .failure();

    let v = parse_stderr_json(&assert.get_output().stderr);
    assert_eq!(v["schema_version"], 1);
    // When ~/.ssh has keys, the file-selector should produce MissingInput with
    // candidate paths. When it is empty, we get MissingInput on the same field
    // pointing the agent at --file. Either way the field is the same.
    assert_eq!(v["error"], "missing_input");
    assert_eq!(v["field"], "ssh_public_key_path");
    assert!(v["hint"].as_str().unwrap().contains("--file"));
}

#[test]
fn ssh_keys_add_with_invalid_file_in_non_interactive_emits_json_error() {
    // Pass --json globally; expect a single JSON object on stderr.
    let assert = Command::cargo_bin("basilica")
        .unwrap()
        .env("BASILICA_NON_INTERACTIVE", "1")
        .env_remove("RUST_LOG")
        .args([
            "--json",
            "ssh-keys",
            "add",
            "--file",
            "/this/does/not/exist.pub",
            "--name",
            "test",
        ])
        .assert()
        .failure();

    let v = parse_stderr_json(&assert.get_output().stderr);
    assert_eq!(v["schema_version"], 1);
    // Some error happened, and it was rendered as JSON. Either cli_error or
    // missing_input is acceptable here — the contract is "no hang, structured
    // stderr".
    assert!(v["error"].is_string());
}
