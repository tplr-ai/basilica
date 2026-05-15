//! Asserts that no subcommand hangs when stdin is closed and
//! `BASILICA_NON_INTERACTIVE=1` is set. New commands MUST be added to this
//! list as they ship; missing coverage will let regressions slip in.

use assert_cmd::Command;
use std::time::Duration;

const CASES: &[&[&str]] = &[
    &["ssh-keys", "add"],
    &["ssh-keys", "list"],
    &["balance"],
    // PR2 will append: up h100, down, ssh, exec, ...
    // PR3 will append: volumes create, tokens create, deploy delete, ...
];

#[test]
fn every_known_command_exits_within_5s_with_no_stdin() {
    for argv in CASES {
        let output = Command::cargo_bin("basilica")
            .unwrap()
            .env("BASILICA_NON_INTERACTIVE", "1")
            .env_remove("RUST_LOG")
            .args(*argv)
            .timeout(Duration::from_secs(5))
            .output()
            .unwrap_or_else(|e| panic!("hang-smoke for {argv:?}: {e}"));

        assert!(
            output.status.code().is_some(),
            "{argv:?} did not exit cleanly (likely hung): status={:?}",
            output.status,
        );
    }
}
