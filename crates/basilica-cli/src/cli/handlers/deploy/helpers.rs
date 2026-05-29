//! Helper functions for deploy command

use crate::error::{CliError, DeployError};
use crate::output::{print_info, print_success};
use crate::progress::{complete_spinner_and_clear, create_spinner};
use basilica_sdk::types::{DeploymentResponse, DeploymentSummary};
use basilica_sdk::BasilicaClient;
use color_eyre::eyre::eyre;
use console::{style, Term};
use dialoguer::{theme::ColorfulTheme, Select};
use std::collections::HashMap;

/// Generate RFC 1123 compliant deployment name from source
pub fn generate_deployment_name(source: &str) -> String {
    let path = std::path::Path::new(source);
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("deployment");

    // Sanitize for DNS label (RFC 1123)
    let sanitized: String = stem
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
        .collect();

    // Trim leading/trailing hyphens
    let sanitized = sanitized.trim_matches('-');

    // Enforce max length (63 chars for DNS labels, minus UUID suffix)
    let max_prefix_len = 63 - 9; // 8 for UUID + 1 for dash
    let prefix = if sanitized.len() > max_prefix_len {
        &sanitized[..max_prefix_len]
    } else {
        sanitized
    };

    // Handle empty result
    let prefix = if prefix.is_empty() {
        "deployment"
    } else {
        prefix
    };

    format!("{}-{}", prefix, &uuid::Uuid::new_v4().to_string()[..8])
}

/// Parse KEY=VALUE environment variable strings
pub fn parse_env_vars(env: &[String]) -> Result<HashMap<String, String>, DeployError> {
    let mut map = HashMap::new();

    for entry in env {
        let mut parts = entry.splitn(2, '=');
        let key = parts.next().ok_or_else(|| DeployError::Validation {
            message: format!("Invalid env var format: '{}'", entry),
        })?;
        let value = parts.next().ok_or_else(|| DeployError::Validation {
            message: format!("Invalid env var format: '{}'. Use KEY=VALUE", entry),
        })?;

        map.insert(key.to_string(), value.to_string());
    }

    Ok(map)
}

/// Parse primary port from port specifications
pub fn parse_primary_port(ports: &[String]) -> Result<u16, DeployError> {
    let first_port = ports.first().ok_or_else(|| DeployError::Validation {
        message: "At least one port must be specified".to_string(),
    })?;

    let port_str = first_port.split(':').next().unwrap_or(first_port);
    port_str
        .parse::<u16>()
        .map_err(|_| DeployError::Validation {
            message: format!("Invalid port number: {}", port_str),
        })
}

/// Human label for a deployment: prefer `friendly_name`, fall back to `instance_name`
/// when the friendly name is empty (e.g. responses from older API versions where
/// the field is `#[serde(default)]`).
pub fn display_name<'a>(friendly: &'a str, instance: &'a str) -> &'a str {
    if friendly.is_empty() {
        instance
    } else {
        friendly
    }
}

/// Phase 4 (ADR-ISSUE-783-NVCT-CDI-ROBUSTNESS §7.2): build a one-line
/// parenthetical suffix describing the worst-case container state on a
/// deployment, derived from the API's `container_statuses[]` snapshot.
///
/// Returns `None` when every container is in `running` state, when the
/// snapshot is empty, or when the worst-case container has no
/// machine-readable reason. Returns `Some(" (...)")` (note the leading
/// space) when at least one container is waiting or terminated -- the
/// caller appends it directly to the phase line.
///
/// Examples:
/// - one container waiting CrashLoopBackOff, 3 restarts ->
///   `" (waiting: CrashLoopBackOff x3 restarts)"`
/// - one container waiting ImagePullBackOff, 0 restarts ->
///   `" (waiting: ImagePullBackOff)"`
/// - one container terminated OOMKilled, 5 restarts ->
///   `" (terminated: OOMKilled x5 restarts)"`
/// - phase=failed and >0 total restarts (no live snapshot) is rendered
///   via `container_restart_suffix` separately.
///
/// The "worst-case" selection prefers `terminated` > `waiting` >
/// `running` so the suffix surfaces the most user-actionable signal
/// first. Restart count uses the deployment-wide `phase_progress`
/// rollup so a multi-pod multi-container deployment shows the total
/// restart count regardless of which container's reason was selected.
pub fn container_status_suffix(response: &DeploymentResponse) -> Option<String> {
    if response.container_statuses.is_empty() {
        return None;
    }

    // Worst-case container: terminated outranks waiting outranks running.
    // Ties broken by highest restart_count.
    let worst = response.container_statuses.iter().max_by_key(|cs| {
        let state_rank = match cs.state.as_str() {
            "terminated" => 3,
            "waiting" => 2,
            "" => 1, // unobserved state, slightly above running
            _ => 0,  // "running"
        };
        (state_rank, cs.restart_count)
    })?;

    // Only render a suffix when the worst-case is not running.
    if worst.state == "running" {
        return None;
    }

    let reason = worst.reason.as_deref()?;
    let state_label = if worst.state.is_empty() {
        "waiting"
    } else {
        worst.state.as_str()
    };

    let total_restarts = response.phase_progress.max(0);
    let suffix = if total_restarts > 0 {
        format!(
            " ({}: {} x{} restarts)",
            state_label, reason, total_restarts
        )
    } else {
        format!(" ({}: {})", state_label, reason)
    };
    Some(suffix)
}

/// Phase 4 (ADR §7.2): for `phase=failed` deployments, surface the
/// total restart count even when `container_statuses[]` is empty
/// (e.g. operator has scaled the Deployment to 0 after the Failed TTL).
/// Returns `None` when `phase_progress == 0` or the phase is not
/// `failed`.
pub fn failed_restart_suffix(phase: &str, phase_progress: i32) -> Option<String> {
    if phase != "failed" || phase_progress <= 0 {
        return None;
    }
    Some(format!(" (x{} restarts)", phase_progress))
}

/// Resolved deployment identifier pair returned by `resolve_deployment_name`.
///
/// `instance_name` is the stable UUID the API expects in request paths.
/// `display_name` is what we surface to humans (friendly name, falling back to UUID).
pub struct ResolvedDeployment {
    pub instance_name: String,
    pub display_name: String,
}

/// Print summons success message
pub fn print_deployment_success(deployment: &DeploymentResponse) {
    let label = display_name(&deployment.friendly_name, &deployment.instance_name);
    print_success(&format!("Summons '{}' created successfully!", label));
    println!();
    println!("  URL:      {}", deployment.url);
    println!("  State:    {}", deployment.state);
    println!(
        "  Replicas: {}/{}",
        deployment.replicas.ready, deployment.replicas.desired
    );

    if let Some(ref phase) = deployment.phase {
        // Phase 4 (ADR §7.2): append the worst-case container state as a
        // parenthetical suffix when the underlying pods are not all
        // running; gives the user immediate insight instead of a bare
        // "Phase: starting" while the kubelet is CrashLoopBackOff'ing.
        let suffix = container_status_suffix(deployment)
            .or_else(|| failed_restart_suffix(phase, deployment.phase_progress))
            .unwrap_or_default();
        println!("  Phase:    {}{}", phase, suffix);
    }

    println!();
    println!("Commands:");
    println!("  View status:  basilica summon status {}", label);
    println!("  View logs:    basilica summon logs {}", label);
    println!("  Delete:       basilica summon delete {}", label);
}

/// Print share token information with security warning.
/// Called after creating a private deployment.
pub fn print_share_token_info(token: &str, share_url: &str) {
    println!();
    println!(
        "{}",
        style("Share Token (save this - cannot be retrieved later):")
            .yellow()
            .bold()
    );
    println!("  Token:     {}", style(token).cyan());
    println!("  Share URL: {}", style(share_url).cyan());
    println!();
    println!("{}", style("Access your deployment with:").dim());
    println!("  curl \"{}\"", share_url);
}

/// Print summons table
pub fn print_deployments_table(deployments: &[DeploymentSummary]) {
    if deployments.is_empty() {
        print_info("No summons found");
        return;
    }

    use tabled::{settings::Style, Table, Tabled};

    #[derive(Tabled)]
    struct Row {
        #[tabled(rename = "Name")]
        name: String,
        #[tabled(rename = "State")]
        state: String,
        #[tabled(rename = "Access")]
        access: String,
        #[tabled(rename = "Verified")]
        verified: String,
        #[tabled(rename = "Replicas")]
        replicas: String,
        #[tabled(rename = "URL")]
        url: String,
        #[tabled(rename = "Created")]
        created: String,
    }

    let rows: Vec<Row> = deployments
        .iter()
        .map(|d| Row {
            name: display_name(&d.friendly_name, &d.instance_name).to_string(),
            state: d.state.clone(),
            access: if d.public {
                "Public".to_string()
            } else {
                "Token".to_string()
            },
            verified: if d.public_metadata {
                "Yes".to_string()
            } else {
                "-".to_string()
            },
            replicas: format!("{}/{}", d.replicas.ready, d.replicas.desired),
            url: d.url.clone(),
            created: crate::output::table_output::format_timestamp(&d.created_at),
        })
        .collect();

    let mut table = Table::new(rows);
    table.with(Style::modern());
    println!("{table}");
}

/// Print summons details
pub fn print_deployment_details(deployment: &DeploymentResponse, verbose: bool) {
    println!(
        "Summons: {}",
        display_name(&deployment.friendly_name, &deployment.instance_name)
    );
    println!();
    println!("  Namespace:  {}", deployment.namespace);
    println!("  State:      {}", deployment.state);
    println!("  URL:        {}", deployment.url);
    println!(
        "  Replicas:   {}/{}",
        deployment.replicas.ready, deployment.replicas.desired
    );
    println!("  Created:    {}", deployment.created_at);
    if deployment.public_metadata {
        println!("  Public Metadata: Enrolled");
    } else {
        println!("  Public Metadata: Not enrolled");
    }

    if let Some(ref updated) = deployment.updated_at {
        println!("  Updated:    {}", updated);
    }

    if let Some(ref phase) = deployment.phase {
        // Phase 4 (ADR §7.2): same parenthetical-suffix treatment for the
        // verbose details view.
        let suffix = container_status_suffix(deployment)
            .or_else(|| failed_restart_suffix(phase, deployment.phase_progress))
            .unwrap_or_default();
        println!("  Phase:      {}{}", phase, suffix);
    }

    if verbose {
        if let Some(ref progress) = deployment.progress {
            println!();
            println!("Progress:");
            println!("  Current step: {}", progress.current_step);
            if let Some(pct) = progress.percentage {
                println!("  Progress:     {:.1}%", pct);
            }
            println!("  Elapsed:      {}s", progress.elapsed_seconds);
        }

        if let Some(ref pods) = deployment.pods {
            println!();
            println!("Pods:");
            for pod in pods {
                println!("  - {} ({})", pod.name, pod.status);
                if let Some(ref node) = pod.node {
                    println!("    Node: {}", node);
                }
            }
        }
    }
}

/// Sanitize ANSI escape sequences, allowing only SGR (color/style) codes.
///
/// SGR codes have the form `ESC [ <params> m` where params are semicolon-separated numbers.
/// All other escape sequences (OSC, CSI cursor movement, etc.) are stripped for security.
fn sanitize_ansi(input: &str) -> String {
    use std::fmt::Write;

    let mut result = String::with_capacity(input.len());
    let mut chars = input.chars().peekable();

    while let Some(c) = chars.next() {
        if c == '\x1b' {
            // Start of escape sequence
            match chars.peek() {
                Some('[') => {
                    chars.next(); // consume '['

                    // CSI sequence: collect parameter bytes and final byte
                    let mut seq_params = String::new();
                    let mut final_byte = None;

                    for ch in chars.by_ref() {
                        if ch.is_ascii_digit() || ch == ';' {
                            seq_params.push(ch);
                        } else if (0x40..=0x7e).contains(&(ch as u32)) {
                            // Final byte of CSI sequence
                            final_byte = Some(ch);
                            break;
                        } else {
                            // Intermediate byte or invalid - stop parsing
                            break;
                        }
                    }

                    // Only allow SGR sequences (final byte 'm')
                    if final_byte == Some('m') {
                        let _ = write!(result, "\x1b[{}m", seq_params);
                    }
                    // All other CSI sequences are dropped
                }
                Some(']') => {
                    chars.next(); // consume ']'

                    // OSC sequence: consume until BEL (\x07) or ST (ESC \)
                    loop {
                        match chars.next() {
                            Some('\x07') => break, // BEL terminates OSC
                            Some('\x1b') => {
                                // Check for ST (ESC \)
                                if chars.peek() == Some(&'\\') {
                                    chars.next();
                                }
                                break;
                            }
                            None => break, // End of input
                            _ => continue, // Consume OSC content
                        }
                    }
                    // OSC sequences are completely dropped
                }
                _ => {
                    // Other escape sequences (SS2, SS3, etc.) - drop the ESC
                    // but don't consume the next char as it might be regular text
                }
            }
        } else {
            result.push(c);
        }
    }

    result
}

/// Stream logs to stdout with proper SSE parsing
pub async fn stream_logs_to_stdout(
    response: reqwest::Response,
) -> Result<(), crate::error::CliError> {
    use eventsource_stream::Eventsource;
    use futures_util::StreamExt;
    use serde::Deserialize;

    #[derive(Debug, Deserialize)]
    struct LogEntry {
        message: String,
    }

    let stream = response.bytes_stream().eventsource();
    futures_util::pin_mut!(stream);

    while let Some(event) = stream.next().await {
        match event {
            Ok(sse_event) => {
                match serde_json::from_str::<LogEntry>(&sse_event.data) {
                    Ok(entry) => {
                        println!("{}", sanitize_ansi(&entry.message));
                    }
                    Err(_) => {
                        // Fall back to raw data if JSON parsing fails
                        if !sse_event.data.is_empty() {
                            println!("{}", sanitize_ansi(&sse_event.data));
                        }
                    }
                }
            }
            Err(e) => {
                return Err(crate::error::CliError::Internal(color_eyre::eyre::eyre!(
                    "Log stream error: {}",
                    e
                )));
            }
        }
    }

    Ok(())
}

/// Truncate string for display (unicode-safe)
fn truncate(s: &str, max_len: usize) -> String {
    if max_len == 0 {
        return String::new();
    }
    let char_count = s.chars().count();
    if char_count <= max_len {
        s.to_string()
    } else {
        let truncated: String = s.chars().take(max_len - 1).collect();
        format!("{}…", truncated)
    }
}

/// Resolve a user-supplied identifier to the UUID `instance_name` plus a human
/// `display_name`.
///
/// - If `name` is `Some`, list deployments and match `friendly_name` first, then fall
///   back to a direct `instance_name` (UUID) match. This lets users pass either form.
/// - If `name` is `None`, prompt the user to pick from a list rendered by friendly
///   name. The selected row's UUID is returned for API calls.
///
/// API request paths still use `instance_name`; `display_name` is for output only.
pub async fn resolve_deployment_name(
    name: Option<String>,
    client: &BasilicaClient,
) -> Result<ResolvedDeployment, CliError> {
    let spinner = create_spinner("Fetching summons...");
    let response = client.list_deployments().await.map_err(CliError::Api)?;
    complete_spinner_and_clear(spinner);

    if let Some(input) = name {
        if let Some(d) = response
            .deployments
            .iter()
            .find(|d| !d.friendly_name.is_empty() && d.friendly_name == input)
        {
            return Ok(ResolvedDeployment {
                instance_name: d.instance_name.clone(),
                display_name: display_name(&d.friendly_name, &d.instance_name).to_string(),
            });
        }
        if let Some(d) = response
            .deployments
            .iter()
            .find(|d| d.instance_name == input)
        {
            return Ok(ResolvedDeployment {
                instance_name: d.instance_name.clone(),
                display_name: display_name(&d.friendly_name, &d.instance_name).to_string(),
            });
        }
        return Err(CliError::Deploy(DeployError::NotFound { name: input }));
    }

    if response.deployments.is_empty() {
        return Err(CliError::Internal(eyre!(
            "No summons found. Create one with 'basilica summon <source>'"
        )));
    }

    // Format for selection display - render friendly name (falling back to UUID).
    let items: Vec<String> = response
        .deployments
        .iter()
        .map(|d| {
            format!(
                "{:<30} │ {:<10} │ {}/{}",
                truncate(display_name(&d.friendly_name, &d.instance_name), 30),
                d.state,
                d.replicas.ready,
                d.replicas.desired
            )
        })
        .collect();

    // Build prompt with header on next line
    let header = style("  Name                           │ State      │ Replicas").dim();
    let prompt = format!("Select summons\n{}", header);

    let selection = Select::with_theme(&ColorfulTheme::default())
        .with_prompt(prompt)
        .items(&items)
        .default(0)
        .interact_opt()
        .map_err(|e| CliError::Internal(eyre!("Selection failed: {}", e)))?;

    let selection = selection.ok_or_else(|| CliError::Internal(eyre!("Selection cancelled")))?;

    // Clear prompt (includes header) and selection
    let _ = Term::stdout().clear_last_lines(2);

    let chosen = &response.deployments[selection];
    Ok(ResolvedDeployment {
        instance_name: chosen.instance_name.clone(),
        display_name: display_name(&chosen.friendly_name, &chosen.instance_name).to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_deployment_name_sanitization() {
        // "My App_v2.py" -> file_stem="My App_v2" -> sanitized="my-app-v2" (space filtered, underscore to hyphen)
        let name = generate_deployment_name("My_App_v2.py");
        assert!(name.starts_with("my-app-v2-"));
        assert!(name.len() <= 63);
        assert!(name
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-'));
    }

    #[test]
    fn test_generate_deployment_name_long_input() {
        let long_source = format!("{}.py", "a".repeat(100));
        let name = generate_deployment_name(&long_source);
        assert!(name.len() <= 63);
    }

    #[test]
    fn test_parse_env_vars_valid() {
        let vars = vec!["KEY1=value1".to_string(), "KEY2=value2".to_string()];
        let result = parse_env_vars(&vars).unwrap();
        assert_eq!(result.get("KEY1"), Some(&"value1".to_string()));
        assert_eq!(result.get("KEY2"), Some(&"value2".to_string()));
    }

    #[test]
    fn test_parse_env_vars_with_equals_in_value() {
        let vars = vec!["URL=http://example.com?foo=bar".to_string()];
        let result = parse_env_vars(&vars).unwrap();
        assert_eq!(
            result.get("URL"),
            Some(&"http://example.com?foo=bar".to_string())
        );
    }

    #[test]
    fn test_parse_env_vars_invalid() {
        let vars = vec!["INVALID_NO_EQUALS".to_string()];
        assert!(parse_env_vars(&vars).is_err());
    }

    #[test]
    fn test_parse_primary_port_valid() {
        let ports = vec!["8000".to_string(), "9090".to_string()];
        assert_eq!(parse_primary_port(&ports).unwrap(), 8000);
    }

    #[test]
    fn test_parse_primary_port_with_name() {
        let ports = vec!["8000:http".to_string()];
        assert_eq!(parse_primary_port(&ports).unwrap(), 8000);
    }

    #[test]
    fn test_parse_primary_port_empty() {
        let ports: Vec<String> = vec![];
        assert!(parse_primary_port(&ports).is_err());
    }

    #[test]
    fn test_sanitize_ansi_allows_sgr_colors() {
        // SGR color codes should be preserved
        let input = "\x1b[1;36mHello\x1b[0m World";
        let output = sanitize_ansi(input);
        assert_eq!(output, "\x1b[1;36mHello\x1b[0m World");
    }

    #[test]
    fn test_sanitize_ansi_allows_reset() {
        let input = "\x1b[0mReset\x1b[m";
        let output = sanitize_ansi(input);
        assert_eq!(output, "\x1b[0mReset\x1b[m");
    }

    #[test]
    fn test_sanitize_ansi_strips_cursor_movement() {
        // CSI cursor movement (H, A, B, C, D, etc.) should be stripped
        let input = "\x1b[2;5HMoved cursor";
        let output = sanitize_ansi(input);
        assert_eq!(output, "Moved cursor");
    }

    #[test]
    fn test_sanitize_ansi_strips_clear_screen() {
        // CSI erase display (J) should be stripped
        let input = "\x1b[2JCleared screen";
        let output = sanitize_ansi(input);
        assert_eq!(output, "Cleared screen");
    }

    #[test]
    fn test_sanitize_ansi_strips_osc_sequences() {
        // OSC sequences (title change, hyperlinks, clipboard) should be stripped
        // OSC starts with ESC ] and ends with BEL or ST
        let input = "\x1b]0;Malicious Title\x07Normal text";
        let output = sanitize_ansi(input);
        assert_eq!(output, "Normal text");
    }

    #[test]
    fn test_sanitize_ansi_plain_text_unchanged() {
        let input = "Just plain text with no escape codes";
        let output = sanitize_ansi(input);
        assert_eq!(output, input);
    }

    #[test]
    fn test_display_name_prefers_friendly() {
        assert_eq!(display_name("my-app", "uuid-1234"), "my-app");
    }

    #[test]
    fn test_display_name_falls_back_to_instance_when_friendly_empty() {
        assert_eq!(display_name("", "uuid-1234"), "uuid-1234");
    }

    #[test]
    fn test_sanitize_ansi_mixed_content() {
        // Mix of allowed SGR and disallowed CSI
        let input = "\x1b[31mRed\x1b[0m \x1b[2JCleared \x1b[32mGreen\x1b[0m";
        let output = sanitize_ansi(input);
        assert_eq!(output, "\x1b[31mRed\x1b[0m Cleared \x1b[32mGreen\x1b[0m");
    }

    fn dr_with_containers(
        cs: Vec<basilica_sdk::types::ContainerStatusSnapshot>,
        phase_progress: i32,
    ) -> DeploymentResponse {
        DeploymentResponse {
            instance_name: "i".into(),
            friendly_name: "f".into(),
            user_id: "u".into(),
            namespace: "ns".into(),
            image: "img".into(),
            state: "Pending".into(),
            url: "".into(),
            replicas: basilica_sdk::types::ReplicaStatus {
                desired: 1,
                ready: 0,
            },
            created_at: "".into(),
            updated_at: None,
            pods: None,
            phase: Some("starting".into()),
            message: None,
            progress: None,
            share_token: None,
            share_url: None,
            websocket: None,
            public_metadata: false,
            distributed: None,
            container_statuses: cs,
            phase_progress,
        }
    }

    // Phase 4 (ADR §7.2): SDK suffix renderer must surface the parenthetical
    // only when the worst-case container is not running. Healthy paths must
    // return None so the existing "Phase: starting" output is unchanged.
    #[test]
    fn test_container_status_suffix_healthy_returns_none() {
        let cs = vec![basilica_sdk::types::ContainerStatusSnapshot {
            pod_name: "p1".into(),
            container_name: "c1".into(),
            state: "running".into(),
            reason: None,
            message: None,
            restart_count: 0,
        }];
        let dr = dr_with_containers(cs, 0);
        assert_eq!(container_status_suffix(&dr), None);
    }

    // Phase 4 (ADR §7.2): waiting + restarts -> include " x{} restarts".
    #[test]
    fn test_container_status_suffix_waiting_crashloop_with_restarts() {
        let cs = vec![basilica_sdk::types::ContainerStatusSnapshot {
            pod_name: "p1".into(),
            container_name: "c1".into(),
            state: "waiting".into(),
            reason: Some("CrashLoopBackOff".into()),
            message: None,
            restart_count: 3,
        }];
        let dr = dr_with_containers(cs, 3);
        assert_eq!(
            container_status_suffix(&dr),
            Some(" (waiting: CrashLoopBackOff x3 restarts)".to_string())
        );
    }

    // Phase 4 (ADR §7.2): waiting + 0 restarts -> no restart-count suffix.
    #[test]
    fn test_container_status_suffix_waiting_imagepullbackoff_no_restarts() {
        let cs = vec![basilica_sdk::types::ContainerStatusSnapshot {
            pod_name: "p1".into(),
            container_name: "c1".into(),
            state: "waiting".into(),
            reason: Some("ImagePullBackOff".into()),
            message: None,
            restart_count: 0,
        }];
        let dr = dr_with_containers(cs, 0);
        assert_eq!(
            container_status_suffix(&dr),
            Some(" (waiting: ImagePullBackOff)".to_string())
        );
    }

    // Phase 4 (ADR §7.2): terminated outranks waiting in the worst-case pick.
    #[test]
    fn test_container_status_suffix_terminated_outranks_waiting() {
        let cs = vec![
            basilica_sdk::types::ContainerStatusSnapshot {
                pod_name: "p1".into(),
                container_name: "c1".into(),
                state: "waiting".into(),
                reason: Some("ImagePullBackOff".into()),
                message: None,
                restart_count: 1,
            },
            basilica_sdk::types::ContainerStatusSnapshot {
                pod_name: "p2".into(),
                container_name: "c2".into(),
                state: "terminated".into(),
                reason: Some("OOMKilled".into()),
                message: None,
                restart_count: 5,
            },
        ];
        let dr = dr_with_containers(cs, 6);
        assert_eq!(
            container_status_suffix(&dr),
            Some(" (terminated: OOMKilled x6 restarts)".to_string())
        );
    }

    // Phase 4: empty containerStatuses[] returns None (no second-roundtrip surface).
    #[test]
    fn test_container_status_suffix_empty_returns_none() {
        let dr = dr_with_containers(vec![], 0);
        assert_eq!(container_status_suffix(&dr), None);
    }

    // Phase 4: failed phase with positive restart count even when container
    // snapshot is empty (operator scaled to 0).
    #[test]
    fn test_failed_restart_suffix_emits_when_failed_with_restarts() {
        assert_eq!(
            failed_restart_suffix("failed", 7),
            Some(" (x7 restarts)".to_string())
        );
    }

    // Phase 4: non-failed phases must NOT trigger the failed-restart suffix.
    #[test]
    fn test_failed_restart_suffix_omits_when_not_failed() {
        assert_eq!(failed_restart_suffix("ready", 7), None);
        assert_eq!(failed_restart_suffix("starting", 3), None);
    }

    // Phase 4: failed + zero restarts -> no suffix.
    #[test]
    fn test_failed_restart_suffix_omits_when_zero_restarts() {
        assert_eq!(failed_restart_suffix("failed", 0), None);
    }
}
