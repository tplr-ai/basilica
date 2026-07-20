//! Managed inference handlers for the Basilica CLI
//!
//! Thin wrappers over [`basilica_sdk::inference`] — the SDK owns the wire
//! contract, endpoint resolution, and typed errors (see
//! `MANAGED-INFERENCE-ENDPOINT-ARCHITECTURE.md` §4). This module only adds
//! clap dispatch, table/JSON rendering, and actionable error surfacing.
//!
//! Authentication reuses the CLI's existing context: a `basilica_...` API key
//! from `BASILICA_API_TOKEN` when set, otherwise the OAuth tokens from
//! `basilica login`. Both hosts accept either credential as a Bearer token.

use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use basilica_sdk::inference::{
    InferenceClient, InferenceError, InferenceModel, InferenceUsageRow, UsageQuery,
};
use color_eyre::eyre::eyre;
use console::style;
use rust_decimal::Decimal;
use tabled::{settings::Style, Table, Tabled};

use crate::cli::commands::{InferenceAction, InferenceCommand, ResolvedOutput};
use crate::config::CliConfig;
use crate::error::CliError;
use crate::output::{format_credits, json_output};
use crate::progress::{complete_spinner_and_clear, complete_spinner_error, create_spinner};

/// Environment variable holding a `basilica_...` API key
const API_TOKEN_ENV: &str = "BASILICA_API_TOKEN";

/// Hint shown when the inference gateway is unreachable
const GATEWAY_OVERRIDE_HINT: &str = "override it with --endpoint or BASILICA_INFERENCE_ENDPOINT";

/// Hint shown when the management plane (basilica-api) is unreachable.
/// (`BASILCA_` is the CLI config loader's actual env prefix — the missing
/// second "I" is a long-standing quirk of basilica-common.)
const API_OVERRIDE_HINT: &str = "set api.base_url in the CLI config or BASILCA_API__BASE_URL";

// ============================================================================
// Credential and client construction
// ============================================================================

/// Credential used to authenticate against the inference surface. Both the
/// gateway and the management plane accept either form as a Bearer token.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Credential {
    /// `basilica_...` API key (from `BASILICA_API_TOKEN`)
    ApiKey(String),
    /// OAuth tokens (from `basilica login`)
    Jwt {
        access_token: String,
        refresh_token: String,
    },
}

/// Pick the credential to use: an explicit API key wins over the OAuth login
fn credential_from_parts(
    api_token: Option<String>,
    jwt: Option<(String, String)>,
) -> Option<Credential> {
    api_token
        .filter(|t| !t.trim().is_empty())
        .map(Credential::ApiKey)
        .or_else(|| {
            jwt.map(|(access_token, refresh_token)| Credential::Jwt {
                access_token,
                refresh_token,
            })
        })
}

/// Resolve the active credential from the environment or the token store
async fn resolve_credential(config: &CliConfig) -> Result<Credential, CliError> {
    let api_token = std::env::var(API_TOKEN_ENV).ok();
    if let Some(credential) = credential_from_parts(api_token, None) {
        return Ok(credential);
    }

    // Falls back to the OAuth login; surfaces AuthError::UserNotLoggedIn
    // (which triggers the CLI's login-retry flow) when no tokens are stored.
    let tokens = crate::client::get_valid_jwt_tokens(config).await?;
    credential_from_parts(None, Some((tokens.access_token, tokens.refresh_token)))
        .ok_or_else(|| CliError::Internal(eyre!("No usable credential found")))
}

/// Build the SDK client from CLI inputs.
///
/// The management-plane base URL comes from the CLI config (same as every
/// other command); the gateway endpoint resolution — `--endpoint` flag >
/// `BASILICA_INFERENCE_ENDPOINT` env > built-in default — is the SDK's job,
/// so the flag is passed through untouched.
fn build_inference_client(
    endpoint_flag: Option<&str>,
    config: &CliConfig,
    credential: &Credential,
) -> Result<InferenceClient, InferenceError> {
    let mut builder = InferenceClient::builder().api_base(config.api.base_url.clone());

    if let Some(endpoint) = endpoint_flag.filter(|e| !e.trim().is_empty()) {
        builder = builder.endpoint(endpoint);
    }

    builder = match credential {
        Credential::ApiKey(api_key) => builder.with_api_key(api_key),
        Credential::Jwt {
            access_token,
            refresh_token,
        } => builder.with_tokens(access_token.clone(), refresh_token.clone()),
    };

    builder.build()
}

// ============================================================================
// Error surfacing (typed SDK errors -> actionable CLI messages)
// ============================================================================

/// Map a typed SDK error to an actionable CLI error. The SDK already carries
/// the contract detail (status, cap, retry hint); the CLI adds the next step.
fn map_inference_error(error: &InferenceError, connectivity_hint: &str) -> CliError {
    let message = match error {
        InferenceError::Auth => format!(
            "{error} Run `basilica login` (or `basilica login --device-code` on headless \
             hosts), or set {API_TOKEN_ENV} to a valid API key from `basilica tokens create`."
        ),
        InferenceError::Forbidden => {
            format!("{error} Create a fresh API key with `basilica tokens create`.")
        }
        InferenceError::InsufficientCredits => {
            format!("{error} — top up with `basilica fund`.")
        }
        InferenceError::QuotaExceeded { cap, retry_after } => {
            let retry = retry_after
                .map(|secs| format!(" Retry in {secs}s."))
                .unwrap_or_default();
            format!("Quota exceeded (429): the {cap} cap tripped.{retry} Slow down or reduce concurrency.")
        }
        InferenceError::Unavailable { retry_after } => {
            let retry = retry_after
                .map(|secs| format!(" Retry in {secs}s."))
                .unwrap_or_default();
            format!("{error}.{retry} Try again; if this persists, the service may be temporarily unavailable.")
        }
        InferenceError::Transport(_) => format!(
            "{error} Check your network connection and that the endpoint is correct \
             ({connectivity_hint})."
        ),
        InferenceError::ModelNotFound { .. }
        | InferenceError::Invalid(_)
        | InferenceError::Decode(_) => error.to_string(),
    };

    CliError::Internal(eyre!(message))
}

// ============================================================================
// Table rendering
// ============================================================================

#[derive(Tabled)]
struct ModelTableRow {
    #[tabled(rename = "ID")]
    id: String,
    #[tabled(rename = "Owned By")]
    owned_by: String,
}

/// Render the model list as a table (pure; golden-tested)
fn render_models_table(models: &[InferenceModel]) -> String {
    let rows: Vec<ModelTableRow> = models
        .iter()
        .map(|m| ModelTableRow {
            id: m.id.clone(),
            owned_by: m.owned_by.clone().unwrap_or_else(|| "-".to_string()),
        })
        .collect();

    let mut table = Table::new(rows);
    table.with(Style::modern());
    table.to_string()
}

#[derive(Tabled)]
struct UsageTableRow {
    #[tabled(rename = "Date")]
    date: String,
    #[tabled(rename = "Model")]
    model: String,
    #[tabled(rename = "Tokens In")]
    tokens_in: String,
    #[tabled(rename = "Tokens Out")]
    tokens_out: String,
    #[tabled(rename = "Cached")]
    cached: String,
    #[tabled(rename = "Charge (credits)")]
    charge: String,
}

/// Render usage rows with a totals footer (pure; golden-tested)
fn render_usage_table(rows: &[InferenceUsageRow]) -> String {
    let mut prompt_total = 0u64;
    let mut completion_total = 0u64;
    let mut cached_total = 0u64;
    let mut charge_total = Decimal::ZERO;

    let mut table_rows: Vec<UsageTableRow> = rows
        .iter()
        .map(|row| {
            prompt_total = prompt_total.saturating_add(row.prompt_tokens);
            completion_total = completion_total.saturating_add(row.completion_tokens);
            cached_total = cached_total.saturating_add(row.cached_tokens);
            charge_total += row.charge_credits;

            UsageTableRow {
                date: row.date.format("%Y-%m-%d").to_string(),
                model: row.model.clone(),
                tokens_in: row.prompt_tokens.to_string(),
                tokens_out: row.completion_tokens.to_string(),
                cached: row.cached_tokens.to_string(),
                charge: format_credits(&row.charge_credits.to_string()),
            }
        })
        .collect();

    table_rows.push(UsageTableRow {
        date: "TOTAL".to_string(),
        model: String::new(),
        tokens_in: prompt_total.to_string(),
        tokens_out: completion_total.to_string(),
        cached: cached_total.to_string(),
        charge: format!("{charge_total:.2}"),
    });

    let mut table = Table::new(table_rows);
    table.with(Style::modern());
    table.to_string()
}

// ============================================================================
// Command handlers
// ============================================================================

/// Entry point for the `basilica inference` command group
pub async fn handle_inference(
    cmd: InferenceCommand,
    config: &CliConfig,
    json: bool,
) -> Result<(), CliError> {
    match &cmd.action {
        InferenceAction::Models { output } => {
            let output = ResolvedOutput::resolve(json, *output)
                .map_err(|message| CliError::Internal(eyre!(message)))?;
            handle_models(&cmd, config, output).await
        }
        InferenceAction::Usage {
            from,
            to,
            model,
            kid,
            output,
        } => {
            let output = ResolvedOutput::resolve(json, *output)
                .map_err(|message| CliError::Internal(eyre!(message)))?;
            let query = UsageQuery {
                from: *from,
                to: *to,
                model: model.clone(),
                kid: kid.clone(),
            };
            handle_usage(&cmd, config, query, output).await
        }
        InferenceAction::Whoami => handle_whoami(&cmd, config, json).await,
    }
}

/// `basilica inference models` — list the tenant-visible model catalog
async fn handle_models(
    cmd: &InferenceCommand,
    config: &CliConfig,
    output: ResolvedOutput,
) -> Result<(), CliError> {
    let credential = resolve_credential(config).await?;
    let client = build_inference_client(cmd.endpoint.as_deref(), config, &credential)
        .map_err(|e| map_inference_error(&e, GATEWAY_OVERRIDE_HINT))?;

    let spinner = create_spinner("Fetching inference models...");
    let models = match client.list_models().await {
        Ok(models) => models,
        Err(e) => {
            complete_spinner_error(spinner.clone(), "Failed to fetch models");
            return Err(map_inference_error(&e, GATEWAY_OVERRIDE_HINT));
        }
    };
    complete_spinner_and_clear(spinner);

    if output.is_json() {
        return json_output(&models).map_err(CliError::Internal);
    }

    if models.is_empty() {
        println!("No models available on the inference gateway.");
    } else {
        println!("{}", render_models_table(&models));
    }

    Ok(())
}

/// `basilica inference usage` — usage rollups from the management plane
async fn handle_usage(
    cmd: &InferenceCommand,
    config: &CliConfig,
    query: UsageQuery,
    output: ResolvedOutput,
) -> Result<(), CliError> {
    let credential = resolve_credential(config).await?;
    let client = build_inference_client(cmd.endpoint.as_deref(), config, &credential)
        .map_err(|e| map_inference_error(&e, API_OVERRIDE_HINT))?;

    let spinner = create_spinner("Fetching inference usage...");
    let rows = match client.usage(query).await {
        Ok(rows) => rows,
        Err(e) => {
            complete_spinner_error(spinner.clone(), "Failed to fetch usage");
            return Err(map_inference_error(&e, API_OVERRIDE_HINT));
        }
    };
    complete_spinner_and_clear(spinner);

    if output.is_json() {
        return json_output(&rows).map_err(CliError::Internal);
    }

    if rows.is_empty() {
        println!("No inference usage found for the given filters.");
    } else {
        println!("{}", render_usage_table(&rows));
    }

    Ok(())
}

/// `basilica inference whoami` — resolved endpoint, credential, and tenant
async fn handle_whoami(
    cmd: &InferenceCommand,
    config: &CliConfig,
    json: bool,
) -> Result<(), CliError> {
    let credential = resolve_credential(config).await?;
    let client = build_inference_client(cmd.endpoint.as_deref(), config, &credential)
        .map_err(|e| map_inference_error(&e, GATEWAY_OVERRIDE_HINT))?;

    // Tenant identity is the Auth0 subject, recoverable offline from the JWT.
    // With an API key, fall back to the stored login token when available.
    let (tenant, auth_kind, kid) = match &credential {
        Credential::Jwt { access_token, .. } => (
            decode_jwt_sub(access_token),
            "OAuth login (JWT)".to_string(),
            None,
        ),
        Credential::ApiKey(token) => {
            let tenant = crate::client::get_valid_jwt_tokens(config)
                .await
                .ok()
                .and_then(|tokens| decode_jwt_sub(&tokens.access_token));
            (
                tenant,
                format!("API key ({API_TOKEN_ENV})"),
                decode_api_key_kid(token),
            )
        }
    };

    // Best-effort: match the kid against the account's keys to find its alias
    let key_alias = match (&credential, &kid) {
        (Credential::ApiKey(_), Some(kid)) => lookup_key_alias(config, kid).await,
        _ => None,
    };

    if json {
        return json_output(&serde_json::json!({
            "inference_endpoint": client.endpoint(),
            "openai_base_url": client.openai_base_url(),
            "api_base_url": client.api_base(),
            "auth": auth_kind,
            "kid": kid,
            "key_alias": key_alias,
            "tenant": tenant,
        }))
        .map_err(CliError::Internal);
    }

    println!("{}", style("Inference").bold());
    println!("  {}: {}", style("Endpoint").cyan(), client.endpoint());
    println!(
        "  {}: {}",
        style("OpenAI base URL").cyan(),
        client.openai_base_url()
    );
    println!("  {}: {}", style("API base").cyan(), client.api_base());
    println!("  {}: {}", style("Auth").cyan(), auth_kind);
    if let Some(kid) = &kid {
        let alias = key_alias
            .as_ref()
            .map(|a| format!(" (alias: {a})"))
            .unwrap_or_default();
        println!("  {}: {kid}{alias}", style("Key ID").cyan());
    }
    println!(
        "  {}: {}",
        style("Tenant").cyan(),
        tenant.unwrap_or_else(|| "resolved server-side per request".to_string())
    );

    Ok(())
}

/// Decode the key ID (kid) from a `basilica_...` API key.
///
/// The token body is base64url(kid (16 bytes) || secret (32 bytes)); the kid
/// is conventionally rendered as 32 lowercase hex characters.
fn decode_api_key_kid(token: &str) -> Option<String> {
    let body = token.strip_prefix("basilica_")?;
    let bytes = URL_SAFE_NO_PAD.decode(body).ok()?;
    if bytes.len() < 16 {
        return None;
    }
    Some(bytes[..16].iter().map(|b| format!("{b:02x}")).collect())
}

/// Extract the `sub` claim from a JWT without verifying the signature
/// (display only; the server remains the authority)
fn decode_jwt_sub(token: &str) -> Option<String> {
    let payload = token.split('.').nth(1)?;
    let decoded = URL_SAFE_NO_PAD.decode(payload).ok()?;
    let json: serde_json::Value = serde_json::from_slice(&decoded).ok()?;
    json.get("sub")?.as_str().map(str::to_string)
}

/// Best-effort lookup of an API key's name (alias) by kid; needs a login session
async fn lookup_key_alias(config: &CliConfig, kid: &str) -> Option<String> {
    let client = crate::client::create_authenticated_client(config)
        .await
        .ok()?;
    client
        .list_api_keys()
        .await
        .ok()?
        .into_iter()
        .find(|key| key.kid == kid)
        .map(|key| key.name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::args::Args;
    use basilica_sdk::inference::{QuotaCap, INFERENCE_ENDPOINT_ENV};
    use clap::Parser;

    // ------------------------------------------------------------------
    // Argument parsing
    // ------------------------------------------------------------------

    fn parse(args: &[&str]) -> Result<Args, clap::Error> {
        Args::try_parse_from(args)
    }

    #[test]
    fn parses_models_subcommand() {
        let args = parse(&["basilica", "inference", "models"]).unwrap();
        match args.command {
            crate::cli::commands::Commands::Inference(cmd) => {
                assert!(matches!(
                    cmd.action,
                    InferenceAction::Models { output: None }
                ));
                assert_eq!(cmd.endpoint, None);
            }
            other => panic!("expected inference command, got {other:?}"),
        }
    }

    #[test]
    fn parses_usage_subcommand_with_all_flags() {
        let args = parse(&[
            "basilica",
            "--json",
            "inference",
            "usage",
            "--from",
            "2026-07-01",
            "--to",
            "2026-07-20",
            "--model",
            "llama-3.1-70b-instruct",
            "--kid",
            "0123456789abcdef0123456789abcdef",
            "--endpoint",
            "http://localhost:9999",
        ])
        .unwrap();

        match args.command {
            crate::cli::commands::Commands::Inference(cmd) => {
                match cmd.action {
                    InferenceAction::Usage {
                        from,
                        to,
                        model,
                        kid,
                        output,
                    } => {
                        assert_eq!(
                            from.unwrap(),
                            chrono::NaiveDate::from_ymd_opt(2026, 7, 1).unwrap()
                        );
                        assert_eq!(
                            to.unwrap(),
                            chrono::NaiveDate::from_ymd_opt(2026, 7, 20).unwrap()
                        );
                        assert_eq!(model.as_deref(), Some("llama-3.1-70b-instruct"));
                        assert_eq!(kid.as_deref(), Some("0123456789abcdef0123456789abcdef"));
                        assert!(output.is_none());
                    }
                    other => panic!("expected usage action, got {other:?}"),
                }
                assert_eq!(cmd.endpoint.as_deref(), Some("http://localhost:9999"));
            }
            other => panic!("expected inference command, got {other:?}"),
        }
        assert!(args.json);
    }

    #[test]
    fn endpoint_flag_is_global_across_subcommands() {
        let args = parse(&[
            "basilica",
            "inference",
            "--endpoint",
            "http://localhost:9999",
            "whoami",
        ])
        .unwrap();
        match args.command {
            crate::cli::commands::Commands::Inference(cmd) => {
                assert!(matches!(cmd.action, InferenceAction::Whoami));
                assert_eq!(cmd.endpoint.as_deref(), Some("http://localhost:9999"));
            }
            other => panic!("expected inference command, got {other:?}"),
        }
    }

    #[test]
    fn rejects_invalid_dates() {
        for bad in ["not-a-date", "2026-13-01", "2026-02-30", "2026/07/01"] {
            let result = parse(&["basilica", "inference", "usage", "--from", bad]);
            assert!(result.is_err(), "expected '{bad}' to be rejected");
        }
    }

    #[test]
    fn rejects_conflicting_json_and_output_flags() {
        let err = ResolvedOutput::resolve(true, Some(crate::cli::commands::OutputFormat::Wide));
        assert!(err.is_err());
        let ok = ResolvedOutput::resolve(true, None).unwrap();
        assert!(ok.is_json());
    }

    // ------------------------------------------------------------------
    // Endpoint flag pass-through (SDK owns resolution; CLI forwards)
    // ------------------------------------------------------------------

    fn test_config() -> CliConfig {
        CliConfig {
            api: crate::config::ApiConfig {
                base_url: "http://api.local:8000".to_string(),
                request_timeout: 30,
            },
            ..Default::default()
        }
    }

    fn test_credential() -> Credential {
        Credential::ApiKey("basilica_test-key".to_string())
    }

    #[test]
    fn endpoint_flag_is_passed_through_to_the_sdk() {
        let client = build_inference_client(
            Some("http://gateway.local:9999/"),
            &test_config(),
            &test_credential(),
        )
        .unwrap();
        assert_eq!(client.endpoint(), "http://gateway.local:9999");
        assert_eq!(client.openai_base_url(), "http://gateway.local:9999/v1");
        // api_base always comes from the CLI config
        assert_eq!(client.api_base(), "http://api.local:8000");
    }

    #[test]
    fn no_flag_leaves_resolution_to_the_sdk() {
        // Without a flag the SDK applies env > default; the test environment
        // must not have BASILICA_INFERENCE_ENDPOINT set for the default arm.
        std::env::remove_var(INFERENCE_ENDPOINT_ENV);
        let client = build_inference_client(None, &test_config(), &test_credential()).unwrap();
        assert_eq!(
            client.endpoint(),
            basilica_sdk::inference::DEFAULT_INFERENCE_ENDPOINT
        );
    }

    #[test]
    fn blank_flag_is_treated_as_unset() {
        std::env::remove_var(INFERENCE_ENDPOINT_ENV);
        let client =
            build_inference_client(Some("  "), &test_config(), &test_credential()).unwrap();
        assert_eq!(
            client.endpoint(),
            basilica_sdk::inference::DEFAULT_INFERENCE_ENDPOINT
        );
    }

    // ------------------------------------------------------------------
    // Credential resolution (pure parts)
    // ------------------------------------------------------------------

    #[test]
    fn api_key_wins_over_jwt() {
        let jwt = Some(("access".to_string(), "refresh".to_string()));
        let credential = credential_from_parts(Some("basilica_key".to_string()), jwt.clone());
        assert_eq!(credential, Some(Credential::ApiKey("basilica_key".into())));

        let credential = credential_from_parts(None, jwt.clone());
        assert_eq!(
            credential,
            Some(Credential::Jwt {
                access_token: "access".into(),
                refresh_token: "refresh".into(),
            })
        );

        // Blank env token is ignored
        let credential = credential_from_parts(Some("  ".to_string()), jwt);
        assert!(matches!(credential, Some(Credential::Jwt { .. })));

        assert_eq!(credential_from_parts(None, None), None);
    }

    #[test]
    fn decodes_api_key_kid() {
        // kid = 0x00..0x0f, secret = 32 zero bytes
        let mut raw = (0u8..16).collect::<Vec<_>>();
        raw.extend([0u8; 32]);
        let token = format!("basilica_{}", URL_SAFE_NO_PAD.encode(&raw));
        assert_eq!(
            decode_api_key_kid(&token).as_deref(),
            Some("000102030405060708090a0b0c0d0e0f")
        );

        assert!(decode_api_key_kid("wrong_prefix").is_none());
        assert!(decode_api_key_kid("basilica_!!!").is_none());
        assert!(decode_api_key_kid("basilica_c2hvcnQ").is_none()); // too short
    }

    #[test]
    fn decodes_jwt_sub() {
        let payload = URL_SAFE_NO_PAD.encode(br#"{"sub":"github|434149"}"#);
        let token = format!("header.{payload}.signature");
        assert_eq!(decode_jwt_sub(&token).as_deref(), Some("github|434149"));
        assert!(decode_jwt_sub("not-a-jwt").is_none());
    }

    // ------------------------------------------------------------------
    // Output formatting (golden tables, JSON)
    // ------------------------------------------------------------------

    fn fixture_models() -> Vec<InferenceModel> {
        vec![
            InferenceModel {
                id: "llama-3.1-70b-instruct".to_string(),
                object: Some("model".to_string()),
                created: Some(1_700_000_000),
                owned_by: Some("basilica".to_string()),
            },
            InferenceModel {
                id: "qwen-2.5-32b".to_string(),
                object: Some("model".to_string()),
                created: Some(1_700_000_100),
                owned_by: None,
            },
        ]
    }

    fn fixture_usage_rows() -> Vec<InferenceUsageRow> {
        vec![
            InferenceUsageRow {
                date: chrono::NaiveDate::from_ymd_opt(2026, 7, 19).unwrap(),
                model: "llama-3.1-70b-instruct".to_string(),
                tenant_id: Some("github|434149".to_string()),
                kid: Some("0123456789abcdef0123456789abcdef".to_string()),
                prompt_tokens: 1200,
                completion_tokens: 340,
                cached_tokens: 100,
                charge_credits: "1.23456".parse().unwrap(),
            },
            InferenceUsageRow {
                date: chrono::NaiveDate::from_ymd_opt(2026, 7, 20).unwrap(),
                model: "qwen-2.5-32b".to_string(),
                tenant_id: Some("github|434149".to_string()),
                kid: None,
                prompt_tokens: 800,
                completion_tokens: 160,
                cached_tokens: 0,
                charge_credits: "0.00544".parse().unwrap(),
            },
        ]
    }

    #[test]
    fn models_table_golden() {
        let rendered = render_models_table(&fixture_models());
        let expected = "┌────────────────────────┬──────────┐\n\
                        │ ID                     │ Owned By │\n\
                        ├────────────────────────┼──────────┤\n\
                        │ llama-3.1-70b-instruct │ basilica │\n\
                        ├────────────────────────┼──────────┤\n\
                        │ qwen-2.5-32b           │ -        │\n\
                        └────────────────────────┴──────────┘";
        assert_eq!(rendered, expected);
    }

    #[test]
    fn usage_table_golden_with_totals() {
        let rendered = render_usage_table(&fixture_usage_rows());
        let expected = "┌────────────┬────────────────────────┬───────────┬────────────┬────────┬──────────────────┐\n\
                        │ Date       │ Model                  │ Tokens In │ Tokens Out │ Cached │ Charge (credits) │\n\
                        ├────────────┼────────────────────────┼───────────┼────────────┼────────┼──────────────────┤\n\
                        │ 2026-07-19 │ llama-3.1-70b-instruct │ 1200      │ 340        │ 100    │ 1.23             │\n\
                        ├────────────┼────────────────────────┼───────────┼────────────┼────────┼──────────────────┤\n\
                        │ 2026-07-20 │ qwen-2.5-32b           │ 800       │ 160        │ 0      │ 0.01             │\n\
                        ├────────────┼────────────────────────┼───────────┼────────────┼────────┼──────────────────┤\n\
                        │ TOTAL      │                        │ 2000      │ 500        │ 100    │ 1.24             │\n\
                        └────────────┴────────────────────────┴───────────┴────────────┴────────┴──────────────────┘";
        assert_eq!(rendered, expected);
    }

    #[test]
    fn usage_rows_serialize_to_the_wire_shape_for_json_output() {
        // The --json path prints the typed SDK rows; their serialized shape
        // must carry the wire field names (decimal-string charge included).
        let rows = fixture_usage_rows();
        let json = serde_json::to_value(&rows).unwrap();
        let first = &json[0];
        assert_eq!(first["date"], "2026-07-19");
        assert_eq!(first["model"], "llama-3.1-70b-instruct");
        assert_eq!(first["tenant_id"], "github|434149");
        assert_eq!(first["kid"], "0123456789abcdef0123456789abcdef");
        assert_eq!(first["prompt_tokens"], 1200);
        assert_eq!(first["completion_tokens"], 340);
        assert_eq!(first["cached_tokens"], 100);
        assert_eq!(first["charge_credits"], "1.23456");

        // Missing optional fields serialize as null, matching the wire shape
        assert!(json[1]["kid"].is_null());
    }

    #[test]
    fn models_serialize_to_openai_shape_for_json_output() {
        let models = fixture_models();
        let json = serde_json::to_value(&models).unwrap();
        assert_eq!(json[0]["id"], "llama-3.1-70b-instruct");
        assert_eq!(json[0]["object"], "model");
        assert_eq!(json[0]["owned_by"], "basilica");
        assert!(json[1]["owned_by"].is_null());
    }

    // ------------------------------------------------------------------
    // Error surfacing from each InferenceError variant
    // ------------------------------------------------------------------

    #[test]
    fn auth_error_points_at_login() {
        let err = map_inference_error(&InferenceError::Auth, GATEWAY_OVERRIDE_HINT);
        let text = err.to_string();
        assert!(text.contains("authentication failed (401)"), "{text}");
        assert!(text.contains("basilica login"), "{text}");
        assert!(text.contains(API_TOKEN_ENV), "{text}");
    }

    #[test]
    fn forbidden_error_points_at_fresh_key() {
        let err = map_inference_error(&InferenceError::Forbidden, GATEWAY_OVERRIDE_HINT);
        let text = err.to_string();
        assert!(text.contains("permission denied (403)"), "{text}");
        assert!(text.contains("basilica tokens create"), "{text}");
    }

    #[test]
    fn insufficient_credits_points_at_fund() {
        let err = map_inference_error(&InferenceError::InsufficientCredits, API_OVERRIDE_HINT);
        let text = err.to_string();
        assert!(text.contains("insufficient credits (402)"), "{text}");
        assert!(text.contains("basilica fund"), "{text}");
    }

    #[test]
    fn quota_exceeded_names_cap_and_retry_after() {
        let err = map_inference_error(
            &InferenceError::QuotaExceeded {
                cap: QuotaCap::Tpm,
                retry_after: Some(30),
            },
            GATEWAY_OVERRIDE_HINT,
        );
        let text = err.to_string();
        assert!(text.contains("Quota exceeded (429)"), "{text}");
        assert!(text.contains("the tpm cap tripped"), "{text}");
        assert!(text.contains("Retry in 30s"), "{text}");

        // Without a Retry-After header, no retry hint is printed
        let err = map_inference_error(
            &InferenceError::QuotaExceeded {
                cap: QuotaCap::Rpm,
                retry_after: None,
            },
            GATEWAY_OVERRIDE_HINT,
        );
        let text = err.to_string();
        assert!(text.contains("the rpm cap tripped"), "{text}");
        assert!(!text.contains("Retry in"), "{text}");
    }

    #[test]
    fn unavailable_is_retryable_language() {
        let err = map_inference_error(
            &InferenceError::Unavailable {
                retry_after: Some(5),
            },
            GATEWAY_OVERRIDE_HINT,
        );
        let text = err.to_string();
        assert!(
            text.contains("inference service unavailable (5xx)"),
            "{text}"
        );
        assert!(text.contains("Retry in 5s"), "{text}");
    }

    #[test]
    fn transport_error_carries_the_per_command_connectivity_hint() {
        // A real connect-refused transport error (loopback port 1; no server
        // is mocked or contacted beyond the refused connection).
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        runtime.block_on(async {
            let transport = reqwest::Client::builder()
                .timeout(std::time::Duration::from_millis(500))
                .build()
                .unwrap()
                .get("http://127.0.0.1:1/v1/models")
                .send()
                .await
                .expect_err("port 1 must refuse");

            let gateway =
                map_inference_error(&InferenceError::Transport(transport), GATEWAY_OVERRIDE_HINT);
            let text = gateway.to_string();
            assert!(text.contains("transport error"), "{text}");
            assert!(text.contains(INFERENCE_ENDPOINT_ENV), "{text}");

            let transport = reqwest::Client::new()
                .get("http://127.0.0.1:1/v1/inference/usage/summary")
                .send()
                .await
                .expect_err("port 1 must refuse");
            let api = map_inference_error(&InferenceError::Transport(transport), API_OVERRIDE_HINT);
            assert!(api.to_string().contains("api.base_url"), "{}", api);
        });
    }

    #[test]
    fn invalid_decode_and_model_not_found_pass_the_sdk_message_through() {
        let err = map_inference_error(
            &InferenceError::Invalid("`from` must be on or before `to`".to_string()),
            API_OVERRIDE_HINT,
        );
        assert_eq!(
            err.to_string(),
            "invalid request: `from` must be on or before `to`"
        );

        let err = map_inference_error(
            &InferenceError::Decode("expected rows".to_string()),
            API_OVERRIDE_HINT,
        );
        assert_eq!(err.to_string(), "failed to decode response: expected rows");

        let err = map_inference_error(
            &InferenceError::ModelNotFound {
                model: "no-such-model".to_string(),
            },
            GATEWAY_OVERRIDE_HINT,
        );
        assert_eq!(err.to_string(), "model not found: no-such-model");
    }
}
