//! Help text and documentation for CLI commands
//!
//! This module provides consistent help text for the dual identifier system
//! and executor management commands.

use std::fmt::Write;

/// Help text for the dual identifier system
pub const IDENTIFIER_HELP: &str = r#"
EXECUTOR IDENTIFIERS

Basilica uses a dual identifier system for executors:

1. UUID (Universally Unique Identifier)
   - Format: 550e8400-e29b-41d4-a716-446655440000
   - Used internally for data integrity
   - Guaranteed globally unique
   - Ideal for scripts and automation

2. HUID (Human-Unique Identifier)  
   - Format: adjective-noun-hex (e.g., swift-falcon-a3f2)
   - User-friendly and memorable
   - Used in CLI output by default
   - Easy to communicate verbally

USAGE

You can use either identifier in commands:
- Full UUID: basilica executor show 550e8400-e29b-41d4-a716-446655440000
- Full HUID: basilica executor show swift-falcon-a3f2
- UUID prefix: basilica executor show 550e8400 (min 3 chars)
- HUID prefix: basilica executor show swift-fal (min 3 chars)

PREFIX MATCHING

Minimum 3 characters required for prefix matching.
If multiple executors match, you'll see all matches with suggestions.

EXAMPLES

List all executors (shows HUID by default):
  $ basilica executor list
  swift-falcon-a3f2    HEALTHY     localhost:50051
  bright-phoenix-7b9e  HEALTHY     localhost:50052

List with full details (shows both UUID and HUID):
  $ basilica executor list --verbose
  550e8400-e29b-41d4-a716-446655440000  swift-falcon-a3f2    HEALTHY     localhost:50051
  660f9500-f39c-52e5-b827-557766550001  bright-phoenix-7b9e  HEALTHY     localhost:50052

Search by prefix:
  $ basilica executor show swift
  Error: Multiple executors match 'swift':
    1. swift-falcon-a3f2 (550e8400)
    2. swift-eagle-b7d1 (660f9500)
  Please use a longer prefix or the full identifier.
"#;

/// Generate help text for a specific command
pub fn generate_command_help(command: &str) -> String {
    match command {
        "list" => LIST_COMMAND_HELP.to_string(),
        "show" => SHOW_COMMAND_HELP.to_string(),
        "assign" => ASSIGN_COMMAND_HELP.to_string(),
        _ => format!("Help for '{command}' command"),
    }
}

const LIST_COMMAND_HELP: &str = r#"
List all executors with optional filtering

USAGE:
    basilica executor list [OPTIONS]

OPTIONS:
    -f, --filter <FILTER>    Filter by UUID or HUID prefix (min 3 chars)
    -v, --verbose            Show UUID + HUID (default: HUID only)
    -o, --output <FORMAT>    Output format [default: table]
                            Possible values: table, json, compact, verbose

EXAMPLES:
    List all executors:
        $ basilica executor list

    Filter by prefix:
        $ basilica executor list --filter swift

    Show full details:
        $ basilica executor list --verbose

    JSON output:
        $ basilica executor list --output json
"#;

const SHOW_COMMAND_HELP: &str = r#"
Show detailed information about a specific executor

USAGE:
    basilica executor show <EXECUTOR_ID> [OPTIONS]

ARGS:
    <EXECUTOR_ID>    UUID or HUID (full or prefix with min 3 chars)

OPTIONS:
    -o, --output <FORMAT>    Output format [default: text]
                            Possible values: text, json

EXAMPLES:
    Show by full HUID:
        $ basilica executor show swift-falcon-a3f2

    Show by UUID prefix:
        $ basilica executor show 550e8400

    JSON output:
        $ basilica executor show swift-falcon-a3f2 --output json
"#;

const ASSIGN_COMMAND_HELP: &str = r#"
Assign an executor to a validator

USAGE:
    basilica executor assign <EXECUTOR_ID> <VALIDATOR>

ARGS:
    <EXECUTOR_ID>    UUID or HUID (full or prefix with min 3 chars)
    <VALIDATOR>      Validator address

EXAMPLES:
    Assign by HUID:
        $ basilica executor assign swift-falcon-a3f2 5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY

    Assign by UUID:
        $ basilica executor assign 550e8400-e29b-41d4-a716-446655440000 5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY
"#;

/// Format error message with helpful context
pub fn format_error_with_help(error: &str, command: Option<&str>) -> String {
    let mut output = format!("Error: {error}\n");

    if let Some(cmd) = command {
        writeln!(
            &mut output,
            "\nFor help, run: basilica executor {cmd} --help"
        )
        .ok();
    }

    output
}

/// Format disambiguation error with examples
pub fn format_disambiguation_help(query: &str, matches: &[String]) -> String {
    let mut output = format!("Multiple executors match '{query}':\n");

    for (i, id) in matches.iter().enumerate() {
        writeln!(&mut output, "  {}. {}", i + 1, id).ok();
    }

    writeln!(&mut output, "\nTo disambiguate, use one of:").ok();

    writeln!(&mut output, "  - A longer prefix (e.g., '{query}xxx')").ok();
    writeln!(&mut output, "  - The full identifier").ok();
    writeln!(&mut output, "  - The UUID if known").ok();

    output
}

/// Generate examples for the README or documentation
pub fn generate_usage_examples() -> String {
    r#"
# Executor Management with UUID/HUID

## Basic Usage

List all executors (HUID display by default):
```bash
$ basilica executor list
swift-falcon-a3f2    HEALTHY     localhost:50051
bright-phoenix-7b9e  HEALTHY     localhost:50052
quiet-thunder-2d4c   UNHEALTHY   localhost:50053
```

Show executor details:
```bash
$ basilica executor show swift-falcon-a3f2
Executor Details:
  UUID: 550e8400-e29b-41d4-a716-446655440000
  HUID: swift-falcon-a3f2
  Status: HEALTHY
  Address: localhost:50051
  Created: 2024-01-01T00:00:00Z
```

## Prefix Matching

Use minimum 3 characters for prefix matching:
```bash
$ basilica executor show swi
Executor Details:
  UUID: 550e8400-e29b-41d4-a716-446655440000
  HUID: swift-falcon-a3f2
  ...
```

## Handling Ambiguous Matches

When multiple executors match:
```bash
$ basilica executor show swift
Error: Multiple executors match 'swift':
  1. swift-falcon-a3f2 (550e8400)
  2. swift-eagle-b7d1 (660f9500)
Please use a longer prefix or the full identifier.
```

## Working with UUIDs

For scripts and automation:
```bash
# Using full UUID
$ basilica executor assign 550e8400-e29b-41d4-a716-446655440000 $VALIDATOR_ADDRESS

# Using UUID prefix
$ basilica executor show 550e8400
```

## Verbose Output

Show both UUID and HUID:
```bash
$ basilica executor list --verbose
550e8400-e29b-41d4-a716-446655440000  swift-falcon-a3f2    HEALTHY     localhost:50051
660f9500-f39c-52e5-b827-557766550001  bright-phoenix-7b9e  HEALTHY     localhost:50052
```

## JSON Output

For programmatic access:
```bash
$ basilica executor show swift-falcon-a3f2 --output json
{
  "uuid": "550e8400-e29b-41d4-a716-446655440000",
  "huid": "swift-falcon-a3f2",
  "status": "HEALTHY",
  "address": "localhost:50051",
  "created_at": 1704067200
}
```
"#
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_command_help() {
        let help = generate_command_help("list");
        assert!(help.contains("List all executors"));
        assert!(help.contains("--filter"));
        assert!(help.contains("--verbose"));
    }

    #[test]
    fn test_format_error_with_help() {
        let error = format_error_with_help("Executor not found", Some("show"));
        assert!(error.contains("Error: Executor not found"));
        assert!(error.contains("basilica executor show --help"));
    }

    #[test]
    fn test_format_disambiguation_help() {
        let matches = vec![
            "swift-falcon-a3f2".to_string(),
            "swift-eagle-b7d1".to_string(),
        ];
        let help = format_disambiguation_help("swift", &matches);

        assert!(help.contains("Multiple executors match 'swift'"));
        assert!(help.contains("swift-falcon-a3f2"));
        assert!(help.contains("swift-eagle-b7d1"));
        assert!(help.contains("To disambiguate"));
    }
}
