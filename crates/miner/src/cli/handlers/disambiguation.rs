//! Disambiguation helpers for executor identity matching
//!
//! This module provides utilities for handling ambiguous executor matches
//! when searching by UUID or HUID prefix.

use anyhow::{anyhow, Result};
use common::executor_identity::ExecutorIdentity;
use std::fmt::Write;

/// Result of an identity search
#[derive(Debug)]
pub enum IdentitySearchResult<T> {
    /// Exactly one match found
    Unique(T),
    /// No matches found
    NotFound,
    /// Multiple matches found
    Ambiguous(Vec<T>),
}

/// Disambiguation options for displaying multiple matches
#[derive(Debug, Clone)]
pub struct DisambiguationOptions {
    /// Maximum number of matches to show
    pub max_display: usize,
    /// Whether to show short UUIDs for disambiguation
    pub show_short_uuid: bool,
    /// Whether to suggest the full identifier
    pub suggest_full_id: bool,
}

impl Default for DisambiguationOptions {
    fn default() -> Self {
        Self {
            max_display: 10,
            show_short_uuid: true,
            suggest_full_id: true,
        }
    }
}

/// Search for executors by identifier with disambiguation support
pub fn search_by_identifier<T, I>(
    query: &str,
    items: I,
    matcher: impl Fn(&T, &str) -> bool,
) -> IdentitySearchResult<T>
where
    I: IntoIterator<Item = T>,
{
    let matches: Vec<T> = items
        .into_iter()
        .filter(|item| matcher(item, query))
        .collect();

    match matches.len() {
        0 => IdentitySearchResult::NotFound,
        1 => IdentitySearchResult::Unique(matches.into_iter().next().unwrap()),
        _ => IdentitySearchResult::Ambiguous(matches),
    }
}

/// Format an ambiguous match error with helpful disambiguation info
pub fn format_ambiguous_error<T>(
    query: &str,
    matches: &[T],
    display_fn: impl Fn(&T) -> String,
    options: &DisambiguationOptions,
) -> String {
    let mut error_msg = format!("Multiple executors match '{query}'. Matches found:\n");

    // Show up to max_display matches
    let display_count = matches.len().min(options.max_display);
    for (i, item) in matches.iter().take(display_count).enumerate() {
        writeln!(&mut error_msg, "  {}. {}", i + 1, display_fn(item)).ok();
    }

    // If there are more matches than displayed
    if matches.len() > options.max_display {
        writeln!(
            &mut error_msg,
            "  ... and {} more",
            matches.len() - options.max_display
        )
        .ok();
    }

    // Add suggestions
    if options.suggest_full_id {
        write!(
            &mut error_msg,
            "\nPlease use a longer prefix or the full identifier to disambiguate."
        )
        .ok();
    }

    error_msg
}

/// Interactive disambiguation prompt
pub async fn prompt_for_disambiguation<T>(
    query: &str,
    matches: Vec<T>,
    display_fn: impl Fn(&T) -> String,
) -> Result<T> {
    use std::io::{self, Write as IoWrite};

    println!("Multiple executors match '{query}'. Please select:");

    for (i, item) in matches.iter().enumerate() {
        println!("  {}. {}", i + 1, display_fn(item));
    }

    print!("\nEnter selection (1-{}): ", matches.len());
    io::stdout().flush()?;

    let mut input = String::new();
    io::stdin().read_line(&mut input)?;

    let selection: usize = input
        .trim()
        .parse()
        .map_err(|_| anyhow!("Invalid selection"))?;

    if selection < 1 || selection > matches.len() {
        return Err(anyhow!("Selection out of range"));
    }

    matches
        .into_iter()
        .nth(selection - 1)
        .ok_or_else(|| anyhow!("Invalid selection"))
}

/// Validate identifier query length
pub fn validate_query_length(query: &str, min_length: usize) -> Result<()> {
    // Check if it's a valid UUID (always allowed regardless of length)
    if uuid::Uuid::parse_str(query).is_ok() {
        return Ok(());
    }

    // Otherwise, enforce minimum length for prefix matching
    if query.len() < min_length {
        return Err(anyhow!(
            "Identifier must be a valid UUID or at least {} characters for prefix matching",
            min_length
        ));
    }

    Ok(())
}

/// Format executor identity for disambiguation display
pub fn format_executor_for_disambiguation<T: ExecutorIdentity>(
    executor: &T,
    show_short_uuid: bool,
) -> String {
    if show_short_uuid {
        format!("{} ({})", executor.huid(), executor.short_uuid())
    } else {
        executor.huid().to_string()
    }
}

/// Suggestion builder for command help
pub struct CommandSuggestion {
    command: String,
    examples: Vec<String>,
}

impl CommandSuggestion {
    pub fn new(command: &str) -> Self {
        Self {
            command: command.to_string(),
            examples: Vec::new(),
        }
    }

    pub fn add_example(mut self, example: &str) -> Self {
        self.examples.push(example.to_string());
        self
    }

    pub fn format(&self) -> String {
        let mut output = format!("Usage: {}\n", self.command);

        if !self.examples.is_empty() {
            output.push_str("\nExamples:\n");
            for example in &self.examples {
                writeln!(&mut output, "  {example}").ok();
            }
        }

        output
    }
}

/// Helper to create disambiguation suggestions
pub fn create_disambiguation_suggestion(base_command: &str) -> CommandSuggestion {
    CommandSuggestion::new(base_command)
        .add_example(&format!("{base_command} swift-falcon-a3f2"))
        .add_example(&format!("{base_command} swift-fal"))
        .add_example(&format!(
            "{base_command} 550e8400-e29b-41d4-a716-446655440000"
        ))
        .add_example(&format!("{base_command} 550e8400"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::SystemTime;
    use uuid::Uuid;

    #[derive(Clone)]
    struct MockExecutor {
        uuid: Uuid,
        huid: String,
    }

    impl ExecutorIdentity for MockExecutor {
        fn uuid(&self) -> &Uuid {
            &self.uuid
        }

        fn huid(&self) -> &str {
            &self.huid
        }

        fn created_at(&self) -> SystemTime {
            SystemTime::now()
        }

        fn matches(&self, query: &str) -> bool {
            self.huid.starts_with(query) || self.uuid.to_string().starts_with(query)
        }

        fn full_display(&self) -> String {
            format!("{} ({})", self.huid, self.uuid)
        }

        fn short_uuid(&self) -> String {
            self.uuid.to_string()[..8].to_string()
        }
    }

    #[test]
    fn test_search_by_identifier() {
        let executors = vec![
            MockExecutor {
                uuid: Uuid::new_v4(),
                huid: "swift-falcon-a3f2".to_string(),
            },
            MockExecutor {
                uuid: Uuid::new_v4(),
                huid: "swift-eagle-b4c3".to_string(),
            },
            MockExecutor {
                uuid: Uuid::new_v4(),
                huid: "brave-lion-d5e4".to_string(),
            },
        ];

        // Test unique match
        let result = search_by_identifier("brave", executors.clone(), |e, q| e.matches(q));
        assert!(matches!(result, IdentitySearchResult::Unique(_)));

        // Test ambiguous match
        let result = search_by_identifier("swift", executors.clone(), |e, q| e.matches(q));
        assert!(matches!(result, IdentitySearchResult::Ambiguous(ref v) if v.len() == 2));

        // Test not found
        let result = search_by_identifier("missing", executors.clone(), |e, q| e.matches(q));
        assert!(matches!(result, IdentitySearchResult::NotFound));
    }

    #[test]
    fn test_format_ambiguous_error() {
        let executors = vec![
            MockExecutor {
                uuid: Uuid::new_v4(),
                huid: "swift-falcon-a3f2".to_string(),
            },
            MockExecutor {
                uuid: Uuid::new_v4(),
                huid: "swift-eagle-b4c3".to_string(),
            },
        ];

        let options = DisambiguationOptions::default();
        let error = format_ambiguous_error(
            "swift",
            &executors,
            |e| format_executor_for_disambiguation(e, true),
            &options,
        );

        assert!(error.contains("Multiple executors match 'swift'"));
        assert!(error.contains("swift-falcon-a3f2"));
        assert!(error.contains("swift-eagle-b4c3"));
        assert!(error.contains("Please use a longer prefix"));
    }

    #[test]
    fn test_validate_query_length() {
        // Valid UUID should always pass
        let uuid = Uuid::new_v4().to_string();
        assert!(validate_query_length(&uuid, 3).is_ok());

        // Short non-UUID should fail
        assert!(validate_query_length("ab", 3).is_err());

        // Long enough non-UUID should pass
        assert!(validate_query_length("abc", 3).is_ok());
        assert!(validate_query_length("swift", 3).is_ok());
    }

    #[test]
    fn test_command_suggestion() {
        let suggestion = create_disambiguation_suggestion("basilica executor show");
        let formatted = suggestion.format();

        assert!(formatted.contains("Usage: basilica executor show"));
        assert!(formatted.contains("Examples:"));
        assert!(formatted.contains("swift-falcon-a3f2"));
        assert!(formatted.contains("550e8400"));
    }
}
