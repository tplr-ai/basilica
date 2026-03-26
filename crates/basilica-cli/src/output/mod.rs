//! Output formatting utilities

pub mod banner;
pub mod table_output;

use color_eyre::eyre::{eyre, Result};
use console::style;
use rust_decimal::Decimal;
use serde::Serialize;

/// Output data as JSON
pub fn json_output<T: Serialize>(data: &T) -> Result<()> {
    let json = serde_json::to_string_pretty(data)
        .map_err(|e| eyre!("Failed to serialize to JSON: {}", e))?;
    println!("{json}");
    Ok(())
}

/// Print a success message with green checkmark
pub fn print_success(message: &str) {
    println!("{} {}", style("✓").green().bold(), message);
}

/// Print an error message with red X
pub fn print_error(message: &str) {
    eprintln!("{} {}", style("✗").red().bold(), style(message).red());
}

/// Print an informational message with blue info icon
pub fn print_info(message: &str) {
    println!("{} {}", style("ℹ").blue(), message);
}

/// Print a warning message with yellow warning icon
pub fn print_warning(message: &str) {
    println!("{} {}", style("⚠").yellow(), style(message).yellow());
}

/// Print a link/URL with label
pub fn print_link(label: &str, url: &str) {
    println!("{} {}: {}", style("→").cyan(), label, style(url).dim());
}

/// Print a security/auth related message  
pub fn print_auth(message: &str) {
    println!("{} {}", style("🔐").cyan(), message);
}

/// Compress a path to use tilde notation for home directory
pub fn compress_path(path: &std::path::Path) -> String {
    if let Ok(home_dir) = std::env::var("HOME") {
        let home_path = std::path::Path::new(&home_dir);
        if let Ok(relative) = path.strip_prefix(home_path) {
            return format!("~/{}", relative.display());
        }
    }
    path.display().to_string()
}

/// Format credit value with 2 decimal places
///
/// Parses a string credit value as Decimal and formats it with exactly 2 decimal places.
/// Falls back to the original string if parsing fails.
///
/// # Examples
/// ```
/// use basilica_cli::output::format_credits;
///
/// let formatted = format_credits("1234.56789");
/// assert_eq!(formatted, "1234.57");
/// ```
pub fn format_credits(value: &str) -> String {
    value
        .parse::<Decimal>()
        .map(|d| format!("{:.2}", d.round_dp(2)))
        .unwrap_or_else(|_| value.to_string())
}

/// Format a USD amount string for display.
///
/// - Uses 2 decimal places for normal values
/// - Uses "<$0.01" for tiny positive values to avoid rounding to "$0.00"
/// - Preserves the original string if parsing fails
pub fn format_usd(value: &str) -> String {
    let amount = match value.parse::<Decimal>() {
        Ok(amount) => amount,
        Err(_) => return value.to_string(),
    };

    if amount.is_zero() {
        return "$0.00".to_string();
    }

    let threshold = Decimal::new(1, 2); // 0.01
    let is_negative = amount.is_sign_negative();
    let abs_amount = if is_negative { -amount } else { amount };

    if abs_amount < threshold {
        return if is_negative {
            "-$0.01".to_string()
        } else {
            "<$0.01".to_string()
        };
    }

    if is_negative {
        format!("-${:.2}", abs_amount)
    } else {
        format!("${:.2}", abs_amount)
    }
}
