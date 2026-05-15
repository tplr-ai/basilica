//! Progress indicators and user feedback utilities

use crate::interactive::gate::{current, Interactivity};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Global progress manager for coordinating multiple progress indicators
pub struct ProgressManager {
    multi: MultiProgress,
    active_bars: Arc<Mutex<HashMap<String, ProgressBar>>>,
}

impl ProgressManager {
    /// Create a new progress manager
    pub fn new() -> Self {
        Self {
            multi: MultiProgress::new(),
            active_bars: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Start a spinner operation with a given name and message
    pub fn start_spinner(&self, name: &str, message: &str) -> ProgressBar {
        let spinner = match current() {
            Interactivity::NonInteractive => ProgressBar::hidden(),
            Interactivity::Interactive => {
                let s = ProgressBar::new_spinner();
                s.set_style(
                    ProgressStyle::default_spinner()
                        .template("{spinner:.cyan} {msg}")
                        .unwrap()
                        .tick_chars("⠁⠂⠄⡀⢀⠠⠐⠈ "),
                );
                s.enable_steady_tick(Duration::from_millis(120));
                s
            }
        };
        spinner.set_message(message.to_string());

        let pb = self.multi.add(spinner);

        // Store reference for potential cleanup
        if let Ok(mut bars) = self.active_bars.lock() {
            bars.insert(name.to_string(), pb.clone());
        }

        pb
    }

    /// Start a progress bar operation with known total
    pub fn start_progress(&self, name: &str, total: u64, message: &str) -> ProgressBar {
        let pb = ProgressBar::new(total);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} {msg} [{bar:40.cyan/blue}] {pos}/{len}")
                .unwrap()
                .progress_chars("=>-"),
        );
        pb.set_message(message.to_string());

        let progress_bar = self.multi.add(pb);

        // Store reference
        if let Ok(mut bars) = self.active_bars.lock() {
            bars.insert(name.to_string(), progress_bar.clone());
        }

        progress_bar
    }

    /// Complete an operation with success message
    pub fn complete_success(&self, pb: ProgressBar, message: &str) {
        pb.finish_with_message(format!("✓ {}", message));
    }

    /// Complete an operation with error message
    pub fn complete_error(&self, pb: ProgressBar, message: &str) {
        pb.finish_with_message(format!("✗ {}", message));
    }

    /// Clear and finish a progress bar
    pub fn clear(&self, pb: ProgressBar) {
        pb.finish_and_clear();
    }

    /// Clear all active progress bars (for cleanup on error)
    pub fn clear_all(&self) {
        if let Ok(mut bars) = self.active_bars.lock() {
            for (_, pb) in bars.drain() {
                pb.finish_and_clear();
            }
        }
    }
}

impl Default for ProgressManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Simple spinner for standalone operations
pub fn create_spinner(message: &str) -> ProgressBar {
    let spinner = match current() {
        Interactivity::NonInteractive => ProgressBar::hidden(),
        Interactivity::Interactive => {
            let s = ProgressBar::new_spinner();
            s.set_style(
                ProgressStyle::default_spinner()
                    .template("{spinner:.cyan} {msg}")
                    .unwrap()
                    .tick_chars("⠁⠂⠄⡀⢀⠠⠐⠈ "),
            );
            s.enable_steady_tick(Duration::from_millis(120));
            s
        }
    };
    spinner.set_message(message.to_string());
    spinner
}

/// Simple progress bar for standalone operations
pub fn create_progress_bar(total: u64, message: &str) -> ProgressBar {
    let pb = ProgressBar::new(total);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} {msg} [{bar:40.cyan/blue}] {pos}/{len}")
            .unwrap()
            .progress_chars("=>-"),
    );
    pb.set_message(message.to_string());
    pb
}

/// Finish spinner with success message
pub fn complete_spinner_success(spinner: ProgressBar, message: &str) {
    spinner.finish_with_message(format!("✓ {}", message));
}

/// Finish spinner with error and clear
pub fn complete_spinner_error(spinner: ProgressBar, _message: &str) {
    // Clear the spinner completely instead of leaving error message
    // The error will be displayed through the error handling system
    spinner.finish_and_clear();
}

/// Clear spinner completely without leaving any message
pub fn complete_spinner_and_clear(spinner: ProgressBar) {
    spinner.finish_and_clear();
}
