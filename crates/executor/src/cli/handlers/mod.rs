//! CLI command handlers

use crate::config::ExecutorConfig;
use crate::ExecutorState;
use anyhow::Result;

pub mod container;
pub mod network;
pub mod resource;
pub mod service;
pub mod system;
pub mod validator;

pub struct HandlerUtils;

impl HandlerUtils {
    pub fn load_config(config_path: &str) -> Result<ExecutorConfig> {
        if std::path::Path::new(config_path).exists() {
            ExecutorConfig::load_from_file(std::path::Path::new(config_path))
        } else {
            ExecutorConfig::load()
        }
    }

    pub async fn init_executor_state(config: ExecutorConfig) -> Result<ExecutorState> {
        ExecutorState::new(config).await
    }

    pub fn format_json<T: serde::Serialize>(data: &T) -> Result<String> {
        Ok(serde_json::to_string_pretty(data)?)
    }

    pub fn format_table<T: std::fmt::Display>(headers: &[&str], rows: &[Vec<T>]) -> String {
        let mut output = String::new();

        output.push_str(&format!("{:<20}", headers.join(" | ")));
        output.push('\n');
        output.push_str(&"-".repeat(headers.len() * 20));
        output.push('\n');

        for row in rows {
            let row_str: Vec<String> = row.iter().map(|cell| format!("{cell:<20}")).collect();
            output.push_str(&row_str.join(" | "));
            output.push('\n');
        }

        output
    }

    pub fn print_success(message: &str) {
        println!("[SUCCESS] {message}");
    }

    pub fn print_error(message: &str) {
        eprintln!("[ERROR] {message}");
    }

    pub fn print_info(message: &str) {
        println!("[INFO] {message}");
    }

    pub fn print_warning(message: &str) {
        println!("[WARNING] {message}");
    }
}
