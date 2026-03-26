pub mod bidding;
pub mod collateral;
pub mod emission;
pub mod pricing;

#[cfg(test)]
mod emission_tests;

#[cfg(test)]
mod integration_tests;

// Re-export all the main config structs and functions
mod main_config;
pub use main_config::*;
