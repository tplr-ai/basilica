//! Command handlers for the Basilica CLI

pub mod auth;
pub mod balance;
pub mod deploy;
pub mod fund;
pub mod gpu_rental;
pub mod gpu_rental_helpers;
pub mod inference;
pub mod region_mapping;
pub mod skills;
pub mod ssh_keys;
#[cfg(debug_assertions)]
pub mod test_auth;
pub mod tokens;
pub mod train;
pub mod upgrade;
pub mod volumes;
