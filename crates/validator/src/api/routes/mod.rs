//! Route handlers for the validator API

pub mod capacity;
pub mod health;
pub mod logs;
pub mod miners;
pub mod rentals;

pub use capacity::*;
pub use health::*;
pub use logs::*;
pub use miners::*;
pub use rentals::*;
