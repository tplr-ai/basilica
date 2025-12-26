//! Screen implementations

pub mod billing;
pub mod dashboard;
pub mod deployments;
pub mod marketplace;
pub mod miner;
pub mod rentals;

/// Common trait for screens
pub trait Screen {
    fn render(&self, frame: &mut ratatui::Frame, app: &crate::app::App);
}

