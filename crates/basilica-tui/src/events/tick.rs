//! Tick/timer utilities

use std::time::{Duration, Instant};

/// Timer for tracking periodic events
pub struct Timer {
    interval: Duration,
    last_tick: Instant,
}

impl Timer {
    /// Create a new timer with the given interval in seconds
    pub fn new(interval_secs: u64) -> Self {
        Self {
            interval: Duration::from_secs(interval_secs),
            last_tick: Instant::now(),
        }
    }

    /// Check if the timer has elapsed and reset if so
    pub fn check(&mut self) -> bool {
        if self.last_tick.elapsed() >= self.interval {
            self.last_tick = Instant::now();
            true
        } else {
            false
        }
    }

    /// Reset the timer
    pub fn reset(&mut self) {
        self.last_tick = Instant::now();
    }
}
