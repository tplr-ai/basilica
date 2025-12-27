//! Event handling for the TUI

mod input;
mod tick;

use anyhow::Result;
use crossterm::event::{Event as CrosstermEvent, KeyEvent, MouseEvent};
use futures::StreamExt;
use std::time::Duration;
use tokio::sync::mpsc;

/// Application events
#[derive(Debug, Clone)]
pub enum Event {
    /// Tick event for periodic updates
    Tick,
    /// Keyboard event
    Key(KeyEvent),
    /// Mouse event
    Mouse(MouseEvent),
    /// Terminal resize event
    Resize(u16, u16),
}

/// Event handler that manages crossterm events
pub struct EventHandler {
    /// Receiver for events
    rx: mpsc::UnboundedReceiver<Event>,
    /// Handle to the event task (kept alive)
    _tx: mpsc::UnboundedSender<Event>,
}

impl EventHandler {
    /// Create a new event handler
    pub fn new(tick_rate_ms: u64) -> Self {
        let tick_rate = Duration::from_millis(tick_rate_ms);
        let (tx, rx) = mpsc::unbounded_channel();

        let tx_clone = tx.clone();
        tokio::spawn(async move {
            let mut reader = crossterm::event::EventStream::new();
            let mut tick_interval = tokio::time::interval(tick_rate);

            loop {
                let tick_delay = tick_interval.tick();
                let crossterm_event = reader.next();

                tokio::select! {
                    _ = tick_delay => {
                        if tx_clone.send(Event::Tick).is_err() {
                            break;
                        }
                    }
                    Some(Ok(event)) = crossterm_event => {
                        let event = match event {
                            CrosstermEvent::Key(key) => Some(Event::Key(key)),
                            CrosstermEvent::Mouse(mouse) => Some(Event::Mouse(mouse)),
                            CrosstermEvent::Resize(w, h) => Some(Event::Resize(w, h)),
                            _ => None,
                        };
                        if let Some(event) = event {
                            if tx_clone.send(event).is_err() {
                                break;
                            }
                        }
                    }
                }
            }
        });

        Self { rx, _tx: tx }
    }

    /// Get the next event
    pub async fn next(&mut self) -> Result<Event> {
        self.rx
            .recv()
            .await
            .ok_or_else(|| anyhow::anyhow!("Event channel closed"))
    }
}
