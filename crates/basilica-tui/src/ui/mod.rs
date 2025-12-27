//! UI module - rendering and terminal management

pub mod components;
pub mod screens;
pub mod theme;
pub mod widgets;

pub use theme::Theme;

use anyhow::Result;
use crossterm::{
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{backend::CrosstermBackend, Terminal};
use std::io::{self, Stdout};

/// Terminal UI wrapper
pub struct Tui {
    terminal: Terminal<CrosstermBackend<Stdout>>,
}

impl Tui {
    /// Create a new TUI
    pub fn new() -> Result<Self> {
        let backend = CrosstermBackend::new(io::stdout());
        let terminal = Terminal::new(backend)?;
        Ok(Self { terminal })
    }

    /// Enter the TUI (setup terminal)
    pub fn enter(&mut self) -> Result<()> {
        enable_raw_mode()?;
        execute!(io::stdout(), EnterAlternateScreen)?;

        // Enable mouse capture
        execute!(io::stdout(), crossterm::event::EnableMouseCapture)?;

        self.terminal.hide_cursor()?;
        self.terminal.clear()?;
        Ok(())
    }

    /// Exit the TUI (restore terminal)
    pub fn exit(&mut self) -> Result<()> {
        // Disable mouse capture
        execute!(io::stdout(), crossterm::event::DisableMouseCapture)?;

        disable_raw_mode()?;
        execute!(io::stdout(), LeaveAlternateScreen)?;
        self.terminal.show_cursor()?;
        Ok(())
    }

    /// Draw to the terminal
    pub fn draw<F>(&mut self, f: F) -> Result<()>
    where
        F: FnOnce(&mut ratatui::Frame),
    {
        self.terminal.draw(f)?;
        Ok(())
    }

    /// Suspend the TUI temporarily to run an external process
    ///
    /// Returns the terminal to normal mode, runs the provided closure,
    /// then restores the TUI. Use this for interactive SSH sessions, etc.
    pub fn suspend_and_run<F, R>(&mut self, f: F) -> Result<R>
    where
        F: FnOnce() -> R,
    {
        // Exit TUI mode
        self.exit()?;

        // Clear the screen and show a message
        print!("\x1B[2J\x1B[1;1H"); // Clear screen, move cursor to top-left
        println!("Suspended TUI. Running external command...\n");

        // Run the external process
        let result = f();

        // Wait for user to press Enter before resuming
        println!("\n\nPress Enter to return to TUI...");
        let _ = std::io::stdin().read_line(&mut String::new());

        // Re-enter TUI mode
        self.enter()?;

        Ok(result)
    }
}

impl Drop for Tui {
    fn drop(&mut self) {
        let _ = self.exit();
    }
}
