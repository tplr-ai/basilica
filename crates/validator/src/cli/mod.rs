pub mod args;
pub mod commands;
pub mod handlers;

pub use args::Args;
pub use commands::Command;
pub use handlers::CommandHandler;

use anyhow::Result;

pub struct Cli {
    handler: CommandHandler,
}

impl Cli {
    pub fn new() -> Self {
        Self {
            handler: CommandHandler::new(),
        }
    }

    pub async fn run(&self, args: Args) -> Result<()> {
        self.handler
            .execute_with_context(args.command, args.local_test)
            .await
    }
}

impl Default for Cli {
    fn default() -> Self {
        Self::new()
    }
}
