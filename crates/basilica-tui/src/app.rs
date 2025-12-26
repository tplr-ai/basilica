//! Core application state machine

use anyhow::Result;
use basilica_sdk::{BasilicaClient, ClientBuilder};
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, error, info};

use crate::config::TuiConfig;
use crate::data::streams::LogStreamManager;
use crate::data::{MinerData, UserData};
use crate::events::{Event, EventHandler};
use crate::ui::{Theme, Tui};

/// Application mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AppMode {
    User,
    Miner,
}

/// Active screen in user mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum UserScreen {
    #[default]
    Dashboard,
    Rentals,
    Marketplace,
    Deployments,
    Billing,
}

impl UserScreen {
    pub fn all() -> &'static [UserScreen] {
        &[
            UserScreen::Dashboard,
            UserScreen::Rentals,
            UserScreen::Marketplace,
            UserScreen::Deployments,
            UserScreen::Billing,
        ]
    }

    pub fn label(&self) -> &'static str {
        match self {
            UserScreen::Dashboard => "Dashboard",
            UserScreen::Rentals => "Rentals",
            UserScreen::Marketplace => "Marketplace",
            UserScreen::Deployments => "Deploy",
            UserScreen::Billing => "Billing",
        }
    }

    pub fn index(&self) -> usize {
        match self {
            UserScreen::Dashboard => 0,
            UserScreen::Rentals => 1,
            UserScreen::Marketplace => 2,
            UserScreen::Deployments => 3,
            UserScreen::Billing => 4,
        }
    }

    pub fn from_index(index: usize) -> Self {
        match index {
            0 => UserScreen::Dashboard,
            1 => UserScreen::Rentals,
            2 => UserScreen::Marketplace,
            3 => UserScreen::Deployments,
            4 => UserScreen::Billing,
            _ => UserScreen::Dashboard,
        }
    }
}

/// Active screen in miner mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MinerScreen {
    #[default]
    Fleet,
    Validators,
    Nodes,
    Earnings,
    Logs,
}

impl MinerScreen {
    pub fn all() -> &'static [MinerScreen] {
        &[
            MinerScreen::Fleet,
            MinerScreen::Validators,
            MinerScreen::Nodes,
            MinerScreen::Earnings,
            MinerScreen::Logs,
        ]
    }

    pub fn label(&self) -> &'static str {
        match self {
            MinerScreen::Fleet => "Fleet",
            MinerScreen::Validators => "Validators",
            MinerScreen::Nodes => "Nodes",
            MinerScreen::Earnings => "Earnings",
            MinerScreen::Logs => "Logs",
        }
    }

    pub fn index(&self) -> usize {
        match self {
            MinerScreen::Fleet => 0,
            MinerScreen::Validators => 1,
            MinerScreen::Nodes => 2,
            MinerScreen::Earnings => 3,
            MinerScreen::Logs => 4,
        }
    }

    pub fn from_index(index: usize) -> Self {
        match index {
            0 => MinerScreen::Fleet,
            1 => MinerScreen::Validators,
            2 => MinerScreen::Nodes,
            3 => MinerScreen::Earnings,
            4 => MinerScreen::Logs,
            _ => MinerScreen::Fleet,
        }
    }
}

/// Notification message
#[derive(Debug, Clone)]
pub struct Notification {
    pub message: String,
    pub level: NotificationLevel,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NotificationLevel {
    Info,
    Success,
    Warning,
    Error,
}

/// Main application state
pub struct App {
    /// Application running state
    pub running: bool,

    /// Current mode (user or miner)
    pub mode: AppMode,

    /// Active user screen
    pub user_screen: UserScreen,

    /// Active miner screen
    pub miner_screen: MinerScreen,

    /// Whether help overlay is shown
    pub show_help: bool,

    /// Configuration
    pub config: TuiConfig,

    /// Theme
    pub theme: Theme,

    /// API client for data fetching
    client: Option<Arc<BasilicaClient>>,

    /// User mode data
    pub user_data: Arc<RwLock<UserData>>,

    /// Miner mode data
    pub miner_data: Arc<RwLock<MinerData>>,

    /// Notifications queue
    pub notifications: Vec<Notification>,

    /// Event handler
    event_handler: EventHandler,

    /// Terminal UI
    tui: Tui,

    /// Selected item index (for lists)
    pub selected_index: usize,

    /// Screen-specific state for screens
    pub screens: ScreenStates,

    /// Log stream manager for real-time logs
    #[allow(dead_code)]
    log_streams: LogStreamManager,

    /// Last data refresh time
    last_refresh: std::time::Instant,

    /// Tick count for periodic operations
    tick_count: u64,
}

/// Screen-specific state storage
#[derive(Debug, Default, Clone)]
#[allow(dead_code)]
pub struct ScreenStates {
    pub rentals: RentalsScreenState,
    pub marketplace: MarketplaceScreenState,
    pub deployments: DeploymentsScreenState,
    pub fleet: FleetScreenState,
}

#[derive(Debug, Default, Clone)]
#[allow(dead_code)]
pub struct RentalsScreenState {
    pub selected: usize,
    pub show_logs: bool,
    pub log_follow: bool,
}

#[derive(Debug, Default, Clone)]
#[allow(dead_code)]
pub struct MarketplaceScreenState {
    pub selected: usize,
    pub filter_gpu_type: Option<String>,
    pub sort_by_price: bool,
}

#[derive(Debug, Default, Clone)]
#[allow(dead_code)]
pub struct DeploymentsScreenState {
    pub selected: usize,
    pub show_logs: bool,
}

#[derive(Debug, Default, Clone)]
#[allow(dead_code)]
pub struct FleetScreenState {
    pub selected_node: usize,
    pub show_details: bool,
}

impl App {
    /// Create a new application
    pub async fn new(config: TuiConfig, miner_mode: bool, tick_rate: u64) -> Result<Self> {
        let theme = Theme::dark(); // TODO: Respect config.theme preference

        let event_handler = EventHandler::new(tick_rate);
        let tui = Tui::new()?;

        let user_data = Arc::new(RwLock::new(UserData::default()));
        let miner_data = Arc::new(RwLock::new(MinerData::default()));

        let log_streams = LogStreamManager::new(config.api_url.clone());

        // Try to create API client with file-based auth
        let client = match Self::create_client(&config.api_url).await {
            Ok(c) => {
                info!("API client initialized successfully");
                Some(Arc::new(c))
            }
            Err(e) => {
                error!(
                    "Failed to initialize API client: {}. Running in offline mode.",
                    e
                );
                None
            }
        };

        Ok(Self {
            running: true,
            mode: if miner_mode {
                AppMode::Miner
            } else {
                AppMode::User
            },
            user_screen: UserScreen::default(),
            miner_screen: MinerScreen::default(),
            show_help: false,
            config,
            theme,
            client,
            user_data,
            miner_data,
            notifications: Vec::new(),
            event_handler,
            tui,
            selected_index: 0,
            screens: ScreenStates::default(),
            log_streams,
            last_refresh: std::time::Instant::now(),
            tick_count: 0,
        })
    }

    /// Create API client with authentication
    async fn create_client(api_url: &str) -> Result<BasilicaClient> {
        // Try to use file-based auth (reads from ~/.local/share/basilica/)
        let client = ClientBuilder::default()
            .base_url(api_url)
            .with_file_auth()
            .build()?;

        // Verify connection with health check
        client.health_check().await?;

        Ok(client)
    }

    /// Run the main application loop
    pub async fn run(&mut self) -> Result<()> {
        self.tui.enter()?;

        // Initial data fetch
        self.refresh_data().await;

        while self.running {
            // Render UI - extract what we need before the mutable borrow
            let mode = self.mode;
            let user_screen = self.user_screen;
            let miner_screen = self.miner_screen;
            let show_help = self.show_help;
            let theme = self.theme.clone();
            let selected_index = self.selected_index;
            let screens = self.screens.clone();
            let notifications = self.notifications.clone();
            let connected = self.client.is_some();

            // Clone data for rendering (quick read lock)
            let user_data = self.user_data.read().await.clone();
            let miner_data = self.miner_data.read().await.clone();

            self.tui.draw(|frame| {
                let ctx = RenderContext {
                    mode,
                    user_screen,
                    miner_screen,
                    show_help,
                    theme: &theme,
                    selected_index,
                    screens: &screens,
                    notifications: &notifications,
                    user_data: &user_data,
                    miner_data: &miner_data,
                    connected,
                };
                render_app(frame, &ctx);
            })?;

            // Handle events
            match self.event_handler.next().await? {
                Event::Tick => {
                    self.on_tick().await;
                }
                Event::Key(key) => {
                    self.handle_key(key).await;
                }
                Event::Mouse(mouse) => {
                    self.handle_mouse(mouse);
                }
                Event::Resize(width, height) => {
                    self.handle_resize(width, height);
                }
            }
        }

        self.tui.exit()?;
        Ok(())
    }

    /// Handle keyboard input
    async fn handle_key(&mut self, key: KeyEvent) {
        // Global keybindings
        match (key.modifiers, key.code) {
            // Quit
            (KeyModifiers::CONTROL, KeyCode::Char('c')) | (_, KeyCode::Char('q')) => {
                if !self.show_help {
                    self.running = false;
                } else {
                    self.show_help = false;
                }
            }
            // Toggle help
            (_, KeyCode::Char('?')) => {
                self.show_help = !self.show_help;
            }
            // Switch mode
            (_, KeyCode::Char('m')) if !self.show_help => {
                self.toggle_mode();
            }
            // Tab navigation
            (_, KeyCode::Tab) if !self.show_help => {
                self.next_screen();
            }
            (KeyModifiers::SHIFT, KeyCode::BackTab) if !self.show_help => {
                self.prev_screen();
            }
            // Number keys for direct screen access
            (_, KeyCode::Char(c)) if !self.show_help && c.is_ascii_digit() => {
                if let Some(idx) = c.to_digit(10) {
                    if (1..=5).contains(&idx) {
                        self.goto_screen((idx - 1) as usize);
                    }
                }
            }
            // List navigation
            (_, KeyCode::Char('j')) | (_, KeyCode::Down) if !self.show_help => {
                self.select_next();
            }
            (_, KeyCode::Char('k')) | (_, KeyCode::Up) if !self.show_help => {
                self.select_prev();
            }
            // Refresh
            (_, KeyCode::Char('r')) if !self.show_help => {
                self.refresh_data().await;
            }
            // Screen-specific keys (handled by individual screens)
            _ if !self.show_help => {
                self.handle_screen_key(key).await;
            }
            _ => {}
        }
    }

    /// Handle screen-specific key events
    async fn handle_screen_key(&mut self, key: KeyEvent) {
        match self.mode {
            AppMode::User => match self.user_screen {
                UserScreen::Dashboard => {
                    // 'u' to go to marketplace for new rental
                    if key.code == KeyCode::Char('u') {
                        self.user_screen = UserScreen::Marketplace;
                    }
                }
                UserScreen::Rentals => {
                    match key.code {
                        KeyCode::Enter | KeyCode::Char('l') => {
                            // Toggle logs view
                            self.screens.rentals.show_logs = !self.screens.rentals.show_logs;
                        }
                        KeyCode::Char('s') => {
                            // SSH into selected rental
                            self.ssh_into_rental().await;
                        }
                        KeyCode::Char('d') => {
                            // Stop rental
                            self.stop_rental().await;
                        }
                        _ => {}
                    }
                }
                UserScreen::Marketplace => {
                    if key.code == KeyCode::Enter {
                        // Start rental
                        self.start_rental().await;
                    }
                }
                UserScreen::Deployments => {
                    match key.code {
                        KeyCode::Enter | KeyCode::Char('l') => {
                            self.screens.deployments.show_logs =
                                !self.screens.deployments.show_logs;
                        }
                        KeyCode::Char('d') => {
                            // Delete deployment
                            self.delete_deployment().await;
                        }
                        _ => {}
                    }
                }
                UserScreen::Billing => {
                    // Billing screen is mostly read-only
                }
            },
            AppMode::Miner => match self.miner_screen {
                MinerScreen::Fleet => {
                    if key.code == KeyCode::Enter {
                        self.screens.fleet.show_details = !self.screens.fleet.show_details;
                    }
                }
                MinerScreen::Nodes => {
                    match key.code {
                        KeyCode::Char('a') => {
                            // Add node (would open input dialog)
                        }
                        KeyCode::Char('d') => {
                            // Remove selected node
                        }
                        _ => {}
                    }
                }
                _ => {}
            },
        }
    }

    /// Handle mouse events
    fn handle_mouse(&mut self, mouse: crossterm::event::MouseEvent) {
        use crossterm::event::{MouseButton, MouseEventKind};

        match mouse.kind {
            MouseEventKind::Down(MouseButton::Left) => {
                // TODO: Implement click handling for interactive elements
                // For now, just track the click position for potential future use
                tracing::debug!("Mouse click at ({}, {})", mouse.column, mouse.row);
            }
            MouseEventKind::ScrollUp => {
                self.select_prev();
            }
            MouseEventKind::ScrollDown => {
                self.select_next();
            }
            _ => {}
        }
    }

    /// Handle terminal resize
    fn handle_resize(&mut self, _width: u16, _height: u16) {
        // Terminal handles this automatically, but we can use it for responsive layouts
    }

    /// Called on each tick
    async fn on_tick(&mut self) {
        self.tick_count += 1;

        // Remove old notifications (older than 5 seconds)
        let now = chrono::Utc::now();
        self.notifications
            .retain(|n| (now - n.timestamp).num_seconds() < 5);

        // Process any pending log events
        while let Some(log_event) = self.log_streams.try_recv() {
            tracing::debug!("Received log event: {}", log_event.line);
            // TODO: Store log events in screen state for display
        }

        // Periodic data refresh (every 30 seconds based on default 250ms tick rate)
        let refresh_interval_ticks = (self.config.refresh.balance * 1000) / 250;
        if self.tick_count.is_multiple_of(refresh_interval_ticks)
            && self.last_refresh.elapsed().as_secs() >= self.config.refresh.balance
        {
            self.last_refresh = std::time::Instant::now();
            // Auto-refresh data in background
            // self.refresh_data().await;
        }
    }

    /// Toggle between user and miner mode
    fn toggle_mode(&mut self) {
        self.mode = match self.mode {
            AppMode::User => AppMode::Miner,
            AppMode::Miner => AppMode::User,
        };
        self.selected_index = 0;
    }

    /// Navigate to next screen
    fn next_screen(&mut self) {
        match self.mode {
            AppMode::User => {
                let idx = (self.user_screen.index() + 1) % UserScreen::all().len();
                self.user_screen = UserScreen::from_index(idx);
            }
            AppMode::Miner => {
                let idx = (self.miner_screen.index() + 1) % MinerScreen::all().len();
                self.miner_screen = MinerScreen::from_index(idx);
            }
        }
        self.selected_index = 0;
    }

    /// Navigate to previous screen
    fn prev_screen(&mut self) {
        match self.mode {
            AppMode::User => {
                let len = UserScreen::all().len();
                let idx = (self.user_screen.index() + len - 1) % len;
                self.user_screen = UserScreen::from_index(idx);
            }
            AppMode::Miner => {
                let len = MinerScreen::all().len();
                let idx = (self.miner_screen.index() + len - 1) % len;
                self.miner_screen = MinerScreen::from_index(idx);
            }
        }
        self.selected_index = 0;
    }

    /// Go to specific screen by index
    fn goto_screen(&mut self, index: usize) {
        match self.mode {
            AppMode::User => {
                if index < UserScreen::all().len() {
                    self.user_screen = UserScreen::from_index(index);
                }
            }
            AppMode::Miner => {
                if index < MinerScreen::all().len() {
                    self.miner_screen = MinerScreen::from_index(index);
                }
            }
        }
        self.selected_index = 0;
    }

    /// Select next item in list
    fn select_next(&mut self) {
        let max = self.get_current_list_len();
        if max > 0 {
            self.selected_index = (self.selected_index + 1) % max;
        }
    }

    /// Select previous item in list
    fn select_prev(&mut self) {
        let max = self.get_current_list_len();
        if max > 0 {
            self.selected_index = (self.selected_index + max - 1) % max;
        }
    }

    /// Get the length of the current list being displayed
    fn get_current_list_len(&self) -> usize {
        // TODO: Return actual list lengths based on current screen and data
        match self.mode {
            AppMode::User => match self.user_screen {
                UserScreen::Rentals => 10,     // placeholder
                UserScreen::Marketplace => 20, // placeholder
                UserScreen::Deployments => 5,  // placeholder
                _ => 0,
            },
            AppMode::Miner => match self.miner_screen {
                MinerScreen::Fleet => 8, // placeholder
                MinerScreen::Nodes => 4, // placeholder
                _ => 0,
            },
        }
    }

    /// Refresh data from API
    async fn refresh_data(&mut self) {
        let client = match &self.client {
            Some(c) => c.clone(),
            None => {
                self.add_notification(
                    "Not connected to API. Run 'basilica login' first.",
                    NotificationLevel::Warning,
                );
                return;
            }
        };

        self.add_notification("Refreshing data...", NotificationLevel::Info);
        debug!("Starting data refresh...");

        match self.mode {
            AppMode::User => {
                let mut user_data = self.user_data.write().await;
                if let Err(e) = user_data.refresh_all(&client).await {
                    error!("Failed to refresh user data: {}", e);
                    drop(user_data);
                    self.add_notification(
                        &format!("Refresh failed: {}", e),
                        NotificationLevel::Error,
                    );
                    return;
                }

                // Update notification with summary
                let msg = format!(
                    "Loaded {} rentals, {} GPUs available",
                    user_data.rentals.len(),
                    user_data.offerings.len()
                );
                drop(user_data);
                self.add_notification(&msg, NotificationLevel::Success);
            }
            AppMode::Miner => {
                // TODO: Implement miner data refresh when miner APIs are available
                self.add_notification(
                    "Miner mode data refresh not yet implemented",
                    NotificationLevel::Warning,
                );
            }
        }

        self.last_refresh = std::time::Instant::now();
    }

    /// Add a notification
    pub fn add_notification(&mut self, message: &str, level: NotificationLevel) {
        self.notifications.push(Notification {
            message: message.to_string(),
            level,
            timestamp: chrono::Utc::now(),
        });
        // Keep only last 5 notifications
        if self.notifications.len() > 5 {
            self.notifications.remove(0);
        }
    }

    // Action methods (stubs for now)
    async fn ssh_into_rental(&mut self) {
        self.add_notification("SSH not yet implemented", NotificationLevel::Warning);
    }

    async fn stop_rental(&mut self) {
        self.add_notification(
            "Stop rental not yet implemented",
            NotificationLevel::Warning,
        );
    }

    async fn start_rental(&mut self) {
        self.add_notification(
            "Start rental not yet implemented",
            NotificationLevel::Warning,
        );
    }

    async fn delete_deployment(&mut self) {
        self.add_notification(
            "Delete deployment not yet implemented",
            NotificationLevel::Warning,
        );
    }
}

/// Standalone render function that takes a render context
fn render_app(frame: &mut ratatui::Frame, ctx: &RenderContext) {
    // Render based on current mode and screen
    match ctx.mode {
        AppMode::User => match ctx.user_screen {
            UserScreen::Dashboard => crate::ui::screens::dashboard::render_with_ctx(frame, ctx),
            UserScreen::Rentals => crate::ui::screens::rentals::render_with_ctx(frame, ctx),
            UserScreen::Marketplace => crate::ui::screens::marketplace::render_with_ctx(frame, ctx),
            UserScreen::Deployments => crate::ui::screens::deployments::render_with_ctx(frame, ctx),
            UserScreen::Billing => crate::ui::screens::billing::render_with_ctx(frame, ctx),
        },
        AppMode::Miner => match ctx.miner_screen {
            MinerScreen::Fleet => crate::ui::screens::miner::fleet::render_with_ctx(frame, ctx),
            MinerScreen::Validators => {
                crate::ui::screens::miner::validators::render_with_ctx(frame, ctx)
            }
            MinerScreen::Nodes => crate::ui::screens::miner::nodes::render_with_ctx(frame, ctx),
            MinerScreen::Earnings => {
                crate::ui::screens::miner::earnings::render_with_ctx(frame, ctx)
            }
            MinerScreen::Logs => crate::ui::screens::miner::logs::render_with_ctx(frame, ctx),
        },
    }

    // Render help overlay if active
    if ctx.show_help {
        crate::ui::components::help::render_help_overlay_ctx(frame, ctx);
    }

    // Render notifications
    crate::ui::components::notifications::render_notifications_ctx(frame, ctx);
}

/// Render context for passing to render functions
#[allow(dead_code)]
pub struct RenderContext<'a> {
    pub mode: AppMode,
    pub user_screen: UserScreen,
    pub miner_screen: MinerScreen,
    pub show_help: bool,
    pub theme: &'a crate::ui::Theme,
    pub selected_index: usize,
    pub screens: &'a ScreenStates,
    pub notifications: &'a [Notification],
    pub user_data: &'a crate::data::UserData,
    pub miner_data: &'a crate::data::MinerData,
    pub connected: bool,
}
