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

    /// Dev mode - use mock data
    pub dev_mode: bool,

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
    pub async fn new(
        config: TuiConfig,
        miner_mode: bool,
        dev_mode: bool,
        tick_rate: u64,
    ) -> Result<Self> {
        let theme = Theme::dark(); // TODO: Respect config.theme preference

        let event_handler = EventHandler::new(tick_rate);
        let tui = Tui::new()?;

        let user_data = if dev_mode {
            Arc::new(RwLock::new(Self::create_mock_user_data()))
        } else {
            Arc::new(RwLock::new(UserData::default()))
        };

        let miner_data = if dev_mode {
            Arc::new(RwLock::new(Self::create_mock_miner_data()))
        } else {
            Arc::new(RwLock::new(MinerData::default()))
        };

        let log_streams = LogStreamManager::new(config.api_url.clone());

        // Skip API client in dev mode
        let client = if dev_mode {
            info!("Dev mode enabled - using mock data");
            None
        } else {
            // Try to create API client with file-based auth
            match Self::create_client(&config.api_url).await {
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
            }
        };

        let mut notifications = Vec::new();
        if dev_mode {
            notifications.push(Notification {
                message: "🔧 Dev mode active - using mock data".to_string(),
                level: NotificationLevel::Info,
                timestamp: chrono::Utc::now(),
            });
        }

        Ok(Self {
            running: true,
            mode: if miner_mode {
                AppMode::Miner
            } else {
                AppMode::User
            },
            dev_mode,
            user_screen: UserScreen::default(),
            miner_screen: MinerScreen::default(),
            show_help: false,
            config,
            theme,
            client,
            user_data,
            miner_data,
            notifications,
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

    /// Create mock user data for dev mode
    #[allow(clippy::field_reassign_with_default)]
    fn create_mock_user_data() -> UserData {
        use crate::data::user::{BalanceInfo, DeploymentInfo, GpuOffering, RentalInfo};

        let mut data = UserData::default();

        // Mock rentals
        data.rentals = vec![
            RentalInfo {
                id: "rental-abc123".to_string(),
                gpu_type: "NVIDIA RTX 4090".to_string(),
                gpu_count: 1,
                status: "running".to_string(),
                uptime_minutes: 300, // 5 hours
                cost: 4.25,
                container_image: "pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime".to_string(),
                ssh_host: Some("gpu-001.basilica.cloud".to_string()),
                ssh_port: Some(22),
                ssh_user: Some("root".to_string()),
            },
            RentalInfo {
                id: "rental-def456".to_string(),
                gpu_type: "NVIDIA A100 80GB".to_string(),
                gpu_count: 2,
                status: "running".to_string(),
                uptime_minutes: 720, // 12 hours
                cost: 30.00,
                container_image: "nvcr.io/nvidia/pytorch:23.10-py3".to_string(),
                ssh_host: Some("gpu-002.basilica.cloud".to_string()),
                ssh_port: Some(22),
                ssh_user: Some("root".to_string()),
            },
            RentalInfo {
                id: "rental-ghi789".to_string(),
                gpu_type: "NVIDIA H100".to_string(),
                gpu_count: 1,
                status: "pending".to_string(),
                uptime_minutes: 0,
                cost: 0.0,
                container_image: "vllm/vllm-openai:latest".to_string(),
                ssh_host: None,
                ssh_port: None,
                ssh_user: None,
            },
        ];

        // Mock offerings
        data.offerings = vec![
            GpuOffering {
                gpu_type: "NVIDIA RTX 4090".to_string(),
                gpu_count: 1,
                memory_gb: 24,
                price_per_hour: 0.85,
                source: "hyperstack".to_string(),
                available: 12,
                node_id: Some("node-001".to_string()),
            },
            GpuOffering {
                gpu_type: "NVIDIA RTX 4090".to_string(),
                gpu_count: 4,
                memory_gb: 24,
                price_per_hour: 3.20,
                source: "miner".to_string(),
                available: 3,
                node_id: Some("node-002".to_string()),
            },
            GpuOffering {
                gpu_type: "NVIDIA A100 80GB".to_string(),
                gpu_count: 1,
                memory_gb: 80,
                price_per_hour: 2.50,
                source: "miner".to_string(),
                available: 8,
                node_id: Some("node-003".to_string()),
            },
            GpuOffering {
                gpu_type: "NVIDIA A100 80GB".to_string(),
                gpu_count: 8,
                memory_gb: 80,
                price_per_hour: 18.00,
                source: "hyperstack".to_string(),
                available: 2,
                node_id: None,
            },
            GpuOffering {
                gpu_type: "NVIDIA H100".to_string(),
                gpu_count: 1,
                memory_gb: 80,
                price_per_hour: 4.00,
                source: "miner".to_string(),
                available: 0, // Sold out
                node_id: Some("node-005".to_string()),
            },
            GpuOffering {
                gpu_type: "NVIDIA L40S".to_string(),
                gpu_count: 2,
                memory_gb: 48,
                price_per_hour: 1.80,
                source: "hyperstack".to_string(),
                available: 5,
                node_id: None,
            },
        ];

        // Mock deployments
        data.deployments = vec![
            DeploymentInfo {
                name: "llama-inference".to_string(),
                deployment_type: "inference".to_string(),
                status: "running".to_string(),
                replicas_ready: 2,
                replicas_desired: 2,
                gpu_type: "NVIDIA RTX 4090".to_string(),
                gpu_count: 2,
                url: Some("https://llama.basilica.cloud/v1".to_string()),
            },
            DeploymentInfo {
                name: "stable-diffusion-xl".to_string(),
                deployment_type: "inference".to_string(),
                status: "running".to_string(),
                replicas_ready: 1,
                replicas_desired: 1,
                gpu_type: "NVIDIA A100 80GB".to_string(),
                gpu_count: 1,
                url: Some("https://sdxl.basilica.cloud/generate".to_string()),
            },
            DeploymentInfo {
                name: "training-job-42".to_string(),
                deployment_type: "training".to_string(),
                status: "pending".to_string(),
                replicas_ready: 0,
                replicas_desired: 1,
                gpu_type: "NVIDIA H100".to_string(),
                gpu_count: 4,
                url: None,
            },
        ];

        // Mock balance
        data.balance = Some(BalanceInfo {
            available_tao: 125.75,
            available_usd: 628.75, // Assuming $5/TAO
            spent_today: 12.50,
            spent_this_month: 245.80,
            active_spend_rate: 3.35, // Current hourly spend
        });

        data
    }

    /// Create mock miner data for dev mode
    #[allow(clippy::field_reassign_with_default)]
    fn create_mock_miner_data() -> MinerData {
        use crate::data::miner::{
            EarningsData, LogEntry, LogLevel, MinerInfo, NodeInfo, NodeStatus, PaymentInfo,
            ValidatorInfo, ValidatorStatus,
        };

        let mut data = MinerData::new();

        // Mock nodes
        data.nodes = vec![
            NodeInfo {
                id: "node-001".to_string(),
                host: "gpu-server-1.local".to_string(),
                port: 50051,
                username: "miner".to_string(),
                gpu_type: "NVIDIA RTX 4090".to_string(),
                gpu_count: 4,
                status: NodeStatus::Healthy,
                gpu_utilization: 78.5,
                memory_utilization: 75.0,
                assigned_gpus: 3,
                uptime_hours: 720.5,
            },
            NodeInfo {
                id: "node-002".to_string(),
                host: "gpu-server-2.local".to_string(),
                port: 50051,
                username: "miner".to_string(),
                gpu_type: "NVIDIA A100 80GB".to_string(),
                gpu_count: 8,
                status: NodeStatus::Healthy,
                gpu_utilization: 95.2,
                memory_utilization: 90.6,
                assigned_gpus: 8,
                uptime_hours: 1440.0,
            },
            NodeInfo {
                id: "node-003".to_string(),
                host: "gpu-server-3.local".to_string(),
                port: 50051,
                username: "miner".to_string(),
                gpu_type: "NVIDIA RTX 3090".to_string(),
                gpu_count: 2,
                status: NodeStatus::Offline,
                gpu_utilization: 0.0,
                memory_utilization: 0.0,
                assigned_gpus: 0,
                uptime_hours: 0.0,
            },
            NodeInfo {
                id: "node-004".to_string(),
                host: "gpu-server-4.local".to_string(),
                port: 50051,
                username: "miner".to_string(),
                gpu_type: "NVIDIA H100".to_string(),
                gpu_count: 4,
                status: NodeStatus::Warning,
                gpu_utilization: 45.0,
                memory_utilization: 30.0,
                assigned_gpus: 2,
                uptime_hours: 168.0,
            },
        ];

        // Mock validators
        data.validators = vec![
            ValidatorInfo {
                hotkey: "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY".to_string(),
                name: "TaoStats".to_string(),
                stake: 50000.0,
                status: ValidatorStatus::Active,
                assigned_gpus: 8,
                assigned_nodes: vec!["node-001".to_string(), "node-002".to_string()],
            },
            ValidatorInfo {
                hotkey: "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty".to_string(),
                name: "Foundry".to_string(),
                stake: 35000.0,
                status: ValidatorStatus::Active,
                assigned_gpus: 3,
                assigned_nodes: vec!["node-001".to_string()],
            },
            ValidatorInfo {
                hotkey: "5DAAnrj7VHTznn2AWBemMuyBwZWs6FNFjdyVXUeYum3PTXFy".to_string(),
                name: "Subnet42".to_string(),
                stake: 12000.0,
                status: ValidatorStatus::Pending,
                assigned_gpus: 0,
                assigned_nodes: vec![],
            },
        ];

        // Mock miner info
        data.miner_info = Some(MinerInfo {
            uid: 42,
            hotkey: "5CiPPseXPECbkjWCa6MnjNokrgYjMqmKndv2rSnekmSK2DjL".to_string(),
            stake: 1250.5,
            netuid: 64,
            network: "finney".to_string(),
            axon_port: 8080,
        });

        // Mock earnings
        data.earnings = EarningsData {
            current_rate_per_hour: 0.53,
            today: 12.75,
            this_week: 89.20,
            this_month: 385.50,
            revenue_history: vec![8.5, 10.2, 12.1, 9.8, 11.5, 13.2, 12.75],
            payments: vec![
                PaymentInfo {
                    date: "2024-12-25".to_string(),
                    validator: "TaoStats".to_string(),
                    description: "GPU rental payment".to_string(),
                    amount: 45.30,
                    status: "completed".to_string(),
                },
                PaymentInfo {
                    date: "2024-12-24".to_string(),
                    validator: "Foundry".to_string(),
                    description: "GPU rental payment".to_string(),
                    amount: 32.15,
                    status: "completed".to_string(),
                },
                PaymentInfo {
                    date: "2024-12-23".to_string(),
                    validator: "TaoStats".to_string(),
                    description: "GPU rental payment".to_string(),
                    amount: 28.90,
                    status: "completed".to_string(),
                },
            ],
        };

        // Mock logs
        data.logs = vec![
            LogEntry {
                timestamp: "2024-12-26 10:15:32".to_string(),
                level: LogLevel::Info,
                source: "miner".to_string(),
                message: "New rental started on node-001 (GPU 0-2)".to_string(),
            },
            LogEntry {
                timestamp: "2024-12-26 10:14:28".to_string(),
                level: LogLevel::Info,
                source: "executor".to_string(),
                message: "Container started: llama-inference-abc123".to_string(),
            },
            LogEntry {
                timestamp: "2024-12-26 10:12:15".to_string(),
                level: LogLevel::Warn,
                source: "node-003".to_string(),
                message: "Node health check failed - attempting reconnect".to_string(),
            },
            LogEntry {
                timestamp: "2024-12-26 10:10:00".to_string(),
                level: LogLevel::Info,
                source: "validator".to_string(),
                message: "Heartbeat from TaoStats validator".to_string(),
            },
            LogEntry {
                timestamp: "2024-12-26 10:05:45".to_string(),
                level: LogLevel::Error,
                source: "node-003".to_string(),
                message: "Connection lost to gpu-server-3.local".to_string(),
            },
        ];

        // Populate metrics history with sample data
        for i in 0..20 {
            let gpu = 60.0 + (i as f64 * 1.5) + (fastrand::f64() * 10.0);
            let mem = 50.0 + (i as f64 * 2.0) + (fastrand::f64() * 15.0);
            data.metrics_history.push_gpu_util(gpu.min(100.0));
            data.metrics_history.push_memory_util(mem.min(100.0));
        }

        data
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
        // In dev mode, data is already mocked - just update last refresh time
        if self.dev_mode {
            self.last_refresh = std::time::Instant::now();
            return;
        }

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
