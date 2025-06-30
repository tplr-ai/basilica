//! API module for the Public API Gateway

pub mod middleware;
pub mod routes;
pub mod types;

use crate::server::AppState;
use axum::{
    routing::{get, post},
    Router,
};
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;

/// Create all API routes
pub fn routes(state: AppState) -> Router<AppState> {
    let router = Router::new()
        // Rental endpoints
        .route("/rentals", post(routes::rentals::rent_capacity))
        .route(
            "/rentals/:rental_id",
            get(routes::rentals::get_rental_status),
        )
        .route(
            "/rentals/:rental_id/terminate",
            post(routes::rentals::terminate_rental),
        )
        // Log endpoints
        .route(
            "/rentals/:rental_id/logs",
            get(routes::logs::stream_rental_logs),
        )
        // Executor endpoints
        .route("/executors", get(routes::executors::list_executors))
        .route(
            "/executors/:executor_id",
            get(routes::executors::get_executor),
        )
        // Validator endpoints
        .route("/validators", get(routes::validators::list_validators))
        .route(
            "/validators/:validator_id",
            get(routes::validators::get_validator),
        )
        // Miner endpoints
        .route("/miners", get(routes::miners::list_miners))
        .route("/miners/:miner_id", get(routes::miners::get_miner))
        // Health and telemetry
        .route("/health", get(routes::health::health_check))
        .route("/telemetry", get(routes::telemetry::get_telemetry))
        .with_state(state.clone());

    // Apply middleware
    middleware::apply_middleware(router, state)
}

/// Create OpenAPI documentation routes
pub fn docs_routes() -> Router<AppState> {
    Router::new()
        .merge(SwaggerUi::new("/swagger-ui").url("/api-docs/openapi.json", ApiDoc::openapi()))
        .route("/api-docs/openapi.json", get(openapi_json))
}

/// OpenAPI documentation
#[derive(OpenApi)]
#[openapi(
    paths(
        routes::rentals::rent_capacity,
        routes::rentals::get_rental_status,
        routes::rentals::terminate_rental,
        routes::logs::stream_rental_logs,
        routes::executors::list_executors,
        routes::executors::get_executor,
        routes::validators::list_validators,
        routes::validators::get_validator,
        routes::miners::list_miners,
        routes::miners::get_miner,
        routes::health::health_check,
        routes::telemetry::get_telemetry,
    ),
    components(schemas(
        types::RentCapacityRequest,
        types::RentCapacityResponse,
        types::RentalStatusResponse,
        types::TerminateRentalRequest,
        types::TerminateRentalResponse,
        types::ExecutorDetails,
        types::ValidatorDetails,
        types::MinerDetails,
        types::HealthCheckResponse,
        types::TelemetryResponse,
        crate::error::ErrorResponse,
    )),
    tags(
        (name = "rentals", description = "GPU rental management"),
        (name = "logs", description = "Log streaming"),
        (name = "executors", description = "Executor information"),
        (name = "validators", description = "Validator information"),
        (name = "miners", description = "Miner information"),
        (name = "health", description = "Health and monitoring"),
    ),
    info(
        title = "Basilica Public API",
        version = "1.0.0",
        description = "Public API Gateway for the Basilica validator network",
        contact(
            name = "Basilica Team",
            email = "support@tplr.ai",
        ),
        license(
            name = "MIT",
        ),
    ),
    servers(
        (url = "http://localhost:8000", description = "Local development"),
    ),
)]
struct ApiDoc;

/// Serve OpenAPI JSON
async fn openapi_json() -> impl axum::response::IntoResponse {
    axum::Json(ApiDoc::openapi())
}
