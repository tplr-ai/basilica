use axum::{
    extract::{Path, State},
    http::{header, StatusCode},
    response::IntoResponse,
};
use tokio::fs;

use crate::api::ApiState;

pub async fn get_evidence(
    State(state): State<ApiState>,
    Path(file_name): Path<String>,
) -> impl IntoResponse {
    if !is_valid_evidence_name(&file_name) {
        return StatusCode::NOT_FOUND.into_response();
    }

    let path = state.evidence_storage_path.join(&file_name);
    let contents = match fs::read(&path).await {
        Ok(contents) => contents,
        Err(_) => return StatusCode::NOT_FOUND.into_response(),
    };

    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "application/json")],
        contents,
    )
        .into_response()
}

fn is_valid_evidence_name(name: &str) -> bool {
    if !name.starts_with("evidence-") || !name.ends_with(".json") {
        return false;
    }
    if name.contains("..") || name.contains('/') || name.contains('\\') {
        return false;
    }
    true
}
