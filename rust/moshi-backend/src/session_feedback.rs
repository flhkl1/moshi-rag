// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

//! POST `/api/session_feedback` — forwards JSON to `MOSHI_FEEDBACK_WEBHOOK_URL` (e.g. Google Apps Script web app that appends rows to a Sheet).

use axum::{http::StatusCode, response::IntoResponse, Json};
use lazy_static::lazy_static;
use serde_json::Value;

lazy_static! {
    /// Apps Script cold starts can exceed 15s; keep under typical proxy limits.
    static ref HTTP: reqwest::Client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .connect_timeout(std::time::Duration::from_secs(10))
        .build()
        .expect("reqwest client for session_feedback");
}

pub async fn session_feedback_handler(Json(body): Json<Value>) -> impl IntoResponse {
    let url = match std::env::var("MOSHI_FEEDBACK_WEBHOOK_URL") {
        Ok(u) if !u.trim().is_empty() => u,
        _ => {
            tracing::warn!(
                "session_feedback: MOSHI_FEEDBACK_WEBHOOK_URL not set; accepting but not forwarding"
            );
            return (StatusCode::ACCEPTED, "feedback webhook not configured").into_response();
        }
    };

    match HTTP.post(&url).json(&body).send().await {
        Ok(resp) if resp.status().is_success() => (StatusCode::NO_CONTENT, "").into_response(),
        Ok(resp) => {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            tracing::error!(%status, %text, "session_feedback: webhook returned error");
            (StatusCode::BAD_GATEWAY, "upstream webhook error").into_response()
        }
        Err(e) => {
            tracing::error!(?e, "session_feedback: request failed");
            (StatusCode::BAD_GATEWAY, "webhook request failed").into_response()
        }
    }
}
