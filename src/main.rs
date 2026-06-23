use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use chrono::{DateTime, Utc};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::{env, net::SocketAddr, sync::Arc};
use thiserror::Error;
use tower_http::{cors::CorsLayer, trace::TraceLayer};
use tracing::{info, warn};
use uuid::Uuid;

#[derive(Clone)]
struct AppState {
    openai: OpenAiClient,
}

#[derive(Clone)]
struct OpenAiClient {
    http: Client,
    api_key: Option<String>,
    model: String,
}

#[derive(Debug, Serialize)]
struct HealthResponse {
    service: &'static str,
    status: &'static str,
    ai_layer: &'static str,
    timestamp: DateTime<Utc>,
}

#[derive(Debug, Serialize)]
struct SeasonResponse {
    season: String,
    thesis: String,
    tracks: Vec<Track>,
    gates: Vec<Gate>,
}

#[derive(Debug, Serialize)]
struct Track {
    name: String,
    description: String,
    sample_quests: Vec<String>,
}

#[derive(Debug, Serialize)]
struct Gate {
    name: String,
    unlocks: String,
}

#[derive(Debug, Deserialize)]
struct GenerateQuestRequest {
    build_prompt: String,
    skill_track: Option<String>,
    difficulty: Option<Difficulty>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
enum Difficulty {
    Novice,
    Builder,
    Boss,
}

#[derive(Debug, Serialize)]
struct GenerateQuestResponse {
    run_id: Uuid,
    source: QuestSource,
    quest: QuestBlueprint,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "kebab-case")]
enum QuestSource {
    OpenAi,
    Fallback,
}

#[derive(Debug, Deserialize, Serialize)]
struct QuestBlueprint {
    title: String,
    premise: String,
    build_objective: String,
    comprehension_gates: Vec<String>,
    boss_fight: String,
    reward_logic: String,
    ckb_fiber_hooks: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAiResponse {
    output_text: Option<String>,
}

#[derive(Debug, Error)]
enum ApiError {
    #[error("build_prompt must be at least 12 characters")]
    InvalidPrompt,
    #[error("openai request failed: {0}")]
    OpenAi(#[from] reqwest::Error),
    #[error("openai response did not contain valid quest json")]
    InvalidAiResponse,
}

#[tokio::main]
async fn main() {
    dotenvy::dotenv().ok();

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "vibequest_core=info,tower_http=info".into()),
        )
        .init();

    let port = env::var("PORT")
        .ok()
        .and_then(|value| value.parse::<u16>().ok())
        .unwrap_or(8080);

    let state = Arc::new(AppState {
        openai: OpenAiClient {
            http: Client::new(),
            api_key: env::var("OPENAI_API_KEY").ok(),
            model: env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-5.4-mini".to_string()),
        },
    });

    if state.openai.api_key.is_none() {
        warn!("OPENAI_API_KEY is not set; /ai/quests/generate will use fallback quest generation");
    }

    let app = Router::new()
        .route("/health", get(health))
        .route("/season", get(season))
        .route("/ai/quests/generate", post(generate_quest))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .expect("failed to bind TCP listener");

    info!("vibequest-core listening on http://{addr}");

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .expect("server failed");
}

async fn health(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    Json(HealthResponse {
        service: "vibequest-core",
        status: "ok",
        ai_layer: if state.openai.api_key.is_some() {
            "openai"
        } else {
            "fallback"
        },
        timestamp: Utc::now(),
    })
}

async fn season() -> Json<SeasonResponse> {
    Json(SeasonResponse {
        season: "Season 0: Escape Black Box Mode".to_string(),
        thesis: "Vibecode a real app, then unlock shipping by explaining, debugging, testing, and remixing the code.".to_string(),
        tracks: vec![
            Track {
                name: "CKB Fundamentals".to_string(),
                description: "Learn the Cell model, transactions, xUDT assets, and proof receipts through generated app missions.".to_string(),
                sample_quests: vec![
                    "Cell Lab Escape".to_string(),
                    "Forge an xUDT".to_string(),
                    "Proof Receipt Mint".to_string(),
                ],
            },
            Track {
                name: "Fiber Builder".to_string(),
                description: "Build paywalls, rewards, game loops, and creator payouts that use Fiber-style instant payments.".to_string(),
                sample_quests: vec![
                    "Paywall Reactor".to_string(),
                    "Channel Gate".to_string(),
                    "No-Prompt Checkout".to_string(),
                ],
            },
            Track {
                name: "AI Discipline".to_string(),
                description: "Use AI aggressively while proving you can reason about, defend, and extend generated code.".to_string(),
                sample_quests: vec![
                    "Prompt Budget Trial".to_string(),
                    "Explain Room".to_string(),
                    "Boss Diff Defense".to_string(),
                ],
            },
        ],
        gates: vec![
            Gate {
                name: "Explain".to_string(),
                unlocks: "User explains the generated subsystem in their own words.".to_string(),
            },
            Gate {
                name: "Debug".to_string(),
                unlocks: "User fixes a seeded bug and passes tests.".to_string(),
            },
            Gate {
                name: "Remix".to_string(),
                unlocks: "User extends the feature with limited AI help.".to_string(),
            },
            Gate {
                name: "Attack".to_string(),
                unlocks: "User finds or defends against a real failure mode.".to_string(),
            },
            Gate {
                name: "Ship".to_string(),
                unlocks: "CKB proof badge and Fiber reward become claimable.".to_string(),
            },
        ],
    })
}

async fn generate_quest(
    State(state): State<Arc<AppState>>,
    Json(request): Json<GenerateQuestRequest>,
) -> Result<Json<GenerateQuestResponse>, ApiError> {
    if request.build_prompt.trim().chars().count() < 12 {
        return Err(ApiError::InvalidPrompt);
    }

    let run_id = Uuid::new_v4();

    let (source, quest) = match state.openai.generate_quest(&request).await {
        Ok(quest) => (QuestSource::OpenAi, quest),
        Err(error) if state.openai.api_key.is_none() => {
            warn!("using fallback quest generation: {error}");
            (QuestSource::Fallback, fallback_quest(&request))
        }
        Err(error) => return Err(error),
    };

    Ok(Json(GenerateQuestResponse {
        run_id,
        source,
        quest,
    }))
}

impl OpenAiClient {
    async fn generate_quest(
        &self,
        request: &GenerateQuestRequest,
    ) -> Result<QuestBlueprint, ApiError> {
        let Some(api_key) = self.api_key.as_ref() else {
            return Err(ApiError::InvalidAiResponse);
        };

        let difficulty = request.difficulty.clone().unwrap_or(Difficulty::Builder);
        let track = request
            .skill_track
            .as_deref()
            .unwrap_or("CKB + Fiber Builder");

        let prompt = format!(
            r#"You are VibeQuest's quest designer.

Create one gamified programming quest for a vibecoder who asked the AI to build:
"{build_prompt}"

Skill track: {track}
Difficulty: {difficulty:?}

Return compact JSON only with this exact shape:
{{
  "title": "short quest name",
  "premise": "one sentence game premise",
  "build_objective": "what the AI-generated app should build",
  "comprehension_gates": ["Explain...", "Debug...", "Remix...", "Attack...", "Ship..."],
  "boss_fight": "final challenge",
  "reward_logic": "how XP/Fiber/credential rewards unlock",
  "ckb_fiber_hooks": ["CKB/Fiber integration hook", "another hook"]
}}

Make the quest playful, practical, and focused on proving understanding of generated code."#,
            build_prompt = request.build_prompt.trim()
        );

        let body = serde_json::json!({
            "model": self.model,
            "input": prompt,
            "text": {
                "format": {
                    "type": "json_object"
                }
            }
        });

        let response = self
            .http
            .post("https://api.openai.com/v1/responses")
            .bearer_auth(api_key)
            .json(&body)
            .send()
            .await?
            .error_for_status()?
            .json::<OpenAiResponse>()
            .await?;

        let output_text = response.output_text.ok_or(ApiError::InvalidAiResponse)?;
        serde_json::from_str::<QuestBlueprint>(&output_text)
            .map_err(|_| ApiError::InvalidAiResponse)
    }
}

fn fallback_quest(request: &GenerateQuestRequest) -> QuestBlueprint {
    let track = request
        .skill_track
        .clone()
        .unwrap_or_else(|| "CKB + Fiber Builder".to_string());

    QuestBlueprint {
        title: "Black Box Breakout".to_string(),
        premise: format!(
            "You vibecoded a {track} app, but the ship gate is sealed until you prove ownership."
        ),
        build_objective: request.build_prompt.trim().to_string(),
        comprehension_gates: vec![
            "Explain the generated architecture and name the trust boundary.".to_string(),
            "Debug a seeded payment or auth failure without asking for a full rewrite.".to_string(),
            "Remix one feature under a three-prompt budget.".to_string(),
            "Attack the generated code path and document the exploit.".to_string(),
            "Ship after tests pass and the comprehension meter clears 80%.".to_string(),
        ],
        boss_fight: "A generated shortcut lets users claim rewards without proving payment; patch it and defend the fix.".to_string(),
        reward_logic: "XP unlocks at each gate; Fiber rewards and the CKB proof badge unlock only after the boss fight.".to_string(),
        ckb_fiber_hooks: vec![
            "CKB records the proof-of-understanding badge and quest receipt.".to_string(),
            "Fiber pays instant quest rewards, hint fees, and sponsor bounties.".to_string(),
        ],
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = match self {
            ApiError::InvalidPrompt => StatusCode::BAD_REQUEST,
            ApiError::OpenAi(_) | ApiError::InvalidAiResponse => StatusCode::BAD_GATEWAY,
        };

        let body = serde_json::json!({
            "error": self.to_string(),
        });

        (status, Json(body)).into_response()
    }
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
}
