use axum::{
    Json, Router,
    extract::State,
    http::{HeaderValue, Method, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use chrono::{DateTime, Utc};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{env, net::SocketAddr, sync::Arc, time::Duration};
use thiserror::Error;
use tower_http::{
    cors::{AllowOrigin, CorsLayer},
    trace::TraceLayer,
};
use tracing::{info, warn};
use uuid::Uuid;

const DEFAULT_OPENAI_MODEL: &str = "gpt-5.4-mini";
const DEFAULT_OPENAI_BASE_URL: &str = "https://api.openai.com/v1";

#[derive(Clone)]
struct AppState {
    config: AppConfig,
    openai: OpenAiClient,
}

#[derive(Clone)]
struct AppConfig {
    port: u16,
    app_env: String,
    cors_origins: Vec<String>,
    ckb_rpc_url: Option<String>,
    fiber_rpc_url: Option<String>,
}

#[derive(Clone)]
struct OpenAiClient {
    http: Client,
    api_key: Option<String>,
    model: String,
    base_url: String,
    timeout: Duration,
}

#[derive(Debug, Serialize)]
struct HealthResponse {
    service: &'static str,
    status: &'static str,
    environment: String,
    ai_layer: AiLayer,
    integrations: IntegrationStatus,
    timestamp: DateTime<Utc>,
}

#[derive(Debug, Serialize)]
struct ReadyResponse {
    ready: bool,
    missing: Vec<&'static str>,
    timestamp: DateTime<Utc>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "kebab-case")]
enum AiLayer {
    OpenAi,
    Fallback,
}

#[derive(Debug, Serialize)]
struct IntegrationStatus {
    openai: bool,
    ckb_rpc: bool,
    fiber_rpc: bool,
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
    output: Option<Vec<OpenAiOutputItem>>,
}

#[derive(Debug, Deserialize)]
struct OpenAiOutputItem {
    content: Option<Vec<OpenAiContentItem>>,
}

#[derive(Debug, Deserialize)]
struct OpenAiContentItem {
    #[serde(rename = "type")]
    content_type: Option<String>,
    text: Option<String>,
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

#[derive(Debug, Serialize)]
struct ErrorResponse {
    error: String,
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

    let config = AppConfig::from_env();
    let state = Arc::new(AppState {
        openai: OpenAiClient::from_env(),
        config,
    });

    if state.openai.api_key.is_none() {
        warn!("OPENAI_API_KEY is not set; /ai/quests/generate will use fallback quest generation");
    }

    let app = build_router(state.clone());
    let addr = SocketAddr::from(([0, 0, 0, 0], state.config.port));
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .expect("failed to bind TCP listener");

    info!("vibequest-core listening on http://{addr}");

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .expect("server failed");
}

fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/ready", get(ready))
        .route("/season", get(season))
        .route("/ai/quests/generate", post(generate_quest))
        .layer(cors_layer(&state.config))
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

fn cors_layer(config: &AppConfig) -> CorsLayer {
    let layer = CorsLayer::new()
        .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
        .allow_headers([axum::http::header::CONTENT_TYPE]);

    if config.cors_origins.iter().any(|origin| origin == "*") {
        return layer.allow_origin(AllowOrigin::any());
    }

    let origins = config
        .cors_origins
        .iter()
        .filter_map(|origin| origin.parse::<HeaderValue>().ok())
        .collect::<Vec<_>>();

    layer.allow_origin(origins)
}

impl AppConfig {
    fn from_env() -> Self {
        Self {
            port: env::var("PORT")
                .ok()
                .and_then(|value| value.parse::<u16>().ok())
                .unwrap_or(8080),
            app_env: env::var("APP_ENV").unwrap_or_else(|_| "development".to_string()),
            cors_origins: parse_csv_env("CORS_ORIGINS", vec!["http://localhost:3000".to_string()]),
            ckb_rpc_url: optional_env("CKB_RPC_URL"),
            fiber_rpc_url: optional_env("FIBER_RPC_URL"),
        }
    }
}

impl OpenAiClient {
    fn from_env() -> Self {
        let timeout_seconds = env::var("OPENAI_TIMEOUT_SECONDS")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(45);

        Self {
            http: Client::new(),
            api_key: optional_env("OPENAI_API_KEY"),
            model: env::var("OPENAI_MODEL").unwrap_or_else(|_| DEFAULT_OPENAI_MODEL.to_string()),
            base_url: env::var("OPENAI_BASE_URL")
                .unwrap_or_else(|_| DEFAULT_OPENAI_BASE_URL.to_string())
                .trim_end_matches('/')
                .to_string(),
            timeout: Duration::from_secs(timeout_seconds),
        }
    }

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

        let prompt = quest_prompt(request.build_prompt.trim(), track, &difficulty);
        let body = serde_json::json!({
            "model": self.model,
            "input": prompt,
            "text": {
                "format": quest_json_schema()
            }
        });

        let response = self
            .http
            .post(format!("{}/responses", self.base_url))
            .bearer_auth(api_key)
            .timeout(self.timeout)
            .json(&body)
            .send()
            .await?
            .error_for_status()?
            .json::<OpenAiResponse>()
            .await?;

        parse_openai_quest_response(response)
    }
}

async fn health(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    Json(HealthResponse {
        service: "vibequest-core",
        status: "ok",
        environment: state.config.app_env.clone(),
        ai_layer: if state.openai.api_key.is_some() {
            AiLayer::OpenAi
        } else {
            AiLayer::Fallback
        },
        integrations: integration_status(&state),
        timestamp: Utc::now(),
    })
}

async fn ready(State(state): State<Arc<AppState>>) -> (StatusCode, Json<ReadyResponse>) {
    let mut missing = Vec::new();

    if state.openai.api_key.is_none() {
        missing.push("OPENAI_API_KEY");
    }

    let status = if missing.is_empty() {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };

    (
        status,
        Json(ReadyResponse {
            ready: missing.is_empty(),
            missing,
            timestamp: Utc::now(),
        }),
    )
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

fn integration_status(state: &AppState) -> IntegrationStatus {
    IntegrationStatus {
        openai: state.openai.api_key.is_some(),
        ckb_rpc: state.config.ckb_rpc_url.is_some(),
        fiber_rpc: state.config.fiber_rpc_url.is_some(),
    }
}

fn optional_env(name: &str) -> Option<String> {
    env::var(name)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn parse_csv_env(name: &str, default: Vec<String>) -> Vec<String> {
    env::var(name)
        .ok()
        .map(|value| {
            value
                .split(',')
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(ToOwned::to_owned)
                .collect::<Vec<_>>()
        })
        .filter(|values| !values.is_empty())
        .unwrap_or(default)
}

fn parse_openai_quest_response(response: OpenAiResponse) -> Result<QuestBlueprint, ApiError> {
    if let Some(output_text) = response.output_text {
        return parse_quest_json(&output_text);
    }

    let text = response
        .output
        .unwrap_or_default()
        .into_iter()
        .flat_map(|item| item.content.unwrap_or_default())
        .filter_map(
            |content| match (content.content_type.as_deref(), content.text) {
                (Some("output_text") | Some("text") | None, Some(text)) => Some(text),
                _ => None,
            },
        )
        .collect::<Vec<_>>()
        .join("\n");

    if text.trim().is_empty() {
        return Err(ApiError::InvalidAiResponse);
    }

    parse_quest_json(&text)
}

fn parse_quest_json(text: &str) -> Result<QuestBlueprint, ApiError> {
    serde_json::from_str::<QuestBlueprint>(text.trim()).map_err(|_| ApiError::InvalidAiResponse)
}

fn quest_prompt(build_prompt: &str, track: &str, difficulty: &Difficulty) -> String {
    format!(
        r#"You are VibeQuest's quest designer.

Create one gamified programming quest for a vibecoder who asked the AI to build:
"{build_prompt}"

Skill track: {track}
Difficulty: {difficulty:?}

Make the quest playful, practical, and focused on proving understanding of generated code. The quest must force the learner to explain, debug, test, attack, remix, and ship the generated app. Include CKB proof and Fiber reward hooks when relevant."#
    )
}

fn quest_json_schema() -> Value {
    serde_json::json!({
        "type": "json_schema",
        "name": "vibequest_quest_blueprint",
        "strict": true,
        "schema": {
            "type": "object",
            "additionalProperties": false,
            "required": [
                "title",
                "premise",
                "build_objective",
                "comprehension_gates",
                "boss_fight",
                "reward_logic",
                "ckb_fiber_hooks"
            ],
            "properties": {
                "title": {
                    "type": "string",
                    "description": "A short quest name."
                },
                "premise": {
                    "type": "string",
                    "description": "One sentence game premise."
                },
                "build_objective": {
                    "type": "string",
                    "description": "What the AI-generated app should build."
                },
                "comprehension_gates": {
                    "type": "array",
                    "minItems": 5,
                    "maxItems": 5,
                    "items": {
                        "type": "string"
                    },
                    "description": "Exactly five gates: explain, debug, remix, attack, ship."
                },
                "boss_fight": {
                    "type": "string",
                    "description": "The final challenge before the learner can ship."
                },
                "reward_logic": {
                    "type": "string",
                    "description": "How XP, Fiber, and credential rewards unlock."
                },
                "ckb_fiber_hooks": {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": 4,
                    "items": {
                        "type": "string"
                    },
                    "description": "CKB/Fiber integration hooks for the quest."
                }
            }
        }
    })
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

        (
            status,
            Json(ErrorResponse {
                error: self.to_string(),
            }),
        )
            .into_response()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_openai_output_text() {
        let quest = sample_quest();
        let response = OpenAiResponse {
            output_text: Some(serde_json::to_string(&quest).unwrap()),
            output: None,
        };

        let parsed = parse_openai_quest_response(response).unwrap();

        assert_eq!(parsed.title, "Receipt Raid");
        assert_eq!(parsed.comprehension_gates.len(), 5);
    }

    #[test]
    fn parses_openai_nested_output_text() {
        let quest = sample_quest();
        let response = OpenAiResponse {
            output_text: None,
            output: Some(vec![OpenAiOutputItem {
                content: Some(vec![OpenAiContentItem {
                    content_type: Some("output_text".to_string()),
                    text: Some(serde_json::to_string(&quest).unwrap()),
                }]),
            }]),
        };

        let parsed = parse_openai_quest_response(response).unwrap();

        assert_eq!(parsed.boss_fight, "Patch the replayable receipt.");
    }

    #[test]
    fn fallback_quest_keeps_user_prompt() {
        let request = GenerateQuestRequest {
            build_prompt: "Build a Fiber checkout for generated lessons".to_string(),
            skill_track: Some("Fiber Builder".to_string()),
            difficulty: Some(Difficulty::Builder),
        };

        let quest = fallback_quest(&request);

        assert_eq!(
            quest.build_objective,
            "Build a Fiber checkout for generated lessons"
        );
        assert_eq!(quest.comprehension_gates.len(), 5);
    }

    #[test]
    fn schema_requires_expected_fields() {
        let schema = quest_json_schema();
        let required = schema
            .pointer("/schema/required")
            .and_then(Value::as_array)
            .unwrap();

        assert!(required.contains(&Value::String("boss_fight".to_string())));
        assert!(required.contains(&Value::String("ckb_fiber_hooks".to_string())));
    }

    fn sample_quest() -> QuestBlueprint {
        QuestBlueprint {
            title: "Receipt Raid".to_string(),
            premise: "A generated app claims it can verify every payment.".to_string(),
            build_objective: "Build a Fiber paywall".to_string(),
            comprehension_gates: vec![
                "Explain the verifier.".to_string(),
                "Debug the unpaid route.".to_string(),
                "Remix the pricing model.".to_string(),
                "Attack receipt replay.".to_string(),
                "Ship with tests.".to_string(),
            ],
            boss_fight: "Patch the replayable receipt.".to_string(),
            reward_logic: "XP per gate, reward after boss.".to_string(),
            ckb_fiber_hooks: vec![
                "CKB proof badge.".to_string(),
                "Fiber bounty payout.".to_string(),
            ],
        }
    }
}
