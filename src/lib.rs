use axum::{
    Json, Router,
    extract::State,
    http::{HeaderValue, Method, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use bech32::Hrp;
use chrono::{DateTime, Utc};
use reqwest::{Client, StatusCode as ReqwestStatusCode};
use secp256k1::{
    Message, Secp256k1,
    ecdsa::{RecoverableSignature, RecoveryId},
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{env, error::Error, sync::Arc, time::Duration};
use thiserror::Error;
use tower_http::{
    cors::{AllowOrigin, CorsLayer},
    trace::TraceLayer,
};
use tracing::warn;
use uuid::Uuid;

const DEFAULT_OPENAI_MODEL: &str = "gpt-5.5";
const DEFAULT_OPENAI_BASE_URL: &str = "https://share-ai.ckbdev.com";
const DEFAULT_OPENAI_REASONING_EFFORT: ReasoningEffort = ReasoningEffort::Xhigh;
const SECP256K1_BLAKE160_CODE_HASH: [u8; 32] = [
    0x9b, 0xd7, 0xe0, 0x6f, 0x3e, 0xcf, 0x4b, 0xe0, 0xf2, 0xfc, 0xd2, 0x18, 0x8b, 0x23, 0xf1, 0xb9,
    0xfc, 0xc8, 0x8e, 0x5d, 0x4b, 0x65, 0xa8, 0x63, 0x7b, 0x17, 0x72, 0x3b, 0xbd, 0xa3, 0xcc, 0xe8,
];

#[derive(Clone)]
pub struct AppState {
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
    reasoning_effort: ReasoningEffort,
    disable_response_storage: bool,
    timeout: Duration,
}

#[derive(Debug, Serialize)]
struct HealthResponse {
    service: &'static str,
    status: &'static str,
    environment: String,
    ai_layer: AiLayer,
    integrations: IntegrationStatus,
    missing: Vec<&'static str>,
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
    wallet: WalletProof,
}

#[derive(Debug, Deserialize, Serialize)]
struct WalletProof {
    address: String,
    message: String,
    signature: WalletSignature,
}

#[derive(Debug, Deserialize, Serialize)]
struct WalletSignature {
    signature: String,
    identity: String,
    sign_type: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
enum Difficulty {
    Novice,
    Builder,
    Boss,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
enum ReasoningEffort {
    None,
    Minimal,
    Low,
    Medium,
    High,
    Xhigh,
}

#[derive(Debug, Serialize)]
struct GenerateQuestResponse {
    run_id: Uuid,
    source: QuestSource,
    wallet: WalletBinding,
    quest: QuestBlueprint,
    ship_requirements: ShipRequirements,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "kebab-case")]
enum QuestSource {
    OpenAi,
}

#[derive(Debug, Serialize)]
struct WalletBinding {
    address: String,
    identity: String,
    sign_type: String,
    message: String,
}

#[derive(Debug, Serialize)]
struct ShipRequirements {
    ckb_rpc_ready: bool,
    fiber_rpc_ready: bool,
    can_claim_rewards: bool,
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
    workbench_files: Vec<WorkbenchFile>,
}

#[derive(Debug, Deserialize, Serialize)]
struct WorkbenchFile {
    path: String,
    language: String,
    content: String,
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
    #[error("wallet address is required")]
    MissingWalletAddress,
    #[error("wallet signature is required")]
    MissingWalletSignature,
    #[error("wallet proof message must include VibeQuest")]
    InvalidWalletProofMessage,
    #[error("wallet proof must use CkbSecp256k1")]
    UnsupportedWalletSignature,
    #[error("wallet signature could not be verified against the signer identity")]
    InvalidWalletSignature,
    #[error("OPENAI_API_KEY is required")]
    MissingOpenAiKey,
    #[error("openai request failed: {0}")]
    OpenAiTransport(String),
    #[error("openai gateway returned {status}: {body}")]
    OpenAiStatus {
        status: ReqwestStatusCode,
        body: String,
    },
    #[error("openai response did not contain valid quest json")]
    InvalidAiResponse,
}

#[derive(Debug, Serialize)]
struct ErrorResponse {
    error: String,
}

pub fn app_state() -> Arc<AppState> {
    dotenvy::dotenv().ok();

    let config = AppConfig::from_env();
    let state = Arc::new(AppState {
        openai: OpenAiClient::from_env(),
        config,
    });

    warn_missing_integrations(&state);

    state
}

pub fn app_port() -> u16 {
    AppConfig::from_env().port
}

pub fn build_router(state: Arc<AppState>) -> Router {
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
            .unwrap_or(180);

        Self {
            http: Client::new(),
            api_key: optional_env("OPENAI_API_KEY"),
            model: optional_env("OPENAI_MODEL")
                .or_else(|| optional_env("MODEL"))
                .unwrap_or_else(|| DEFAULT_OPENAI_MODEL.to_string()),
            base_url: env::var("OPENAI_BASE_URL")
                .unwrap_or_else(|_| DEFAULT_OPENAI_BASE_URL.to_string())
                .trim_end_matches('/')
                .to_string(),
            reasoning_effort: optional_env("OPENAI_REASONING_EFFORT")
                .or_else(|| optional_env("MODEL_REASONING_EFFORT"))
                .and_then(|value| ReasoningEffort::parse(&value))
                .unwrap_or(DEFAULT_OPENAI_REASONING_EFFORT),
            disable_response_storage: parse_bool_env("OPENAI_DISABLE_RESPONSE_STORAGE", true),
            timeout: Duration::from_secs(timeout_seconds),
        }
    }

    async fn generate_quest(
        &self,
        request: &GenerateQuestRequest,
    ) -> Result<QuestBlueprint, ApiError> {
        let Some(api_key) = self.api_key.as_ref() else {
            return Err(ApiError::MissingOpenAiKey);
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
            "reasoning": {
                "effort": self.reasoning_effort
            },
            "store": !self.disable_response_storage,
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
            .await
            .map_err(|error| {
                let detail = if error.is_timeout() {
                    format!(
                        "{error}; source: {}",
                        error
                            .source()
                            .map(ToString::to_string)
                            .unwrap_or_else(|| "request timed out".to_string())
                    )
                } else {
                    error.to_string()
                };

                ApiError::OpenAiTransport(detail)
            })?;

        let status = response.status();
        let response_body = response
            .text()
            .await
            .map_err(|error| ApiError::OpenAiTransport(error.to_string()))?;

        if !status.is_success() {
            return Err(ApiError::OpenAiStatus {
                status,
                body: truncate_error_body(&response_body),
            });
        }

        let response = serde_json::from_str::<OpenAiResponse>(&response_body)
            .map_err(|_| ApiError::InvalidAiResponse)?;

        parse_openai_quest_response(response)
    }
}

impl ReasoningEffort {
    fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "none" => Some(Self::None),
            "minimal" => Some(Self::Minimal),
            "low" => Some(Self::Low),
            "medium" => Some(Self::Medium),
            "high" => Some(Self::High),
            "xhigh" => Some(Self::Xhigh),
            _ => None,
        }
    }
}

async fn health(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    Json(HealthResponse {
        service: "vibequest-core",
        status: "ok",
        environment: state.config.app_env.clone(),
        ai_layer: AiLayer::OpenAi,
        integrations: integration_status(&state),
        missing: missing_integrations(&state),
        timestamp: Utc::now(),
    })
}

async fn ready(State(state): State<Arc<AppState>>) -> (StatusCode, Json<ReadyResponse>) {
    let missing = missing_integrations(&state);

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

    validate_wallet_proof(&request.wallet)?;

    let run_id = Uuid::new_v4();
    let quest = state.openai.generate_quest(&request).await?;

    Ok(Json(GenerateQuestResponse {
        run_id,
        source: QuestSource::OpenAi,
        wallet: WalletBinding {
            address: request.wallet.address.trim().to_string(),
            identity: request.wallet.signature.identity.trim().to_string(),
            sign_type: request.wallet.signature.sign_type.trim().to_string(),
            message: request.wallet.message.trim().to_string(),
        },
        quest,
        ship_requirements: ShipRequirements {
            ckb_rpc_ready: state.config.ckb_rpc_url.is_some(),
            fiber_rpc_ready: state.config.fiber_rpc_url.is_some(),
            can_claim_rewards: state.config.ckb_rpc_url.is_some()
                && state.config.fiber_rpc_url.is_some(),
        },
    }))
}

fn integration_status(state: &AppState) -> IntegrationStatus {
    IntegrationStatus {
        openai: state.openai.api_key.is_some(),
        ckb_rpc: state.config.ckb_rpc_url.is_some(),
        fiber_rpc: state.config.fiber_rpc_url.is_some(),
    }
}

fn missing_integrations(state: &AppState) -> Vec<&'static str> {
    let mut missing = Vec::new();

    if state.openai.api_key.is_none() {
        missing.push("OPENAI_API_KEY");
    }

    if state.config.ckb_rpc_url.is_none() {
        missing.push("CKB_RPC_URL");
    }

    if state.config.fiber_rpc_url.is_none() {
        missing.push("FIBER_RPC_URL");
    }

    missing
}

fn warn_missing_integrations(state: &AppState) {
    let missing = missing_integrations(state);

    if !missing.is_empty() {
        warn!(
            missing = missing.join(", "),
            "vibequest-core is not fully configured"
        );
    }
}

fn validate_wallet_proof(wallet: &WalletProof) -> Result<(), ApiError> {
    if wallet.address.trim().is_empty() {
        return Err(ApiError::MissingWalletAddress);
    }

    if wallet.signature.signature.trim().is_empty()
        || wallet.signature.identity.trim().is_empty()
        || wallet.signature.sign_type.trim().is_empty()
    {
        return Err(ApiError::MissingWalletSignature);
    }

    if !wallet.message.contains("VibeQuest") || !wallet.message.contains(wallet.address.trim()) {
        return Err(ApiError::InvalidWalletProofMessage);
    }

    verify_ckb_wallet_signature(wallet)
}

fn verify_ckb_wallet_signature(wallet: &WalletProof) -> Result<(), ApiError> {
    if wallet.signature.sign_type.trim() != "CkbSecp256k1" {
        return Err(ApiError::UnsupportedWalletSignature);
    }

    let signature = decode_hex(&wallet.signature.signature)?;
    let identity = decode_hex(&wallet.signature.identity)?;

    if signature.len() != 65 || !matches!(identity.len(), 33 | 65) {
        return Err(ApiError::InvalidWalletSignature);
    }

    let mut compact_signature = [0_u8; 64];
    compact_signature.copy_from_slice(&signature[..64]);

    let recovery_id = parse_recovery_id(signature[64])?;
    let recoverable_signature = RecoverableSignature::from_compact(&compact_signature, recovery_id)
        .map_err(|_| ApiError::InvalidWalletSignature)?;
    let message = Message::from_digest(message_hash_ckb_secp256k1(&wallet.message));
    let secp = Secp256k1::verification_only();
    let public_key = secp
        .recover_ecdsa(message, &recoverable_signature)
        .map_err(|_| ApiError::InvalidWalletSignature)?;

    let compressed = public_key.serialize();
    let uncompressed = public_key.serialize_uncompressed();

    if identity.as_slice() != compressed && identity.as_slice() != uncompressed {
        return Err(ApiError::InvalidWalletSignature);
    }

    if wallet_address_matches_public_key(&wallet.address, &compressed)? {
        return Ok(());
    }

    Err(ApiError::InvalidWalletSignature)
}

fn parse_recovery_id(value: u8) -> Result<RecoveryId, ApiError> {
    let normalized = match value {
        0..=3 => value,
        27..=30 => value - 27,
        _ => return Err(ApiError::InvalidWalletSignature),
    };

    RecoveryId::try_from(i32::from(normalized)).map_err(|_| ApiError::InvalidWalletSignature)
}

fn decode_hex(value: &str) -> Result<Vec<u8>, ApiError> {
    hex::decode(value.trim().trim_start_matches("0x")).map_err(|_| ApiError::InvalidWalletSignature)
}

fn message_hash_ckb_secp256k1(message: &str) -> [u8; 32] {
    let payload = format!("Nervos Message:{message}");
    let digest = blake2b_simd::Params::new()
        .hash_length(32)
        .personal(b"ckb-default-hash")
        .hash(payload.as_bytes());
    let mut bytes = [0_u8; 32];
    bytes.copy_from_slice(digest.as_bytes());
    bytes
}

fn wallet_address_matches_public_key(
    address: &str,
    public_key: &[u8; 33],
) -> Result<bool, ApiError> {
    let lock_args = decode_ckb_secp256k1_lock_args(address)?;
    Ok(lock_args == blake160(public_key))
}

fn decode_ckb_secp256k1_lock_args(address: &str) -> Result<[u8; 20], ApiError> {
    let (hrp, data) = decode_ckb_address(address)?;

    if hrp.as_str() != "ckt" && hrp.as_str() != "ckb" {
        return Err(ApiError::InvalidWalletSignature);
    }

    let Some((&format_type, payload)) = data.split_first() else {
        return Err(ApiError::InvalidWalletSignature);
    };

    match format_type {
        0x00 => decode_full_secp256k1_payload(payload),
        0x01 => decode_short_secp256k1_payload(payload),
        _ => Err(ApiError::InvalidWalletSignature),
    }
}

fn decode_ckb_address(address: &str) -> Result<(Hrp, Vec<u8>), ApiError> {
    bech32::decode(address).map_err(|_| ApiError::InvalidWalletSignature)
}

fn decode_short_secp256k1_payload(payload: &[u8]) -> Result<[u8; 20], ApiError> {
    if payload.len() != 21 || payload[0] != 0 {
        return Err(ApiError::InvalidWalletSignature);
    }

    let mut args = [0_u8; 20];
    args.copy_from_slice(&payload[1..]);
    Ok(args)
}

fn decode_full_secp256k1_payload(payload: &[u8]) -> Result<[u8; 20], ApiError> {
    if payload.len() != 53 {
        return Err(ApiError::InvalidWalletSignature);
    }

    let (code_hash, rest) = payload.split_at(32);
    let Some((&hash_type, args)) = rest.split_first() else {
        return Err(ApiError::InvalidWalletSignature);
    };

    if code_hash != SECP256K1_BLAKE160_CODE_HASH || hash_type != 0x01 || args.len() != 20 {
        return Err(ApiError::InvalidWalletSignature);
    }

    let mut lock_args = [0_u8; 20];
    lock_args.copy_from_slice(args);
    Ok(lock_args)
}

fn blake160(public_key: &[u8; 33]) -> [u8; 20] {
    let digest = blake2b_simd::Params::new()
        .hash_length(32)
        .personal(b"ckb-default-hash")
        .hash(public_key);
    let mut bytes = [0_u8; 20];
    bytes.copy_from_slice(&digest.as_bytes()[..20]);
    bytes
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

fn parse_bool_env(name: &str, default: bool) -> bool {
    optional_env(name)
        .and_then(|value| match value.to_ascii_lowercase().as_str() {
            "true" | "1" | "yes" | "on" => Some(true),
            "false" | "0" | "no" | "off" => Some(false),
            _ => None,
        })
        .unwrap_or(default)
}

fn truncate_error_body(body: &str) -> String {
    const MAX_ERROR_BODY_CHARS: usize = 700;

    let trimmed = body.trim();
    if trimmed.chars().count() <= MAX_ERROR_BODY_CHARS {
        return trimmed.to_string();
    }

    let mut truncated = trimmed
        .chars()
        .take(MAX_ERROR_BODY_CHARS)
        .collect::<String>();
    truncated.push_str("...");
    truncated
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

Make the quest playful, practical, and focused on proving understanding of generated code. Use a crisp builder, security, or network metaphor instead of fantasy creature mascots. The quest must force the learner to explain, debug, test, attack, remix, and ship the generated app. Include CKB proof and Fiber reward hooks when relevant.

Return realistic workbench files for the learner to inspect and edit. The files should be small enough for a browser editor but concrete enough to expose trust boundaries, tests, and payment/proof logic."#
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
                "ckb_fiber_hooks",
                "workbench_files"
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
                },
                "workbench_files": {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": 4,
                    "items": {
                        "type": "object",
                        "additionalProperties": false,
                        "required": ["path", "language", "content"],
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Project-relative file path."
                            },
                            "language": {
                                "type": "string",
                                "description": "Short language id such as ts, rs, or test.ts."
                            },
                            "content": {
                                "type": "string",
                                "description": "Small but concrete code file content."
                            }
                        }
                    },
                    "description": "Generated workbench files the learner must inspect and improve."
                }
            }
        }
    })
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = match self {
            ApiError::InvalidPrompt
            | ApiError::MissingWalletAddress
            | ApiError::MissingWalletSignature
            | ApiError::InvalidWalletProofMessage
            | ApiError::UnsupportedWalletSignature
            | ApiError::InvalidWalletSignature => StatusCode::BAD_REQUEST,
            ApiError::MissingOpenAiKey => StatusCode::SERVICE_UNAVAILABLE,
            ApiError::OpenAiTransport(_)
            | ApiError::OpenAiStatus { .. }
            | ApiError::InvalidAiResponse => StatusCode::BAD_GATEWAY,
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

#[cfg(test)]
mod tests {
    use super::*;
    use bech32::Bech32m;

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
    fn wallet_proof_requires_real_signature_fields() {
        let wallet = signed_wallet_fixture();

        validate_wallet_proof(&wallet).unwrap();

        let missing_signature = WalletProof {
            signature: WalletSignature {
                signature: String::new(),
                identity: "0xidentity".to_string(),
                sign_type: "CkbSecp256k1".to_string(),
            },
            ..wallet
        };

        assert!(matches!(
            validate_wallet_proof(&missing_signature),
            Err(ApiError::MissingWalletSignature)
        ));
    }

    #[test]
    fn wallet_proof_rejects_tampered_signature_message() {
        let wallet = WalletProof {
            message: "VibeQuest wallet proof for a different signer".to_string(),
            ..signed_wallet_fixture()
        };

        assert!(matches!(
            validate_wallet_proof(&wallet),
            Err(ApiError::InvalidWalletProofMessage | ApiError::InvalidWalletSignature)
        ));
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
        assert!(required.contains(&Value::String("workbench_files".to_string())));
    }

    #[test]
    fn parses_provider_config_values() {
        assert_eq!(
            ReasoningEffort::parse("xhigh"),
            Some(ReasoningEffort::Xhigh)
        );
        assert_eq!(
            ReasoningEffort::parse(" HIGH "),
            Some(ReasoningEffort::High)
        );
        assert_eq!(ReasoningEffort::parse("maximum"), None);
    }

    #[test]
    fn missing_integrations_include_ckb_and_fiber() {
        let state = AppState {
            config: AppConfig {
                port: 8080,
                app_env: "test".to_string(),
                cors_origins: vec!["http://localhost:3000".to_string()],
                ckb_rpc_url: None,
                fiber_rpc_url: None,
            },
            openai: OpenAiClient {
                http: Client::new(),
                api_key: Some("key".to_string()),
                model: DEFAULT_OPENAI_MODEL.to_string(),
                base_url: DEFAULT_OPENAI_BASE_URL.to_string(),
                reasoning_effort: DEFAULT_OPENAI_REASONING_EFFORT,
                disable_response_storage: true,
                timeout: Duration::from_secs(1),
            },
        };

        assert_eq!(
            missing_integrations(&state),
            vec!["CKB_RPC_URL", "FIBER_RPC_URL"]
        );
    }

    fn signed_wallet_fixture() -> WalletProof {
        let secret_key = secp256k1::SecretKey::from_byte_array([7_u8; 32]).unwrap();
        let secp = Secp256k1::new();
        let public_key = secp256k1::PublicKey::from_secret_key(&secp, &secret_key);
        let address = full_testnet_address_from_public_key(&public_key.serialize());
        let message = format!(
            "VibeQuest wallet proof\nAddress: {address}\nIssued: 2026-06-24T00:00:00.000Z\nPurpose: bind generated quest runs, proof notes, and reward claims to this signer."
        );
        let digest = Message::from_digest(message_hash_ckb_secp256k1(&message));
        let signature = secp.sign_ecdsa_recoverable(digest, &secret_key);
        let (recovery_id, compact_signature) = signature.serialize_compact();
        let mut signature_bytes = compact_signature.to_vec();
        signature_bytes.push(i32::from(recovery_id) as u8);

        WalletProof {
            address,
            message,
            signature: WalletSignature {
                signature: format!("0x{}", hex::encode(signature_bytes)),
                identity: format!("0x{}", hex::encode(public_key.serialize())),
                sign_type: "CkbSecp256k1".to_string(),
            },
        }
    }

    fn full_testnet_address_from_public_key(public_key: &[u8; 33]) -> String {
        let mut payload = Vec::with_capacity(54);
        payload.push(0x00);
        payload.extend(SECP256K1_BLAKE160_CODE_HASH);
        payload.push(0x01);
        payload.extend(blake160(public_key));

        bech32::encode::<Bech32m>(Hrp::parse_unchecked("ckt"), &payload).unwrap()
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
            workbench_files: vec![
                WorkbenchFile {
                    path: "app/api/unlock/route.ts".to_string(),
                    language: "ts".to_string(),
                    content: "export async function POST() {}".to_string(),
                },
                WorkbenchFile {
                    path: "tests/unlock.test.ts".to_string(),
                    language: "ts".to_string(),
                    content: "test('blocks unpaid reads', () => {})".to_string(),
                },
            ],
        }
    }
}
