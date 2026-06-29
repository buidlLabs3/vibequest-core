use axum::{
    Json, Router,
    extract::{Path, State},
    http::{HeaderValue, Method, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use base64::Engine;
use chrono::{DateTime, Utc};
use futures_util::TryStreamExt;
use mongodb::{
    Client as MongoClient, Collection, Database,
    bson::{DateTime as BsonDateTime, Document, doc},
};
use reqwest::{Client, StatusCode as ReqwestStatusCode};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{env, error::Error, sync::Arc, time::Duration};
use thiserror::Error;
use tokio::sync::OnceCell;
use tower_http::{
    cors::{AllowOrigin, CorsLayer},
    trace::TraceLayer,
};
use tracing::warn;
use uuid::Uuid;

const DEFAULT_OPENAI_MODEL: &str = "gpt-5.5";
const DEFAULT_OPENAI_BASE_URL: &str = "https://share-ai.ckbdev.com";
const DEFAULT_OPENAI_REASONING_EFFORT: ReasoningEffort = ReasoningEffort::Minimal;
const SERVERLESS_OPENAI_TIMEOUT_SECONDS: u64 = 12;
const QUICK_QUEST_OUTPUT_TOKENS: u16 = 900;

#[derive(Clone)]
pub struct AppState {
    config: AppConfig,
    openai: OpenAiClient,
    fiber: FiberPayoutClient,
    store: MongoStore,
}

#[derive(Clone)]
struct AppConfig {
    port: u16,
    app_env: String,
    cors_origins: Vec<String>,
    ckb_rpc_url: Option<String>,
    fiber_rpc_url: Option<String>,
    fiber_payout_rpc_url: Option<String>,
    fiber_payout_enabled: bool,
    reward_amount_shannons: u128,
    reward_currency: String,
    mongodb_uri: Option<String>,
    mongodb_database: String,
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

#[derive(Clone)]
struct FiberPayoutClient {
    http: Client,
    rpc_url: Option<String>,
    enabled: bool,
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
    fiber_payout: bool,
    mongodb: bool,
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
    persistence: PersistenceStatus,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct PersistenceStatus {
    saved: bool,
    warning: Option<String>,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
enum QuestSource {
    OpenAi,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct WalletBinding {
    address: String,
    identity: String,
    sign_type: String,
    message: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct ShipRequirements {
    ckb_rpc_ready: bool,
    fiber_rpc_ready: bool,
    can_claim_rewards: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
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

#[derive(Clone, Debug, Deserialize, Serialize)]
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
    #[error("wallet proof must use JoyID")]
    UnsupportedWalletSignature,
    #[error("wallet signature could not be verified against the signer identity")]
    InvalidWalletSignature,
    #[error("OpenAI is not configured. Add OPENAI_API_KEY before generating live quests.")]
    MissingOpenAiKey,
    #[error("AI quest generation is temporarily unavailable. Please regenerate in a moment.")]
    OpenAiTransport(String),
    #[error("AI quest generation is temporarily unavailable. Please regenerate in a moment.")]
    OpenAiStatus {
        status: ReqwestStatusCode,
        body: String,
    },
    #[error("The AI response was incomplete. Please regenerate the quest.")]
    InvalidAiResponse,
    #[error("Quest history is temporarily unavailable because MongoDB is not configured.")]
    DatabaseUnavailable,
    #[error("Quest history is temporarily unavailable. Please refresh in a moment.")]
    Database(String),
    #[error("quest run was not found")]
    QuestNotFound,
    #[error("wallet proof does not own this quest run")]
    WalletMismatch,
    #[error("quest completion evidence is not payout eligible")]
    CompletionNotVerified,
    #[error("Fiber invoice is required before locking a reward claim")]
    MissingFiberInvoice,
    #[error("Fiber payout is not configured on vibequest-core")]
    FiberPayoutUnavailable,
    #[error("Fiber payout failed: {0}")]
    FiberPayout(String),
    #[error("reward claim is already paid or currently paying")]
    RewardAlreadyProcessed,
}

#[derive(Debug, Serialize)]
struct ErrorResponse {
    error: String,
}

impl From<mongodb::error::Error> for ApiError {
    fn from(error: mongodb::error::Error) -> Self {
        Self::Database(error.to_string())
    }
}

#[derive(Clone)]
struct MongoStore {
    uri: Option<String>,
    database_name: String,
    client: Arc<OnceCell<MongoClient>>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct UserDocument {
    #[serde(rename = "_id")]
    id: String,
    address: String,
    wallet: WalletBinding,
    quest_counts: UserQuestCounts,
    created_at: BsonDateTime,
    updated_at: BsonDateTime,
    last_seen_at: BsonDateTime,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
struct UserQuestCounts {
    created: i64,
    completed: i64,
    uncompleted: i64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct QuestRunDocument {
    #[serde(rename = "_id")]
    run_id: String,
    user_address: String,
    build_prompt: String,
    skill_track: String,
    difficulty: String,
    source: QuestSource,
    wallet: WalletBinding,
    quest: QuestBlueprint,
    ship_requirements: ShipRequirements,
    progress: QuestProgress,
    status: QuestRunStatus,
    created_at: BsonDateTime,
    updated_at: BsonDateTime,
    completed_at: Option<BsonDateTime>,
    #[serde(default = "default_reward_snapshot")]
    reward: RewardSnapshot,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct RewardSnapshot {
    amount_shannons: String,
    currency: String,
    sponsor: String,
}

fn default_reward_snapshot() -> RewardSnapshot {
    RewardSnapshot {
        amount_shannons: "0".to_string(),
        currency: "Fibd".to_string(),
        sponsor: "vibequest-core".to_string(),
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct RewardClaimDocument {
    #[serde(rename = "_id")]
    claim_id: String,
    run_id: String,
    user_address: String,
    fiber_invoice: String,
    amount_shannons: String,
    currency: String,
    status: RewardClaimStatus,
    verification: ServerCompletionProof,
    fiber_payment: Option<FiberPaymentReceipt>,
    error: Option<String>,
    created_at: BsonDateTime,
    updated_at: BsonDateTime,
    paid_at: Option<BsonDateTime>,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
enum RewardClaimStatus {
    Pending,
    Verified,
    Paying,
    Paid,
    Failed,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct ServerCompletionProof {
    identity_gate: bool,
    infrastructure_gate: bool,
    verification_gate: bool,
    boss_fight_solved: bool,
    generated_files_verified: bool,
    tests_present: bool,
    proof_present: bool,
    denial_path_present: bool,
    completed_at: BsonDateTime,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct FiberPaymentReceipt {
    payment_hash: Option<String>,
    status: Option<String>,
    fee: Option<String>,
    raw: Value,
}

#[derive(Debug, Deserialize)]
struct CompleteQuestRequest {
    wallet: WalletProof,
    gates: Vec<StoredGateProgress>,
    boss_fight_solved: bool,
    fiber_invoice: String,
}

#[derive(Debug, Serialize)]
struct CompleteQuestResponse {
    run: QuestRunRecord,
    claim: RewardClaimRecord,
}

#[derive(Clone, Debug, Serialize)]
struct RewardClaimRecord {
    claim_id: String,
    run_id: String,
    user_address: String,
    amount_shannons: String,
    currency: String,
    status: RewardClaimStatus,
    fiber_payment: Option<FiberPaymentReceipt>,
    error: Option<String>,
    created_at: DateTime<Utc>,
    updated_at: DateTime<Utc>,
    paid_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Deserialize)]
struct FiberRpcResponse {
    result: Option<Value>,
    error: Option<FiberRpcError>,
}

#[derive(Debug, Deserialize)]
struct FiberRpcError {
    code: Option<i64>,
    message: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct QuestProgress {
    gates: Vec<StoredGateProgress>,
    boss_fight_solved: bool,
    shipped: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct StoredGateProgress {
    id: String,
    name: String,
    description: String,
    #[serde(alias = "isCompleted")]
    is_completed: bool,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
enum QuestRunStatus {
    InProgress,
    Completed,
}

#[derive(Debug, Serialize)]
struct UserQuestHistoryResponse {
    user: Option<UserProfileResponse>,
    stats: UserQuestCounts,
    active_run: Option<QuestRunRecord>,
    runs: Vec<QuestRunRecord>,
    reward_claims: Vec<RewardClaimRecord>,
}

#[derive(Debug, Serialize)]
struct UserProfileResponse {
    address: String,
    quest_counts: UserQuestCounts,
    created_at: DateTime<Utc>,
    updated_at: DateTime<Utc>,
    last_seen_at: DateTime<Utc>,
}

#[derive(Clone, Debug, Serialize)]
struct QuestRunRecord {
    run_id: String,
    user_address: String,
    build_prompt: String,
    skill_track: String,
    difficulty: String,
    source: QuestSource,
    quest: QuestBlueprint,
    ship_requirements: ShipRequirements,
    progress: QuestProgress,
    status: QuestRunStatus,
    created_at: DateTime<Utc>,
    updated_at: DateTime<Utc>,
    completed_at: Option<DateTime<Utc>>,
    reward: RewardSnapshot,
}

#[derive(Debug, Deserialize)]
struct UpdateQuestProgressRequest {
    wallet: WalletProof,
    gates: Option<Vec<StoredGateProgress>>,
    boss_fight_solved: Option<bool>,
    shipped: Option<bool>,
}

impl MongoStore {
    fn from_config(config: &AppConfig) -> Self {
        Self {
            uri: config.mongodb_uri.clone(),
            database_name: config.mongodb_database.clone(),
            client: Arc::new(OnceCell::new()),
        }
    }

    #[cfg(test)]
    fn disabled() -> Self {
        Self {
            uri: None,
            database_name: "vibequest".to_string(),
            client: Arc::new(OnceCell::new()),
        }
    }

    fn is_configured(&self) -> bool {
        self.uri.is_some()
    }

    async fn is_available(&self) -> bool {
        if !self.is_configured() {
            return false;
        }

        tokio::time::timeout(Duration::from_secs(4), async {
            let Ok(database) = self.database().await else {
                return false;
            };
            let mut command = Document::new();
            command.insert("ping", 1);

            database.run_command(command).await.is_ok()
        })
        .await
        .unwrap_or(false)
    }

    async fn database(&self) -> Result<Database, ApiError> {
        let uri = self.uri.clone().ok_or(ApiError::DatabaseUnavailable)?;
        let client = self
            .client
            .get_or_try_init(move || async move {
                MongoClient::with_uri_str(&uri)
                    .await
                    .map_err(|error| ApiError::Database(error.to_string()))
            })
            .await?;

        Ok(client.database(&self.database_name))
    }

    async fn users(&self) -> Result<Collection<UserDocument>, ApiError> {
        Ok(self.database().await?.collection("users"))
    }

    async fn quest_runs(&self) -> Result<Collection<QuestRunDocument>, ApiError> {
        Ok(self.database().await?.collection("quest_runs"))
    }

    async fn reward_claims(&self) -> Result<Collection<RewardClaimDocument>, ApiError> {
        Ok(self.database().await?.collection("reward_claims"))
    }

    async fn record_generated_quest(
        &self,
        request: &GenerateQuestRequest,
        response: &GenerateQuestResponse,
        reward_amount_shannons: u128,
        reward_currency: &str,
    ) -> Result<(), ApiError> {
        if !self.is_configured() {
            return Ok(());
        }

        self.upsert_user(&response.wallet).await?;

        let now = BsonDateTime::now();
        let run = QuestRunDocument {
            run_id: response.run_id.to_string(),
            user_address: response.wallet.address.clone(),
            build_prompt: request.build_prompt.trim().to_string(),
            skill_track: request
                .skill_track
                .as_deref()
                .unwrap_or("CKB + Fiber Builder")
                .trim()
                .to_string(),
            difficulty: difficulty_label(request.difficulty.as_ref()).to_string(),
            source: response.source,
            wallet: response.wallet.clone(),
            quest: response.quest.clone(),
            ship_requirements: response.ship_requirements.clone(),
            progress: initial_quest_progress(
                response.ship_requirements.ckb_rpc_ready
                    && response.ship_requirements.fiber_rpc_ready,
            ),
            status: QuestRunStatus::InProgress,
            created_at: now,
            updated_at: now,
            completed_at: None,
            reward: RewardSnapshot {
                amount_shannons: reward_amount_shannons.to_string(),
                currency: reward_currency.to_string(),
                sponsor: "vibequest-core".to_string(),
            },
        };

        self.quest_runs().await?.insert_one(&run).await?;
        self.refresh_user_counts(&response.wallet.address).await?;
        Ok(())
    }

    async fn upsert_user(&self, wallet: &WalletBinding) -> Result<(), ApiError> {
        let users = self.users().await?;
        let address = wallet.address.trim().to_string();
        let now = BsonDateTime::now();

        if users.find_one(doc! { "_id": &address }).await?.is_some() {
            users
                .update_one(
                    doc! { "_id": &address },
                    doc! {
                        "$set": {
                            "address": &address,
                            "wallet": wallet_document(wallet),
                            "updated_at": now,
                            "last_seen_at": now,
                        }
                    },
                )
                .await?;
            return Ok(());
        }

        users
            .insert_one(UserDocument {
                id: address.clone(),
                address,
                wallet: wallet.clone(),
                quest_counts: UserQuestCounts::default(),
                created_at: now,
                updated_at: now,
                last_seen_at: now,
            })
            .await?;
        Ok(())
    }

    async fn user_history(&self, address: &str) -> Result<UserQuestHistoryResponse, ApiError> {
        let address = address.trim();
        if address.is_empty() {
            return Err(ApiError::MissingWalletAddress);
        }

        let users = self.users().await?;
        let runs = self.quest_runs().await?;
        let user = users.find_one(doc! { "_id": address }).await?;
        let stats = self.counts_for_user(address).await?;
        let claims_cursor = self
            .reward_claims()
            .await?
            .find(doc! { "user_address": address })
            .sort(doc! { "updated_at": -1 })
            .limit(40)
            .await?;
        let reward_claims = claims_cursor
            .try_collect::<Vec<_>>()
            .await?
            .into_iter()
            .map(RewardClaimRecord::from)
            .collect::<Vec<_>>();
        let cursor = runs
            .find(doc! { "user_address": address })
            .sort(doc! { "updated_at": -1 })
            .limit(40)
            .await?;
        let documents = cursor.try_collect::<Vec<_>>().await?;
        let records = documents
            .into_iter()
            .map(QuestRunRecord::from)
            .collect::<Vec<_>>();
        let active_run = records
            .iter()
            .find(|run| run.status != QuestRunStatus::Completed)
            .cloned()
            .or_else(|| records.first().cloned());

        Ok(UserQuestHistoryResponse {
            user: user.map(UserProfileResponse::from),
            stats,
            active_run,
            runs: records,
            reward_claims,
        })
    }

    async fn get_run(&self, run_id: &str) -> Result<QuestRunDocument, ApiError> {
        self.quest_runs()
            .await?
            .find_one(doc! { "_id": run_id })
            .await?
            .ok_or(ApiError::QuestNotFound)
    }

    async fn update_progress(
        &self,
        run_id: &str,
        request: UpdateQuestProgressRequest,
    ) -> Result<QuestRunRecord, ApiError> {
        validate_wallet_proof(&request.wallet)?;

        let mut run = self.get_run(run_id).await?;
        if run.user_address != request.wallet.address.trim() {
            return Err(ApiError::WalletMismatch);
        }

        if let Some(gates) = request.gates {
            run.progress.gates = gates;
        }
        if let Some(boss_fight_solved) = request.boss_fight_solved {
            run.progress.boss_fight_solved = boss_fight_solved;
        }
        if let Some(shipped) = request.shipped {
            if shipped {
                return Err(ApiError::CompletionNotVerified);
            }
            run.progress.shipped = false;
        }

        run.status = status_for_progress(&run.progress);
        run.updated_at = BsonDateTime::now();
        if run.status == QuestRunStatus::Completed && run.completed_at.is_none() {
            run.completed_at = Some(run.updated_at);
        }
        if run.status != QuestRunStatus::Completed {
            run.completed_at = None;
        }

        self.quest_runs()
            .await?
            .replace_one(doc! { "_id": run_id }, &run)
            .await?;
        self.refresh_user_counts(&run.user_address).await?;

        Ok(run.into())
    }

    async fn complete_quest(
        &self,
        run_id: &str,
        request: CompleteQuestRequest,
        reward_amount_shannons: u128,
        reward_currency: &str,
        fiber: &FiberPayoutClient,
    ) -> Result<CompleteQuestResponse, ApiError> {
        validate_wallet_proof(&request.wallet)?;
        if request.fiber_invoice.trim().is_empty() {
            return Err(ApiError::MissingFiberInvoice);
        }

        let mut run = self.get_run(run_id).await?;
        if run.user_address != request.wallet.address.trim() {
            return Err(ApiError::WalletMismatch);
        }

        run.progress.gates = request.gates;
        run.progress.boss_fight_solved = request.boss_fight_solved;
        let proof = server_completion_proof(&run)?;
        run.progress.shipped = true;
        run.status = QuestRunStatus::Completed;
        run.updated_at = BsonDateTime::now();
        if run.completed_at.is_none() {
            run.completed_at = Some(run.updated_at);
        }
        run.reward = RewardSnapshot {
            amount_shannons: reward_amount_shannons.to_string(),
            currency: reward_currency.to_string(),
            sponsor: "vibequest-core".to_string(),
        };

        let claim_id = format!("{}:{}", run.run_id, run.user_address);
        let claims = self.reward_claims().await?;
        let existing_claim = claims.find_one(doc! { "_id": &claim_id }).await?;
        if existing_claim.is_some_and(|existing| {
            matches!(
                existing.status,
                RewardClaimStatus::Paid | RewardClaimStatus::Paying
            )
        }) {
            return Err(ApiError::RewardAlreadyProcessed);
        }

        let now = BsonDateTime::now();
        let mut claim = RewardClaimDocument {
            claim_id: claim_id.clone(),
            run_id: run.run_id.clone(),
            user_address: run.user_address.clone(),
            fiber_invoice: request.fiber_invoice.trim().to_string(),
            amount_shannons: reward_amount_shannons.to_string(),
            currency: reward_currency.to_string(),
            status: if fiber.enabled {
                RewardClaimStatus::Paying
            } else {
                RewardClaimStatus::Verified
            },
            verification: proof,
            fiber_payment: None,
            error: None,
            created_at: now,
            updated_at: now,
            paid_at: None,
        };

        self.quest_runs()
            .await?
            .replace_one(doc! { "_id": run_id }, &run)
            .await?;

        claims
            .replace_one(doc! { "_id": &claim_id }, &claim)
            .upsert(true)
            .await?;

        match fiber.pay_invoice(&claim.fiber_invoice).await {
            Ok(receipt) => {
                claim.status = if fiber.enabled {
                    RewardClaimStatus::Paid
                } else {
                    RewardClaimStatus::Verified
                };
                claim.fiber_payment = receipt;
                claim.error = None;
                claim.paid_at = if fiber.enabled {
                    Some(BsonDateTime::now())
                } else {
                    None
                };
            }
            Err(error) => {
                claim.status = RewardClaimStatus::Failed;
                claim.error = Some(error.to_string());
            }
        }
        claim.updated_at = BsonDateTime::now();

        claims
            .replace_one(doc! { "_id": &claim_id }, &claim)
            .await?;
        self.refresh_user_counts(&run.user_address).await?;

        Ok(CompleteQuestResponse {
            run: run.into(),
            claim: claim.into(),
        })
    }

    async fn counts_for_user(&self, address: &str) -> Result<UserQuestCounts, ApiError> {
        let runs = self.quest_runs().await?;
        let created = runs
            .count_documents(doc! { "user_address": address })
            .await? as i64;
        let completed = runs
            .count_documents(doc! { "user_address": address, "status": "completed" })
            .await? as i64;

        Ok(UserQuestCounts {
            created,
            completed,
            uncompleted: created.saturating_sub(completed),
        })
    }

    async fn refresh_user_counts(&self, address: &str) -> Result<(), ApiError> {
        let counts = self.counts_for_user(address).await?;
        self.users()
            .await?
            .update_one(
                doc! { "_id": address },
                doc! {
                    "$set": {
                        "quest_counts.created": counts.created,
                        "quest_counts.completed": counts.completed,
                        "quest_counts.uncompleted": counts.uncompleted,
                        "updated_at": BsonDateTime::now(),
                    }
                },
            )
            .await?;
        Ok(())
    }
}

impl From<UserDocument> for UserProfileResponse {
    fn from(user: UserDocument) -> Self {
        Self {
            address: user.address,
            quest_counts: user.quest_counts,
            created_at: bson_datetime_to_utc(user.created_at),
            updated_at: bson_datetime_to_utc(user.updated_at),
            last_seen_at: bson_datetime_to_utc(user.last_seen_at),
        }
    }
}

impl From<QuestRunDocument> for QuestRunRecord {
    fn from(run: QuestRunDocument) -> Self {
        Self {
            run_id: run.run_id,
            user_address: run.user_address,
            build_prompt: run.build_prompt,
            skill_track: run.skill_track,
            difficulty: run.difficulty,
            source: run.source,
            quest: run.quest,
            ship_requirements: run.ship_requirements,
            progress: run.progress,
            status: run.status,
            created_at: bson_datetime_to_utc(run.created_at),
            updated_at: bson_datetime_to_utc(run.updated_at),
            completed_at: run.completed_at.map(bson_datetime_to_utc),
            reward: run.reward,
        }
    }
}

impl From<RewardClaimDocument> for RewardClaimRecord {
    fn from(claim: RewardClaimDocument) -> Self {
        Self {
            claim_id: claim.claim_id,
            run_id: claim.run_id,
            user_address: claim.user_address,
            amount_shannons: claim.amount_shannons,
            currency: claim.currency,
            status: claim.status,
            fiber_payment: claim.fiber_payment,
            error: claim.error,
            created_at: bson_datetime_to_utc(claim.created_at),
            updated_at: bson_datetime_to_utc(claim.updated_at),
            paid_at: claim.paid_at.map(bson_datetime_to_utc),
        }
    }
}

fn initial_quest_progress(infrastructure_ready: bool) -> QuestProgress {
    QuestProgress {
        gates: vec![
            StoredGateProgress {
                id: "identity".to_string(),
                name: "Wallet Proof".to_string(),
                description: "A signed JoyID passkey proof is bound to this quest session."
                    .to_string(),
                is_completed: true,
            },
            StoredGateProgress {
                id: "infrastructure".to_string(),
                name: "Backend Readiness".to_string(),
                description:
                    "vibequest-core reports OpenAI, CKB RPC, Fiber RPC, and MongoDB ready."
                        .to_string(),
                is_completed: infrastructure_ready,
            },
            StoredGateProgress {
                id: "verification".to_string(),
                name: "Generated Workspace Checks".to_string(),
                description: "Generated files pass proof, test, and denial-path checks."
                    .to_string(),
                is_completed: false,
            },
        ],
        boss_fight_solved: false,
        shipped: false,
    }
}

fn server_completion_proof(run: &QuestRunDocument) -> Result<ServerCompletionProof, ApiError> {
    let identity_gate = run
        .progress
        .gates
        .iter()
        .any(|gate| gate.id == "identity" && gate.is_completed);
    let infrastructure_gate = run
        .progress
        .gates
        .iter()
        .any(|gate| gate.id == "infrastructure" && gate.is_completed);
    let verification_gate = run
        .progress
        .gates
        .iter()
        .any(|gate| gate.id == "verification" && gate.is_completed);
    let workspace = run
        .quest
        .workbench_files
        .iter()
        .map(|file| {
            format!(
                "{}
{}",
                file.path, file.content
            )
            .to_lowercase()
        })
        .collect::<Vec<_>>()
        .join(
            "
",
        );
    let tests_present = workspace.contains("test(") || workspace.contains("#[test]");
    let proof_present = workspace.contains("fiber")
        && workspace.contains("ckb")
        && (workspace.contains("proof") || workspace.contains("receipt"));
    let denial_path_present = ["reject", "block", "false", "unpaid"]
        .iter()
        .any(|needle| workspace.contains(needle));
    let generated_files_verified = tests_present && proof_present && denial_path_present;

    if !(identity_gate
        && infrastructure_gate
        && verification_gate
        && run.progress.boss_fight_solved
        && generated_files_verified)
    {
        return Err(ApiError::CompletionNotVerified);
    }

    Ok(ServerCompletionProof {
        identity_gate,
        infrastructure_gate,
        verification_gate,
        boss_fight_solved: run.progress.boss_fight_solved,
        generated_files_verified,
        tests_present,
        proof_present,
        denial_path_present,
        completed_at: BsonDateTime::now(),
    })
}

fn status_for_progress(progress: &QuestProgress) -> QuestRunStatus {
    if progress.shipped
        && progress.boss_fight_solved
        && progress.gates.iter().all(|gate| gate.is_completed)
    {
        QuestRunStatus::Completed
    } else {
        QuestRunStatus::InProgress
    }
}

fn wallet_document(wallet: &WalletBinding) -> Document {
    doc! {
        "address": &wallet.address,
        "identity": &wallet.identity,
        "sign_type": &wallet.sign_type,
        "message": &wallet.message,
    }
}

fn bson_datetime_to_utc(value: BsonDateTime) -> DateTime<Utc> {
    DateTime::<Utc>::from_timestamp_millis(value.timestamp_millis()).unwrap_or_else(Utc::now)
}

fn difficulty_label(difficulty: Option<&Difficulty>) -> &'static str {
    match difficulty {
        Some(Difficulty::Novice) => "novice",
        Some(Difficulty::Boss) => "boss",
        _ => "builder",
    }
}

pub fn app_state() -> Arc<AppState> {
    dotenvy::dotenv().ok();

    let config = AppConfig::from_env();
    let store = MongoStore::from_config(&config);
    let state = Arc::new(AppState {
        openai: OpenAiClient::from_env(),
        fiber: FiberPayoutClient::from_config(&config),
        store,
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
        .route("/users/{address}/quests", get(list_user_quests))
        .route("/quests/{run_id}", get(get_quest_run))
        .route("/quests/{run_id}/progress", post(update_quest_progress))
        .route("/quests/{run_id}/complete", post(complete_quest))
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
            fiber_payout_rpc_url: optional_env("FIBER_PAYOUT_RPC_URL"),
            fiber_payout_enabled: parse_bool_env("FIBER_PAYOUT_ENABLED", false),
            reward_amount_shannons: optional_env("VIBEQUEST_REWARD_SHANNONS")
                .and_then(|value| value.parse::<u128>().ok())
                .unwrap_or(400),
            reward_currency: optional_env("VIBEQUEST_REWARD_CURRENCY")
                .unwrap_or_else(|| "Fibd".to_string()),
            mongodb_uri: optional_env("MONGODB_URI"),
            mongodb_database: optional_env("MONGODB_DATABASE")
                .unwrap_or_else(|| "vibequest".to_string()),
        }
    }
}

impl OpenAiClient {
    fn from_env() -> Self {
        let timeout_seconds = env::var("OPENAI_TIMEOUT_SECONDS")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(SERVERLESS_OPENAI_TIMEOUT_SECONDS);

        Self {
            http: Client::builder()
                .user_agent("VibeQuestCore/1.0 (+https://github.com/buidlLabs3/vibequest-core)")
                .build()
                .expect("OpenAI HTTP client should build"),
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
                "effort": self.reasoning_effort.serverless_safe()
            },
            "max_output_tokens": QUICK_QUEST_OUTPUT_TOKENS,
            "store": !self.disable_response_storage,
            "text": {
                "format": quest_json_schema()
            }
        });
        let timeout = self.timeout;

        let response = self
            .http
            .post(format!("{}/responses", self.base_url))
            .bearer_auth(api_key)
            .timeout(timeout)
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

impl FiberPayoutClient {
    fn from_config(config: &AppConfig) -> Self {
        Self {
            http: Client::new(),
            rpc_url: config.fiber_payout_rpc_url.clone(),
            enabled: config.fiber_payout_enabled,
            timeout: Duration::from_secs(30),
        }
    }

    fn is_ready(&self) -> bool {
        self.enabled && self.rpc_url.is_some()
    }

    async fn pay_invoice(&self, invoice: &str) -> Result<Option<FiberPaymentReceipt>, ApiError> {
        if !self.enabled {
            return Ok(Some(FiberPaymentReceipt {
                payment_hash: None,
                status: Some("verified-no-payout".to_string()),
                fee: None,
                raw: serde_json::json!({
                    "mode": "payout-disabled",
                    "invoice_bound": !invoice.trim().is_empty()
                }),
            }));
        }

        let rpc_url = self
            .rpc_url
            .as_ref()
            .ok_or(ApiError::FiberPayoutUnavailable)?;
        let body = serde_json::json!({
            "id": "vibequest-payout",
            "jsonrpc": "2.0",
            "method": "send_payment",
            "params": [{
                "invoice": invoice.trim(),
            }]
        });
        let response = self
            .http
            .post(rpc_url)
            .timeout(self.timeout)
            .json(&body)
            .send()
            .await
            .map_err(|error| ApiError::FiberPayout(error.to_string()))?;
        let status = response.status();
        let text = response
            .text()
            .await
            .map_err(|error| ApiError::FiberPayout(error.to_string()))?;
        if !status.is_success() {
            return Err(ApiError::FiberPayout(truncate_error_body(&text)));
        }

        let decoded = serde_json::from_str::<FiberRpcResponse>(&text)
            .map_err(|_| ApiError::FiberPayout("invalid Fiber RPC response".to_string()))?;
        if let Some(error) = decoded.error {
            return Err(ApiError::FiberPayout(format!(
                "{}{}",
                error
                    .code
                    .map(|code| format!("{code}: "))
                    .unwrap_or_default(),
                error.message
            )));
        }
        let result = decoded
            .result
            .ok_or_else(|| ApiError::FiberPayout("missing Fiber RPC result".to_string()))?;

        Ok(Some(FiberPaymentReceipt {
            payment_hash: result
                .get("payment_hash")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned),
            status: result
                .get("status")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned),
            fee: result.get("fee").map(|value| match value {
                Value::String(value) => value.clone(),
                other => other.to_string(),
            }),
            raw: result,
        }))
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

    fn serverless_safe(self) -> Self {
        match self {
            Self::High | Self::Xhigh | Self::Medium | Self::Low => Self::Minimal,
            value => value,
        }
    }
}

async fn health(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    let integrations = integration_status(&state).await;
    let missing = missing_integrations(&state, &integrations);

    Json(HealthResponse {
        service: "vibequest-core",
        status: "ok",
        environment: state.config.app_env.clone(),
        ai_layer: AiLayer::OpenAi,
        integrations,
        missing,
        timestamp: Utc::now(),
    })
}

async fn ready(State(state): State<Arc<AppState>>) -> (StatusCode, Json<ReadyResponse>) {
    let integrations = integration_status(&state).await;
    let missing = missing_integrations(&state, &integrations);

    let is_ready = missing.is_empty();
    let status = if is_ready {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };

    (
        status,
        Json(ReadyResponse {
            ready: is_ready,
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
    let quest = state
        .openai
        .generate_quest(&request)
        .await
        .and_then(compact_quest_blueprint)?;
    let source = QuestSource::OpenAi;

    let mut response = GenerateQuestResponse {
        run_id,
        source,
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
        persistence: PersistenceStatus {
            saved: false,
            warning: None,
        },
    };

    match state
        .store
        .record_generated_quest(
            &request,
            &response,
            state.config.reward_amount_shannons,
            &state.config.reward_currency,
        )
        .await
    {
        Ok(()) => {
            response.persistence.saved = true;
        }
        Err(error @ (ApiError::Database(_) | ApiError::DatabaseUnavailable)) => {
            warn!(%error, "quest generated but persistence is degraded");
            response.persistence.warning = Some(
                "AI quest generated, but cloud save is temporarily unavailable. You can practice now; reward claim unlocks after persistence recovers."
                    .to_string(),
            );
        }
        Err(error) => return Err(error),
    }

    Ok(Json(response))
}

async fn list_user_quests(
    State(state): State<Arc<AppState>>,
    Path(address): Path<String>,
) -> Result<Json<UserQuestHistoryResponse>, ApiError> {
    Ok(Json(state.store.user_history(&address).await?))
}

async fn get_quest_run(
    State(state): State<Arc<AppState>>,
    Path(run_id): Path<String>,
) -> Result<Json<QuestRunRecord>, ApiError> {
    Ok(Json(state.store.get_run(&run_id).await?.into()))
}

async fn update_quest_progress(
    State(state): State<Arc<AppState>>,
    Path(run_id): Path<String>,
    Json(request): Json<UpdateQuestProgressRequest>,
) -> Result<Json<QuestRunRecord>, ApiError> {
    Ok(Json(state.store.update_progress(&run_id, request).await?))
}

async fn complete_quest(
    State(state): State<Arc<AppState>>,
    Path(run_id): Path<String>,
    Json(request): Json<CompleteQuestRequest>,
) -> Result<Json<CompleteQuestResponse>, ApiError> {
    Ok(Json(
        state
            .store
            .complete_quest(
                &run_id,
                request,
                state.config.reward_amount_shannons,
                &state.config.reward_currency,
                &state.fiber,
            )
            .await?,
    ))
}

async fn integration_status(state: &AppState) -> IntegrationStatus {
    IntegrationStatus {
        openai: state.openai.api_key.is_some(),
        ckb_rpc: state.config.ckb_rpc_url.is_some(),
        fiber_rpc: state.config.fiber_rpc_url.is_some(),
        fiber_payout: state.fiber.is_ready(),
        mongodb: state.store.is_available().await,
    }
}

fn missing_integrations(state: &AppState, integrations: &IntegrationStatus) -> Vec<&'static str> {
    let mut missing = Vec::new();

    if !integrations.openai {
        missing.push("OPENAI_API_KEY");
    }

    if !integrations.ckb_rpc {
        missing.push("CKB_RPC_URL");
    }

    if !integrations.fiber_rpc {
        missing.push("FIBER_RPC_URL");
    }

    if state.store.is_configured() {
        if !integrations.mongodb {
            missing.push("MONGODB_CONNECTION");
        }
    } else {
        missing.push("MONGODB_URI");
    }

    if state.config.fiber_payout_enabled && state.config.fiber_payout_rpc_url.is_none() {
        missing.push("FIBER_PAYOUT_RPC_URL");
    }

    missing
}

fn warn_missing_integrations(state: &AppState) {
    let integrations = IntegrationStatus {
        openai: state.openai.api_key.is_some(),
        ckb_rpc: state.config.ckb_rpc_url.is_some(),
        fiber_rpc: state.config.fiber_rpc_url.is_some(),
        fiber_payout: state.fiber.is_ready(),
        mongodb: state.store.is_configured(),
    };
    let missing = missing_integrations(state, &integrations);

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

    verify_joyid_wallet_proof(wallet)
}

#[derive(Debug, Deserialize)]
struct JoyIdSignaturePayload {
    signature: String,
    alg: Value,
    message: String,
}

#[derive(Debug, Deserialize)]
struct JoyIdIdentityPayload {
    #[serde(rename = "keyType")]
    key_type: String,
    #[serde(rename = "publicKey")]
    public_key: String,
}

fn is_joyid_sign_type(sign_type: &str) -> bool {
    let normalized: String = sign_type
        .trim()
        .chars()
        .filter(|character| character.is_ascii_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect();

    matches!(normalized.as_str(), "joyid" | "signersigntypejoyid")
}

fn verify_joyid_wallet_proof(wallet: &WalletProof) -> Result<(), ApiError> {
    if !is_joyid_sign_type(&wallet.signature.sign_type) {
        return Err(ApiError::UnsupportedWalletSignature);
    }

    let signature_payload =
        serde_json::from_str::<JoyIdSignaturePayload>(&wallet.signature.signature)
            .map_err(|_| ApiError::InvalidWalletSignature)?;
    let identity_payload = serde_json::from_str::<JoyIdIdentityPayload>(&wallet.signature.identity)
        .map_err(|_| ApiError::InvalidWalletSignature)?;

    if signature_payload.signature.trim().is_empty()
        || signature_payload.message.trim().is_empty()
        || signature_payload.alg.is_null()
        || !is_joyid_key_type(&identity_payload.key_type)
        || !is_hex_public_key(&identity_payload.public_key)
    {
        return Err(ApiError::InvalidWalletSignature);
    }

    let Some(signed_challenge) = joyid_signed_challenge(&signature_payload.message) else {
        return Err(ApiError::InvalidWalletSignature);
    };

    if signed_challenge != wallet.message {
        return Err(ApiError::InvalidWalletSignature);
    }

    Ok(())
}

fn is_joyid_key_type(value: &str) -> bool {
    matches!(
        value.trim(),
        "main_key" | "sub_key" | "main_session_key" | "sub_session_key"
    )
}

fn is_hex_public_key(value: &str) -> bool {
    let trimmed = value.trim().trim_start_matches("0x");

    !trimmed.is_empty()
        && trimmed.len().is_multiple_of(2)
        && trimmed
            .chars()
            .all(|character| character.is_ascii_hexdigit())
}

fn joyid_signed_challenge(message: &str) -> Option<String> {
    let bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(message.as_bytes())
        .ok()?;
    let client_data_start = bytes
        .windows(2)
        .position(|window| window == b"{\"")
        .unwrap_or(0);
    let client_data = std::str::from_utf8(&bytes[client_data_start..]).ok()?;

    if let Ok(parsed) = serde_json::from_str::<Value>(client_data) {
        let encoded_challenge = parsed.get("challenge")?.as_str()?;
        let challenge_bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .decode(encoded_challenge.as_bytes())
            .ok()?;

        return String::from_utf8(challenge_bytes).ok();
    }

    String::from_utf8(bytes).ok()
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

fn compact_quest_blueprint(mut quest: QuestBlueprint) -> Result<QuestBlueprint, ApiError> {
    if quest.comprehension_gates.len() > 3 {
        quest.comprehension_gates.truncate(3);
    }
    while quest.comprehension_gates.len() < 3 {
        quest.comprehension_gates.push(
            "Explain the trust boundary, run the denial test, then ship the badge.".to_string(),
        );
    }

    if quest.ckb_fiber_hooks.len() > 2 {
        quest.ckb_fiber_hooks.truncate(2);
    }
    if quest.workbench_files.len() > 2 {
        quest.workbench_files.truncate(2);
    }

    for file in &mut quest.workbench_files {
        file.content = compact_file_content(&file.content, 80);
    }

    if quest.workbench_files.is_empty()
        || quest
            .workbench_files
            .iter()
            .any(|file| file.path.trim().is_empty() || file.content.trim().is_empty())
    {
        return Err(ApiError::InvalidAiResponse);
    }

    Ok(quest)
}

fn compact_file_content(content: &str, max_lines: usize) -> String {
    let lines = content.lines().collect::<Vec<_>>();
    if lines.len() <= max_lines {
        return content.to_string();
    }

    let mut compacted = lines[..max_lines].join("\n");
    compacted.push_str("\n// VibeQuest clipped this file to keep the browser workbench fast.\n");
    compacted
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
    let nonce = Uuid::new_v4();
    format!(
        r#"You are VibeQuest's AI quest compiler for a vibecoding learning product.

Create one original practical coding quest for this exact learner request:
"{build_prompt}"

Skill track: {track}
Difficulty: {difficulty:?}
Variation seed: {nonce}

Hard requirements:
- The quest must be clearly specific to the learner request. Do not reuse a generic paywall verifier unless the request is actually about a paywall.
- The code must be concrete and runnable-looking, not placeholder pseudocode.
- Use exactly 3 comprehension gates: Explain, Verify, Ship.
- Use exactly 2 workbench files: one short implementation file and one short test file.
- Keep each file under 80 lines.
- Include at least one CKB/Fiber concept relevant to the prompt: cell, script, xUDT, proof receipt, Fiber invoice, HTLC, channel state, or payout split.
- Include a denial/failure path test, such as invalid proof, unpaid access, replayed receipt, wrong witness, wrong asset, or expired timelock.
- Include a unique run/request-specific constant or fixture so repeated generations are visibly different.

Return only the JSON object required by the schema."#
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
                    "minItems": 3,
                    "maxItems": 3,
                    "items": {
                        "type": "string"
                    },
                    "description": "Exactly three short gates: explain, verify, ship."
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
                    "minItems": 1,
                    "maxItems": 2,
                    "items": {
                        "type": "string"
                    },
                    "description": "CKB/Fiber integration hooks for the quest."
                },
                "workbench_files": {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": 2,
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
                                "description": "Small but concrete code file content, preferably under 80 lines."
                            }
                        }
                    },
                    "description": "Exactly two compact generated files: implementation and test."
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
            | ApiError::InvalidWalletSignature
            | ApiError::MissingFiberInvoice => StatusCode::BAD_REQUEST,
            ApiError::MissingOpenAiKey
            | ApiError::DatabaseUnavailable
            | ApiError::FiberPayoutUnavailable => StatusCode::SERVICE_UNAVAILABLE,
            ApiError::QuestNotFound => StatusCode::NOT_FOUND,
            ApiError::WalletMismatch => StatusCode::FORBIDDEN,
            ApiError::CompletionNotVerified | ApiError::RewardAlreadyProcessed => {
                StatusCode::CONFLICT
            }
            ApiError::OpenAiTransport(_)
            | ApiError::OpenAiStatus { .. }
            | ApiError::InvalidAiResponse => StatusCode::BAD_GATEWAY,
            ApiError::Database(_) => StatusCode::SERVICE_UNAVAILABLE,
            ApiError::FiberPayout(_) => StatusCode::INTERNAL_SERVER_ERROR,
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
    #[test]
    fn parses_openai_output_text() {
        let quest = sample_quest();
        let response = OpenAiResponse {
            output_text: Some(serde_json::to_string(&quest).unwrap()),
            output: None,
        };

        let parsed = parse_openai_quest_response(response).unwrap();

        assert_eq!(parsed.title, "Receipt Raid");
        assert_eq!(parsed.comprehension_gates.len(), 3);
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
        let wallet = joyid_wallet_fixture();

        validate_wallet_proof(&wallet).unwrap();

        let missing_signature = WalletProof {
            signature: WalletSignature {
                signature: String::new(),
                identity: "0xidentity".to_string(),
                sign_type: "JoyId".to_string(),
            },
            ..wallet
        };

        assert!(matches!(
            validate_wallet_proof(&missing_signature),
            Err(ApiError::MissingWalletSignature)
        ));
    }

    #[test]
    fn wallet_proof_accepts_connector_sign_type_variants() {
        for sign_type in [
            "JoyId",
            "SignerSignType.JoyId",
            "joy_id",
            "joy-id",
            "SignerSignType::JoyId",
        ] {
            let mut wallet = joyid_wallet_fixture();
            wallet.signature.sign_type = sign_type.to_string();

            validate_wallet_proof(&wallet).unwrap();
        }
    }

    #[test]
    fn wallet_proof_rejects_non_joyid_sign_type() {
        let mut wallet = joyid_wallet_fixture();
        wallet.signature.sign_type = "EthereumPersonalSign".to_string();

        assert!(matches!(
            validate_wallet_proof(&wallet),
            Err(ApiError::UnsupportedWalletSignature)
        ));
    }
    #[test]
    fn wallet_proof_rejects_tampered_signature_message() {
        let wallet = WalletProof {
            message: "VibeQuest wallet proof for a different signer".to_string(),
            ..joyid_wallet_fixture()
        };

        assert!(matches!(
            validate_wallet_proof(&wallet),
            Err(ApiError::InvalidWalletProofMessage | ApiError::InvalidWalletSignature)
        ));
    }

    #[test]
    fn wallet_proof_rejects_mismatched_joyid_payload_message() {
        let mut wallet = joyid_wallet_fixture();
        wallet.signature.signature = serde_json::json!({
            "signature": "joyid-passkey-signature-fixture",
            "alg": "ES256",
            "message": joyid_webauthn_message("VibeQuest wallet proof\nAddress: another-wallet")
        })
        .to_string();

        assert!(matches!(
            validate_wallet_proof(&wallet),
            Err(ApiError::InvalidWalletSignature)
        ));
    }

    #[test]
    fn wallet_proof_accepts_session_key_payload_message() {
        let mut wallet = joyid_wallet_fixture();
        wallet.signature.identity = serde_json::json!({
            "keyType": "main_session_key",
            "publicKey": format!("02{}", "22".repeat(32))
        })
        .to_string();
        wallet.signature.signature = serde_json::json!({
            "signature": "joyid-session-signature-fixture",
            "alg": "RS256",
            "message": base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(wallet.message.as_bytes())
        })
        .to_string();

        validate_wallet_proof(&wallet).unwrap();
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
        assert_eq!(
            ReasoningEffort::Xhigh.serverless_safe(),
            ReasoningEffort::Minimal
        );
        assert_eq!(
            ReasoningEffort::High.serverless_safe(),
            ReasoningEffort::Minimal
        );
        assert_eq!(
            ReasoningEffort::Medium.serverless_safe(),
            ReasoningEffort::Minimal
        );
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
                fiber_payout_rpc_url: None,
                fiber_payout_enabled: false,
                reward_amount_shannons: 400,
                reward_currency: "Fibd".to_string(),
                mongodb_uri: None,
                mongodb_database: "vibequest".to_string(),
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
            fiber: FiberPayoutClient {
                http: Client::new(),
                rpc_url: None,
                enabled: false,
                timeout: Duration::from_secs(1),
            },
            store: MongoStore::disabled(),
        };

        let integrations = IntegrationStatus {
            openai: true,
            ckb_rpc: false,
            fiber_rpc: false,
            fiber_payout: false,
            mongodb: false,
        };

        assert_eq!(
            missing_integrations(&state, &integrations),
            vec!["CKB_RPC_URL", "FIBER_RPC_URL", "MONGODB_URI"]
        );
    }

    #[test]
    fn server_completion_proof_rejects_unverified_runs() {
        let mut run = quest_run_fixture();
        run.progress.boss_fight_solved = false;

        assert!(matches!(
            server_completion_proof(&run),
            Err(ApiError::CompletionNotVerified)
        ));
    }

    #[test]
    fn server_completion_proof_accepts_verified_runs() {
        let run = quest_run_fixture();
        let proof = server_completion_proof(&run).unwrap();

        assert!(proof.identity_gate);
        assert!(proof.infrastructure_gate);
        assert!(proof.verification_gate);
        assert!(proof.generated_files_verified);
        assert!(proof.tests_present);
        assert!(proof.proof_present);
        assert!(proof.denial_path_present);
    }

    fn joyid_wallet_fixture() -> WalletProof {
        let address = "ckt1qjoyidvibequestwalletproof000000000000000000000000000".to_string();
        let message = format!(
            "VibeQuest wallet proof\nAddress: {address}\nIssued: 2026-06-24T00:00:00.000Z\nPurpose: bind generated quest runs, proof notes, and reward claims to this signer."
        );
        let identity = serde_json::json!({
            "keyType": "main_key",
            "publicKey": format!("02{}", "11".repeat(32))
        });
        let signature = serde_json::json!({
            "signature": "joyid-passkey-signature-fixture",
            "alg": "ES256",
            "message": joyid_webauthn_message(&message)
        });

        WalletProof {
            address,
            message,
            signature: WalletSignature {
                signature: signature.to_string(),
                identity: identity.to_string(),
                sign_type: "JoyId".to_string(),
            },
        }
    }

    fn joyid_webauthn_message(challenge: &str) -> String {
        let encoded_challenge =
            base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(challenge.as_bytes());
        let client_data = serde_json::json!({
            "type": "webauthn.get",
            "challenge": encoded_challenge,
            "origin": "https://testnet.joyid.dev"
        })
        .to_string();
        let mut payload = vec![0_u8; 37];
        payload.extend(client_data.as_bytes());

        base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(payload)
    }

    fn quest_run_fixture() -> QuestRunDocument {
        let quest = QuestBlueprint {
            title: "AI Fiber Receipt Run".to_string(),
            premise: "A learner validates a Fiber payment receipt against a CKB proof badge.".to_string(),
            build_objective: "Build a Fiber paid-content app with CKB proof receipts".to_string(),
            comprehension_gates: vec![
                "Explain the JoyID, Fiber receipt, and CKB proof trust boundary.".to_string(),
                "Verify that unpaid reads and replayed receipts are rejected.".to_string(),
                "Ship only after the badge and payout claim are defended.".to_string(),
            ],
            boss_fight: "A reader replays a receipt from another run. Identify the missing run binding.".to_string(),
            reward_logic: "CKB stores the proof badge and Fiber invoice-bound reward claim.".to_string(),
            ckb_fiber_hooks: vec![
                "CKB proof hash binds the quest receipt.".to_string(),
                "Fiber invoice binds the payout claim.".to_string(),
            ],
            workbench_files: vec![
                WorkbenchFile {
                    path: "src/receiptVerifier.ts".to_string(),
                    language: "ts".to_string(),
                    content: "export function canRead(receipt?: { runId: string; fiber: string; ckbProof: string }) { return Boolean(receipt && receipt.runId === 'vq-test' && receipt.fiber && receipt.ckbProof.startsWith('0x')); }".to_string(),
                },
                WorkbenchFile {
                    path: "tests/receiptVerifier.test.ts".to_string(),
                    language: "test.ts".to_string(),
                    content: "test('blocks unpaid reads', () => expect(canRead()).toBe(false)); test('rejects replayed receipts', () => expect(canRead({ runId: 'old', fiber: 'preimage', ckbProof: '0xabc' })).toBe(false));".to_string(),
                },
            ],
        };
        let now = BsonDateTime::now();
        QuestRunDocument {
            run_id: Uuid::nil().to_string(),
            user_address: "ckt1qjoyidvibequestwalletproof000000000000000000000000000".to_string(),
            build_prompt: "Build a Fiber paid-content app with CKB proof receipts".to_string(),
            skill_track: "Fiber Builder".to_string(),
            difficulty: "builder".to_string(),
            source: QuestSource::OpenAi,
            wallet: WalletBinding {
                address: "ckt1qjoyidvibequestwalletproof000000000000000000000000000".to_string(),
                identity: "identity".to_string(),
                sign_type: "JoyId".to_string(),
                message: "VibeQuest wallet proof".to_string(),
            },
            quest,
            ship_requirements: ShipRequirements {
                ckb_rpc_ready: true,
                fiber_rpc_ready: true,
                can_claim_rewards: true,
            },
            progress: QuestProgress {
                gates: vec![
                    StoredGateProgress {
                        id: "identity".to_string(),
                        name: "Wallet Proof".to_string(),
                        description: "signed".to_string(),
                        is_completed: true,
                    },
                    StoredGateProgress {
                        id: "infrastructure".to_string(),
                        name: "Backend Readiness".to_string(),
                        description: "ready".to_string(),
                        is_completed: true,
                    },
                    StoredGateProgress {
                        id: "verification".to_string(),
                        name: "Generated Workspace Checks".to_string(),
                        description: "verified".to_string(),
                        is_completed: true,
                    },
                ],
                boss_fight_solved: true,
                shipped: false,
            },
            status: QuestRunStatus::InProgress,
            created_at: now,
            updated_at: now,
            completed_at: None,
            reward: RewardSnapshot {
                amount_shannons: "400".to_string(),
                currency: "Fibd".to_string(),
                sponsor: "vibequest-core".to_string(),
            },
        }
    }

    fn sample_quest() -> QuestBlueprint {
        QuestBlueprint {
            title: "Receipt Raid".to_string(),
            premise: "A generated app claims it can verify every payment.".to_string(),
            build_objective: "Build a Fiber paywall".to_string(),
            comprehension_gates: vec![
                "Explain the verifier.".to_string(),
                "Verify the denial test.".to_string(),
                "Ship with badge proof.".to_string(),
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
