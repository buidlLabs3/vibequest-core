# VibeQuest Core

Rust backend for VibeQuest. It owns AI quest generation, wallet proof verification, game progression, and CKB/Fiber readiness gates.

## Run

```bash
cp .env.example .env
cargo run
```

`/ai/quests/generate` requires `OPENAI_API_KEY` and a verified CKB `CkbSecp256k1` wallet proof. `/ready` returns `503` until OpenAI, CKB RPC, and Fiber RPC are configured.

## Environment

| Variable | Required | Default | Purpose |
| --- | --- | --- | --- |
| `APP_ENV` | No | `development` | Environment label returned by `/health`. |
| `PORT` | No | `8080` | HTTP server port. |
| `CORS_ORIGINS` | No | `http://localhost:3000` | Comma-separated frontend origins. Use `*` only for throwaway demos. |
| `OPENAI_API_KEY` | Yes | empty | Enables OpenAI quest generation. |
| `OPENAI_MODEL` | No | `gpt-5.5` | OpenAI model for quest generation. |
| `OPENAI_BASE_URL` | No | `https://share-ai.ckbdev.com` | OpenAI-compatible Responses API gateway. |
| `OPENAI_REASONING_EFFORT` | No | `xhigh` | Reasoning effort sent as `reasoning.effort`. |
| `OPENAI_DISABLE_RESPONSE_STORAGE` | No | `true` | Sends `store: false` to avoid retaining generated responses. |
| `OPENAI_TIMEOUT_SECONDS` | No | `180` | Request timeout for slower high-reasoning OpenAI calls. |
| `CKB_RPC_URL` | Yes | empty | CKB RPC endpoint used by proof receipt and reward-claim readiness gates. |
| `FIBER_RPC_URL` | Yes | empty | Fiber RPC endpoint used by payment, hint fee, and reward-claim readiness gates. Use a public endpoint for cloud deploys. |

## Endpoints

- `GET /health` - service status, integration readiness, and missing configuration.
- `GET /ready` - production readiness check; returns `503` until OpenAI, CKB RPC, and Fiber RPC are configured.
- `GET /season` - Season 0 tracks, gates, and product thesis.
- `POST /ai/quests/generate` - verifies a CKB wallet proof, then turns a vibecoding prompt into a structured quest.

Example:

```bash
curl -X POST http://localhost:8080/ai/quests/generate \
  -H 'content-type: application/json' \
  -d '{
    "build_prompt": "Build a Fiber-powered paid content app with CKB proof receipts",
    "skill_track": "Fiber Builder",
    "difficulty": "builder",
    "wallet": {
      "address": "ckt1...",
      "message": "VibeQuest wallet proof\nAddress: ckt1...\nIssued: 2026-06-24T00:00:00.000Z\nPurpose: bind generated quest runs, proof notes, and reward claims to this signer.",
      "signature": {
        "signature": "0x...",
        "identity": "0x...",
        "sign_type": "CkbSecp256k1"
      }
    }
  }'
```

## Checks

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
cargo build --release
```

## Deploy to Vercel

This backend ships with the official Vercel Rust runtime through `api/index.rs` and `vercel.json`.

```bash
vercel link --project vibequest-core
vercel env add OPENAI_API_KEY production
vercel env add OPENAI_MODEL production
vercel env add OPENAI_BASE_URL production
vercel env add OPENAI_REASONING_EFFORT production
vercel env add OPENAI_DISABLE_RESPONSE_STORAGE production
vercel env add OPENAI_TIMEOUT_SECONDS production
vercel env add CKB_RPC_URL production
vercel env add FIBER_RPC_URL production
vercel --prod
```

`FIBER_RPC_URL=http://127.0.0.1:8227` is only valid for local development. Cloud deployments need a public Fiber RPC endpoint, such as the current testnet public node used by the deployment.

## Docker

```bash
docker build -t vibequest-core .
docker run --rm -p 8080:8080 --env-file .env vibequest-core
```

## Runtime Modules

- OpenAI orchestration for quest design, code explanation, and challenge generation.
- Test-runner workers for debug and no-prompt zones.
- CKB proof adapter for badges, receipts, and skill passport history.
- Fiber reward adapter for hints, bounties, prizes, and creator royalties.
