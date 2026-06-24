# VibeQuest Core

Rust backend for VibeQuest. It owns AI quest generation, game progression, evaluation, and the future CKB/Fiber integration layer.

## Run

```bash
cp .env.example .env
cargo run
```

Without `OPENAI_API_KEY`, `/ai/quests/generate` returns a deterministic fallback quest so local demos still work. Set `OPENAI_API_KEY` to enable the OpenAI-compatible Responses API path.

## Environment

| Variable | Required | Default | Purpose |
| --- | --- | --- | --- |
| `APP_ENV` | No | `development` | Environment label returned by `/health`. |
| `PORT` | No | `8080` | HTTP server port. |
| `CORS_ORIGINS` | No | `http://localhost:3000` | Comma-separated frontend origins. Use `*` only for throwaway demos. |
| `OPENAI_API_KEY` | For AI | empty | Enables OpenAI quest generation. |
| `OPENAI_MODEL` | No | `gpt-5.5` | OpenAI model for quest generation. |
| `OPENAI_BASE_URL` | No | `https://share-ai.ckbdev.com` | OpenAI-compatible Responses API gateway. |
| `OPENAI_REASONING_EFFORT` | No | `xhigh` | Reasoning effort sent as `reasoning.effort`. |
| `OPENAI_DISABLE_RESPONSE_STORAGE` | No | `true` | Sends `store: false` to avoid retaining generated responses. |
| `OPENAI_TIMEOUT_SECONDS` | No | `90` | Request timeout for OpenAI calls. |
| `CKB_RPC_URL` | Later | empty | Planned CKB proof adapter endpoint. |
| `FIBER_RPC_URL` | Later | empty | Planned Fiber reward adapter endpoint. |

## Endpoints

- `GET /health` - service status and active AI layer.
- `GET /ready` - production readiness check; returns `503` until required production secrets are present.
- `GET /season` - Season 0 tracks, gates, and product thesis.
- `POST /ai/quests/generate` - turns a vibecoding prompt into a structured quest.

Example:

```bash
curl -X POST http://localhost:8080/ai/quests/generate \
  -H 'content-type: application/json' \
  -d '{
    "build_prompt": "Build a Fiber-powered paid content app with CKB proof receipts",
    "skill_track": "Fiber Builder",
    "difficulty": "builder"
  }'
```

## Checks

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
cargo build --release
```

## Docker

```bash
docker build -t vibequest-core .
docker run --rm -p 8080:8080 --env-file .env vibequest-core
```

## Planned Modules

- OpenAI orchestration for quest design, code explanation, and challenge generation.
- Test-runner workers for debug and no-prompt zones.
- CKB proof adapter for badges, receipts, and skill passport history.
- Fiber reward adapter for hints, bounties, prizes, and creator royalties.
