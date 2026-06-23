# VibeQuest Core

Rust backend for VibeQuest. It owns AI quest generation, game progression, evaluation, and the future CKB/Fiber integration layer.

## Run

```bash
cp .env.example .env
cargo run
```

Without `OPENAI_API_KEY`, `/ai/quests/generate` returns a deterministic fallback quest so local demos still work. Set `OPENAI_API_KEY` to enable the OpenAI Responses API path.

## Endpoints

- `GET /health` - service status and active AI layer.
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

## Planned Modules

- OpenAI orchestration for quest design, code explanation, and challenge generation.
- Test-runner workers for debug and no-prompt zones.
- CKB proof adapter for badges, receipts, and skill passport history.
- Fiber reward adapter for hints, bounties, prizes, and creator royalties.
