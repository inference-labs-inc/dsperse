# OpenClaw + PicoClaw Verification Demo (DSperse `openclaw` branch)

This directory is a demo harness that shows:

- LLM Guard scanning (prompt injection, toxicity, PII anonymization)
- an ONNX guard model restricted to a small supported op set
- DSperse slicing + compile + run + prove + verify on the output layer
- a dashboard-style request table for observability

It is intentionally minimal and deployable.

## Visual artifacts

- Dashboard demo page: `openclaw_demo/docs/index.html`
- Architecture diagram: `openclaw_demo/docs/architechure.html`

If GitHub Pages is enabled for the repo, the Pages workflow publishes `openclaw_demo/docs/`.

## Run (local)

1) Set env vars:

- Copy `openclaw_demo/.env.example` to a local `.env` (do not commit it).
- Set `OPENROUTER_API_KEY` if you want real OpenRouter calls.

2) Start the gateway:

```bash
source .venv/bin/activate
ENABLE_DSPERSE_PROOF=1 uvicorn openclaw_demo.gateway.app:app --host 127.0.0.1 --port 8000
```

3) Use endpoints:

- `POST /filter` (runs guard + optional proof)
- `POST /guarded_chat` (runs guard; if allowed, calls OpenRouter using the redacted prompt only)
- `GET /metrics` (Prometheus)
- `GET /events` (JSON rows for table dashboards)

## OpenRouter vs OpenClaw options

This demo keeps both integration options:

### Option A: Direct OpenRouter

Use `POST /guarded_chat` with:

- `OPENROUTER_API_KEY` set
- `OPENCLAW_MODEL` set (default `openrouter/auto`)

No OpenClaw installation is required.

### Option B: OpenClaw integration (external)

If you want to use OpenClaw, configure OpenClaw to call this gateway first (filter), then route allowed requests onward.

OpenRouter's OpenClaw integration guide:

- https://openrouter.ai/docs/guides/openclaw-integration

## PicoClaw (mock-only) verification path

This repo includes a PicoClaw mock-only demo as a git submodule at `openclaw_demo/picoclaw` (pinned to the `verification` branch commit).

If you cloned without submodules, initialize it:

```bash
git submodule update --init --recursive
```

Source repo:

- https://github.com/shirin-shahabi/TestVerifiedpicoclaw

That submodule demo includes:

- a mock OpenRouter backend
- Loki/Promtail JSONL log shipping
- Prometheus scraping
- a Grafana dashboard-style table

