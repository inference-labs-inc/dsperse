from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from llm_guard.input_scanners.anonymize import Anonymize
from llm_guard.input_scanners.prompt_injection import PromptInjection
from llm_guard.input_scanners.toxicity import Toxicity
from llm_guard.vault import Vault
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel

from openclaw_demo.gateway.openrouter_client import call_openrouter_chat
from openclaw_demo.gateway.proof_runner import run_dsperse_proof_flow

load_dotenv()

APP_ROOT = Path(__file__).resolve().parent.parent
ONNX_MODEL_PATH = APP_ROOT / "models" / "guard_filter.onnx"
DSPERSE_REPO = APP_ROOT.parent
DSPERSE_SLICES_DIR = APP_ROOT / "models" / "guard_filter" / "slices"
DSPERSE_INPUT_FILE = APP_ROOT / "models" / "guard_filter" / "input.json"
ACCESS_LOG = APP_ROOT / "data" / "guard_access_log.jsonl"
ENABLE_PROOF = os.getenv("ENABLE_DSPERSE_PROOF", "1") == "1"
TARGET_OUTPUT_LAYER = int(os.getenv("DSPERSE_OUTPUT_LAYER_INDEX", "1"))

if not ONNX_MODEL_PATH.exists():
    raise RuntimeError(f"ONNX model not found: {ONNX_MODEL_PATH}")

ort_session = ort.InferenceSession(str(ONNX_MODEL_PATH), providers=["CPUExecutionProvider"])

vault = Vault()
scanner_prompt_injection = PromptInjection(threshold=0.85, use_onnx=True)
scanner_toxicity = Toxicity(threshold=0.8, use_onnx=True)
scanner_anonymize = Anonymize(vault=vault, use_onnx=True)

requests_total = Counter("guard_requests_total", "Total guard requests", ["decision"])
proof_total = Counter("guard_proof_total", "Proof runs by status", ["status"])
block_reason_total = Counter("guard_block_reason_total", "Blocked requests by reason", ["reason"])
inference_seconds = Histogram("guard_inference_seconds", "Filter model inference latency")

app = FastAPI(title="DSperse OpenClaw Guard Demo")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
    allow_origin_regex=r"^https://.*\\.github\\.io$",
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_request_seq = 0


class GuardRequest(BaseModel):
    prompt: str
    user_id: str
    source: str = "openclaw"


class GuardedChatRequest(BaseModel):
    prompt: str
    user_id: str
    source: str = "openclaw"
    system: str | None = None


def _next_request_id() -> str:
    global _request_seq
    _request_seq += 1
    return f"req_{_request_seq:06d}"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def vectorize(prompt: str, pi_score: float, toxicity_score: float) -> np.ndarray:
    length = min(len(prompt), 2000) / 2000.0
    digits = sum(c.isdigit() for c in prompt) / max(len(prompt), 1)
    at_symbols = prompt.count("@") / max(len(prompt), 1)
    secret_keywords = sum(
        kw in prompt.lower()
        for kw in ["password", "token", "api_key", "ssn", "account", "routing", "address"]
    ) / 7.0
    return np.array(
        [[length, digits, at_symbols, secret_keywords, pi_score, toxicity_score, 1.0, 0.0]],
        dtype=np.float32,
    )


def write_log(entry: dict[str, Any]) -> None:
    ACCESS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with ACCESS_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def evaluate_prompt(payload: GuardRequest) -> tuple[dict[str, Any], bool]:
    request_id = _next_request_id()
    ts = _utc_now_iso()
    prompt = payload.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    _, pi_valid, pi_score = scanner_prompt_injection.scan(prompt)
    _, tox_valid, tox_score = scanner_toxicity.scan(prompt)
    redacted_prompt, _, _ = scanner_anonymize.scan(prompt)

    feature_vec = vectorize(redacted_prompt, pi_score, tox_score)
    with inference_seconds.time():
        allow_score = float(
            ort_session.run(["allow_score"], {"features": feature_vec})[0].reshape(-1)[0]
        )

    allowed = bool(pi_valid and tox_valid)
    decision = "allow" if allowed else "block"
    requests_total.labels(decision=decision).inc()
    if not allowed:
        reason = "prompt_injection" if not pi_valid else "toxicity"
        block_reason_total.labels(reason=reason).inc()

    proof_info: dict[str, Any] = {"enabled": ENABLE_PROOF, "success": False, "run_dir": ""}
    proof_verified = False
    proof_hash = ""
    if ENABLE_PROOF:
        result = run_dsperse_proof_flow(
            dsperse_repo=DSPERSE_REPO,
            slices_dir=DSPERSE_SLICES_DIR,
            input_file=DSPERSE_INPUT_FILE,
            layer_index=TARGET_OUTPUT_LAYER,
        )
        proof_verified = bool(result.verified)
        proof_hash = result.proof_hash_sha256
        proof_info = {
            "enabled": True,
            "success": result.success,
            "run_dir": result.run_dir,
            "verified": proof_verified,
            "proof_hash_sha256": proof_hash,
        }
        proof_total.labels(status="ok" if result.success else "fail").inc()

    block_reason = ""
    if decision == "block":
        block_reason = "prompt_injection" if not pi_valid else "toxicity"
    detection_type = block_reason if block_reason else "none"

    write_log(
        {
            "timestamp": ts,
            "request_id": request_id,
            "user_id": payload.user_id,
            "source": payload.source,
            "decision": decision,
            "block_reason": block_reason,
            "detection_type": detection_type,
            "allow_score": allow_score,
            "prompt_injection_score": pi_score,
            "toxicity_score": tox_score,
            "proof_verified": proof_verified,
            "proof_hash_sha256": proof_hash,
            "proof": proof_info,
        }
    )

    return (
        {
            "timestamp": ts,
            "request_id": request_id,
            "decision": decision,
            "block_reason": block_reason,
            "detection_type": detection_type,
            "allow_score": allow_score,
            "redacted_prompt": redacted_prompt,
            "proof": proof_info,
        },
        allowed,
    )


@app.post("/filter")
def filter_prompt(payload: GuardRequest) -> dict[str, Any]:
    result, _ = evaluate_prompt(payload)
    return result


@app.post("/guarded_chat")
def guarded_chat(payload: GuardedChatRequest) -> dict[str, Any]:
    base_payload = GuardRequest(prompt=payload.prompt, user_id=payload.user_id, source=payload.source)
    evaluation, allowed = evaluate_prompt(base_payload)
    if not allowed:
        return evaluation

    redacted_prompt = str(evaluation["redacted_prompt"])
    or_resp = call_openrouter_chat(prompt=redacted_prompt, system=payload.system)
    return {**evaluation, "openrouter": {"model": os.getenv("OPENCLAW_MODEL"), "content": or_resp.content}}


@app.get("/metrics")
def metrics() -> Any:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/events")
def events(limit: int = 200) -> list[dict[str, Any]]:
    if limit < 1:
        limit = 1
    if limit > 2000:
        limit = 2000
    if not ACCESS_LOG.exists():
        return []

    lines = ACCESS_LOG.read_text(encoding="utf-8").splitlines()
    out: list[dict[str, Any]] = []
    for line in lines[-limit:]:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        obj.pop("prompt", None)
        obj.pop("redacted_prompt", None)
        out.append(obj)
    return out
