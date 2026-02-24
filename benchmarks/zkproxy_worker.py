"""Persistent ZK proxy worker.

Reads JSON-RPC requests from stdin, writes responses to stdout.
Keeps JSTprove loaded and circuits cached across requests.
"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import onnx

from dsperse.src.backends.jstprove import JSTprove
from dsperse.src.backends.utils.jstprove_utils import JSTPROVE_SUPPORTED_OPS


class ZkProxyWorker:
    def __init__(self) -> None:
        self.jst = JSTprove()
        self._compiled: dict[str, Path] = {}
        self._work_dir = Path(tempfile.mkdtemp(prefix="zkproxy_"))

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        method = request.get("method", "")
        params = request.get("params", {})
        req_id = request.get("id", 0)

        dispatch = {
            "health": self._health,
            "compile": self._compile,
            "witness": self._witness,
            "prove": self._prove,
            "verify": self._verify,
            "guard_check": self._guard_check,
        }

        handler = dispatch.get(method)
        if handler is None:
            return self._error(req_id, -32601, f"Unknown method: {method}")

        try:
            t0 = time.perf_counter()
            result = handler(params)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            result["timing_ms"] = round(elapsed_ms, 3)
            return {"jsonrpc": "2.0", "id": req_id, "result": result}
        except Exception as e:
            tb = traceback.format_exc()
            return self._error(req_id, -32000, str(e), data=tb)

    def _health(self, params: dict) -> dict:
        version = JSTprove.get_version()
        return {
            "status": "ok",
            "jstprove_version": version,
            "cached_circuits": list(self._compiled.keys()),
            "supported_ops": sorted(JSTPROVE_SUPPORTED_OPS),
        }

    def _compile(self, params: dict) -> dict:
        model_path = Path(params["model_path"])
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        key = hashlib.sha256(model_path.read_bytes()).hexdigest()[:16]
        circuit_dir = self._work_dir / key
        circuit_dir.mkdir(parents=True, exist_ok=True)

        data = self.jst.circuitization_pipeline(
            model_path=model_path,
            output_path=circuit_dir,
        )

        circuit_path = data.get("circuit_path") or str(next(circuit_dir.glob("*_circuit.txt"), ""))
        if circuit_path:
            self._compiled[str(model_path)] = Path(circuit_path)

        return {
            "success": "error" not in data and "compile_error" not in data,
            "circuit_key": key,
            "circuit_path": circuit_path,
            "artifacts": {k: str(v) for k, v in data.items() if isinstance(v, (str, Path))},
        }

    def _witness(self, params: dict) -> dict:
        model_path = params["model_path"]
        features = params["features"]

        input_file = self._work_dir / "witness_input.json"
        output_file = self._work_dir / "witness_output.json"

        input_data = {"input_data": [features]}
        with input_file.open("w") as f:
            json.dump(input_data, f)

        ok, result = self.jst.generate_witness(
            input_file=input_file,
            model_path=model_path,
            output_file=output_file,
        )

        return {
            "success": ok,
            "witness_data": result if isinstance(result, dict) else {"raw": str(result)},
        }

    def _prove(self, params: dict) -> dict:
        witness_path = Path(params["witness_path"])
        circuit_path = Path(params["circuit_path"])
        proof_path = self._work_dir / f"proof_{time.time_ns()}.bin"

        ok, result = self.jst.prove(
            witness_path=witness_path,
            circuit_path=circuit_path,
            proof_path=proof_path,
        )

        proof_hash = ""
        if ok and proof_path.exists():
            proof_hash = hashlib.sha256(proof_path.read_bytes()).hexdigest()

        return {
            "success": ok,
            "proof_path": str(result) if ok else "",
            "proof_hash": proof_hash,
            "error": str(result) if not ok else "",
        }

    def _verify(self, params: dict) -> dict:
        ok = self.jst.verify(
            proof_path=params["proof_path"],
            circuit_path=params["circuit_path"],
            input_path=params["input_path"],
            output_path=params["output_path"],
            witness_path=params["witness_path"],
        )
        return {"verified": ok}

    def _guard_check(self, params: dict) -> dict:
        """Full pipeline: witness -> prove -> verify for a feature vector."""
        model_path = params["model_path"]
        features = params["features"]

        timings: dict[str, float] = {}

        t0 = time.perf_counter()
        witness_result = self._witness({"model_path": model_path, "features": features})
        timings["witness_ms"] = round((time.perf_counter() - t0) * 1000, 3)

        if not witness_result.get("success"):
            return {"success": False, "error": "witness failed", "timings": timings}

        logits = witness_result.get("witness_data", {}).get("logits")
        score = 0.0
        if logits:
            score = float(logits[0]) if isinstance(logits, list) else float(logits)

        circuit_path = self._compiled.get(model_path, "")
        if not circuit_path:
            return {
                "success": True,
                "score": score,
                "proof_hash": "",
                "verified": False,
                "timings": timings,
                "note": "no compiled circuit available, skipping prove/verify",
            }

        witness_bin = self._work_dir / "witness_input_witness.bin"
        if not witness_bin.exists():
            return {
                "success": True,
                "score": score,
                "proof_hash": "",
                "verified": False,
                "timings": timings,
                "note": "witness binary not found, skipping prove/verify",
            }

        t0 = time.perf_counter()
        prove_result = self._prove({
            "witness_path": str(witness_bin),
            "circuit_path": str(circuit_path),
        })
        timings["prove_ms"] = round((time.perf_counter() - t0) * 1000, 3)

        if not prove_result.get("success"):
            return {
                "success": True,
                "score": score,
                "proof_hash": "",
                "verified": False,
                "timings": timings,
                "error": prove_result.get("error", "prove failed"),
            }

        t0 = time.perf_counter()
        verify_result = self._verify({
            "proof_path": prove_result["proof_path"],
            "circuit_path": str(circuit_path),
            "input_path": str(self._work_dir / "witness_input.json"),
            "output_path": str(self._work_dir / "witness_output.json"),
            "witness_path": str(witness_bin),
        })
        timings["verify_ms"] = round((time.perf_counter() - t0) * 1000, 3)

        return {
            "success": True,
            "score": score,
            "proof_hash": prove_result.get("proof_hash", ""),
            "verified": verify_result.get("verified", False),
            "timings": timings,
        }

    @staticmethod
    def _error(req_id: int, code: int, message: str, data: str | None = None) -> dict:
        err: dict[str, Any] = {"code": code, "message": message}
        if data:
            err["data"] = data
        return {"jsonrpc": "2.0", "id": req_id, "error": err}


def main() -> None:
    worker = ZkProxyWorker()

    startup_msg = json.dumps({"jsonrpc": "2.0", "method": "startup", "params": {"status": "ready"}})
    sys.stdout.write(startup_msg + "\n")
    sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
        except json.JSONDecodeError as e:
            response = ZkProxyWorker._error(0, -32700, f"Parse error: {e}")
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()
            continue

        response = worker.handle(request)
        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
