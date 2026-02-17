from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ProofResult:
    success: bool
    run_dir: str
    verified: bool
    proof_file: str
    proof_hash_sha256: str
    details: str


def run_cmd(command: list[str], cwd: Path) -> tuple[int, str]:
    proc = subprocess.run(
        command,
        cwd=str(cwd),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc.returncode, proc.stdout


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_dsperse_proof_flow(
    dsperse_repo: Path,
    slices_dir: Path,
    input_file: Path,
    layer_index: int,
) -> ProofResult:
    run_cmdline = [
        "bash",
        "-lc",
        f"source .venv/bin/activate && dsperse run -s {slices_dir} -i {input_file}",
    ]
    code, output = run_cmd(run_cmdline, dsperse_repo)
    if code != 0:
        return ProofResult(False, "", False, "", "", output)

    run_root = slices_dir.parent / "run"
    run_dirs = sorted([p for p in run_root.glob("run_*") if p.is_dir()])
    if not run_dirs:
        return ProofResult(False, "", False, "", "", "No run directory produced by dsperse run.")

    latest_run = run_dirs[-1]
    prove_cmdline = [
        "bash",
        "-lc",
        f"source .venv/bin/activate && dsperse prove --rd {latest_run} -s {slices_dir}",
    ]
    p_code, p_output = run_cmd(prove_cmdline, dsperse_repo)
    if p_code != 0:
        return ProofResult(False, str(latest_run), False, "", "", p_output)

    verify_cmdline = [
        "bash",
        "-lc",
        f"source .venv/bin/activate && dsperse verify --rd {latest_run} -s {slices_dir}",
    ]
    v_code, v_output = run_cmd(verify_cmdline, dsperse_repo)
    if v_code != 0:
        return ProofResult(False, str(latest_run), False, "", "", v_output)

    result_file = latest_run / "run_results.json"
    if result_file.exists():
        parsed = json.loads(result_file.read_text(encoding="utf-8"))
        proof_file = ""
        verified = False
        try:
            for item in parsed["execution_chain"]["execution_results"]:
                if item.get("slice_id") == f"slice_{layer_index}":
                    proof_file = item.get("proof_execution", {}).get("proof_file", "") or ""
                    verified = bool(item.get("verification_execution", {}).get("verified", False))
                    break
        except Exception:
            proof_file = ""
            verified = False

        proof_hash = ""
        if proof_file:
            pf = Path(proof_file)
            if pf.exists():
                proof_hash = sha256_file(pf)

        details = json.dumps(
            {
                "run_dir": str(latest_run),
                "layer_index_targeted": layer_index,
                "proof_file": proof_file,
                "proof_verified": verified,
                "proof_hash_sha256": proof_hash,
            }
        )
        return ProofResult(True, str(latest_run), verified, proof_file, proof_hash, details)

    return ProofResult(True, str(latest_run), False, "", "", "Proof and verify completed.")
