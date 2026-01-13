import shutil
from pathlib import Path

import pytest


# Session-scoped helpers to locate project and models
@pytest.fixture(scope="session")
def project_root() -> Path:
    """Absolute path to the project root (session-scoped)."""
    # This file resides under <project_root>/tests/, so parent is the project root
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def models_root(project_root: Path) -> Path:
    """Absolute path to bundled test models directory under tests/models."""
    return project_root / "tests" / "models"


# Resolve model directory from parameterized model_name and ensure it exists
@pytest.fixture()
def model_dir(models_root: Path, model_name: str) -> Path:
    p = models_root / model_name
    onnx_file = p / "model.onnx"
    assert p.exists(), f"Model directory not found: {p}"
    assert onnx_file.exists(), f"ONNX model not found: {onnx_file}"
    return p


# Per-test temporary output directory with automatic cleanup
@pytest.fixture()
def output_dir(tmp_path: Path) -> Path:
    """Create an isolated output directory for a test and clean it up afterwards."""
    out = tmp_path / "slices_out"
    out.mkdir(parents=True, exist_ok=True)
    try:
        yield out
    finally:
        # Ensure artifacts produced by the CLI are removed after each test
        if out.exists():
            shutil.rmtree(out, ignore_errors=True)


# Cleanup fixture for the default CLI output directory (<model_dir>/slices)
@pytest.fixture()
def slices_output_dir(model_dir: Path) -> Path:
    out = model_dir / "slices"
    # Pre-clean to avoid leftovers from previous runs
    if out.exists():
        shutil.rmtree(out, ignore_errors=True)
    # Also remove any sibling dsperse artifact that may be generated
    dsperse_artifact = out.parent / f"{out.name}.dsperse"
    if dsperse_artifact.exists():
        try:
            dsperse_artifact.unlink()
        except Exception:
            pass
    try:
        yield out
    finally:
        if out.exists():
            shutil.rmtree(out, ignore_errors=True)
        if dsperse_artifact.exists():
            try:
                dsperse_artifact.unlink()
            except Exception:
                pass


# Relative custom output directory for integration tests
@pytest.fixture()
def hardcoded_output_dir(project_root: Path) -> Path:
    """Provide a relative output directory under tests/models and ensure cleanup.

    Target (relative to project root): tests/models/output
    """
    out = project_root / "tests" / "models" / "output"
    # Pre-clean to avoid leftovers from previous runs
    if out.exists():
        shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)
    # Also remove any sibling dsperse artifact that may be generated
    dsperse_artifact = out.parent / f"{out.name}.dsperse"
    if dsperse_artifact.exists():
        try:
            dsperse_artifact.unlink()
        except Exception:
            pass
    try:
        yield out
    finally:
        if out.exists():
            shutil.rmtree(out, ignore_errors=True)
        if dsperse_artifact.exists():
            try:
                dsperse_artifact.unlink()
            except Exception:
                pass


# Cleanup for model analysis metadata directory (<model_dir>/analysis)
@pytest.fixture()
def analysis_output_dir(model_dir: Path) -> Path:
    analysis = model_dir / "analysis"
    # Pre-clean
    if analysis.exists():
        shutil.rmtree(analysis, ignore_errors=True)
    try:
        yield analysis
    finally:
        if analysis.exists():
            shutil.rmtree(analysis, ignore_errors=True)


# Cleanup for model run artifacts directory (<model_dir>/run)
@pytest.fixture()
def run_output_dir(model_dir: Path) -> Path:
    """Ensure the <model_dir>/run directory is cleaned before and after tests."""
    run_dir = model_dir / "run"
    # Pre-clean
    if run_dir.exists():
        shutil.rmtree(run_dir, ignore_errors=True)
    try:
        yield run_dir
    finally:
        if run_dir.exists():
            shutil.rmtree(run_dir, ignore_errors=True)


@pytest.fixture(scope="session")
def jstprove_available():
    """Fixture to check if JSTprove backend is available."""
    try:
        from dsperse.src.backends.jstprove import JSTprove
        _j = JSTprove()
        return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def ezkl_available():
    """Fixture to check if EZKL backend is available."""
    try:
        from dsperse.src.backends.ezkl import EZKL
        _e = EZKL()
        return True
    except Exception:
        return False
