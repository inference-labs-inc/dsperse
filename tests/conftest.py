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


@pytest.fixture()
def slices_output_dir(tmp_path: Path) -> Path:
    out = tmp_path / "slices"
    out.mkdir(parents=True, exist_ok=True)
    yield out


@pytest.fixture()
def hardcoded_output_dir(tmp_path: Path) -> Path:
    out = tmp_path / "output"
    out.mkdir(parents=True, exist_ok=True)
    yield out


@pytest.fixture()
def analysis_output_dir(tmp_path: Path) -> Path:
    analysis = tmp_path / "analysis"
    yield analysis


@pytest.fixture()
def run_output_dir(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    yield run_dir


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


def _copy_to(src: Path, dest: Path) -> Path:
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest)
    return dest


def _do_slice(model_dir: Path, output_dir: Path, **extra_args):
    from types import SimpleNamespace
    from dsperse.src.cli.slice import slice_model
    args = dict(model_dir=str(model_dir), output_dir=str(output_dir), save_file=None, output_type="dirs")
    args.update(extra_args)
    slice_model(SimpleNamespace(**args))
    return output_dir


@pytest.fixture(scope="session")
def session_cache_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("session_cache")


@pytest.fixture(scope="class")
def doom_sliced_for_compile(models_root, tmp_path_factory):
    out = tmp_path_factory.mktemp("compile_doom") / "slices"
    return _do_slice(models_root / "doom", out)


@pytest.fixture(scope="class")
def doom_tiled_for_compile(models_root, tmp_path_factory):
    out = tmp_path_factory.mktemp("compile_doom_tiled") / "slices"
    return _do_slice(models_root / "doom", out, tile_size=1000)


@pytest.fixture(scope="session")
def pre_sliced_net(models_root, session_cache_dir):
    out = session_cache_dir / "net_sliced"
    return _do_slice(models_root / "net", out)


@pytest.fixture(scope="session")
def pre_sliced_doom(models_root, session_cache_dir):
    out = session_cache_dir / "doom_sliced"
    return _do_slice(models_root / "doom", out)


@pytest.fixture(scope="session")
def pre_sliced_doom_tiled(models_root, session_cache_dir):
    out = session_cache_dir / "doom_tiled_sliced"
    return _do_slice(models_root / "doom", out, tile_size=1000)


def _do_compile(src, dest, **kwargs):
    from types import SimpleNamespace
    from dsperse.src.cli.compile import compile_model
    work = _copy_to(src, dest)
    args = dict(path=str(work), input_file=None, layers=None, backend=None)
    args.update(kwargs)
    compile_model(SimpleNamespace(**args))
    return work


@pytest.fixture(scope="session")
def pre_compiled_net(pre_sliced_net, session_cache_dir, jstprove_available):
    if not jstprove_available:
        pytest.skip("JSTprove unavailable")
    return _do_compile(pre_sliced_net, session_cache_dir / "net_compiled_jst", backend="jstprove")


@pytest.fixture(scope="session")
def pre_compiled_net_both(pre_sliced_net, session_cache_dir, jstprove_available, ezkl_available):
    if not jstprove_available or not ezkl_available:
        pytest.skip("Both backends required")
    return _do_compile(pre_sliced_net, session_cache_dir / "net_compiled_both")


@pytest.fixture(scope="session")
def pre_sliced_doom_tiled_14(models_root, session_cache_dir):
    out = session_cache_dir / "doom_tiled14_sliced"
    return _do_slice(models_root / "doom", out, tile_size=14)


@pytest.fixture(scope="session")
def pre_compiled_doom_tiled(pre_sliced_doom_tiled, session_cache_dir, jstprove_available):
    if not jstprove_available:
        pytest.skip("JSTprove unavailable")
    return _do_compile(pre_sliced_doom_tiled, session_cache_dir / "doom_tiled_compiled", backend="jstprove")


@pytest.fixture(scope="session")
def pre_compiled_doom_tiled_14(pre_sliced_doom_tiled_14, session_cache_dir, jstprove_available):
    if not jstprove_available:
        pytest.skip("JSTprove unavailable")
    return _do_compile(pre_sliced_doom_tiled_14, session_cache_dir / "doom_tiled14_compiled", backend="jstprove")


@pytest.fixture()
def copy_to():
    return _copy_to
