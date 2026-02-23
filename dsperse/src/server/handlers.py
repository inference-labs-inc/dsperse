import asyncio
import json
import logging
import os
import shutil
import tempfile
from pathlib import Path

from dsperse.src.backends.jstprove import JSTprove
from dsperse.src.run.runner import Runner
from dsperse.src.server.state import RunState, RunStatus, WorkItem, SliceExecResult

logger = logging.getLogger(__name__)

DEFAULT_CIRCUIT_CACHE = os.path.expanduser("~/.bittensor/subnet-2/circuit_cache")
RUN_DATA_DIR = Path(tempfile.gettempdir()) / "dsperse_run_data"


class ServerContext:
    def __init__(self, circuit_cache_dir: str | None = None):
        self.state = RunState()
        self.circuit_cache_dir = circuit_cache_dir or DEFAULT_CIRCUIT_CACHE
        self.jstprove = JSTprove()

    def resolve_model_dir(self, circuit_id: str) -> Path:
        return Path(self.circuit_cache_dir) / f"model_{circuit_id}"

    def resolve_slices_dir(self, circuit_id: str) -> Path:
        return self.resolve_model_dir(circuit_id) / "slices"

    async def dispatch(self, request: dict) -> dict:
        method = request.get("method")
        if not method:
            return {"error": "missing 'method' field"}
        handler = HANDLERS.get(method)
        if not handler:
            return {"error": f"unknown method: {method}"}
        try:
            return await handler(self, request)
        except Exception as e:
            logger.exception("handler error for %s", method)
            return {"error": str(e)}


async def handle_start_incremental_run(ctx: ServerContext, req: dict) -> dict:
    circuit_id = req.get("circuit_id", "")
    inputs = req.get("inputs")
    run_source = req.get("run_source", "api")
    max_tiles = req.get("max_tiles")

    slices_dir = ctx.resolve_slices_dir(circuit_id)
    if not slices_dir.exists():
        return {"error": f"slices directory not found: {slices_dir}"}

    run = ctx.state.create_run(circuit_id, inputs, run_source, max_tiles)
    run.status = RunStatus.RUNNING

    data_dir = RUN_DATA_DIR / run.run_uid
    data_dir.mkdir(parents=True, exist_ok=True)

    asyncio.ensure_future(_process_run(ctx, run, inputs, slices_dir, data_dir))
    print(f"[start_incremental_run] run_uid={run.run_uid}, data_dir={data_dir}", flush=True)

    return {"run_uid": run.run_uid, "status": run.status.value}


async def _process_run(ctx: ServerContext, run, inputs, slices_dir: Path, data_dir: Path):
    tmp_dir = tempfile.mkdtemp(prefix="dsperse_run_")
    try:
        input_path = Path(tmp_dir) / "input.json"
        with open(input_path, "w") as f:
            json.dump(inputs, f)

        def on_slice_complete(slice_id, exec_info, run_dir):
            res = SliceExecResult(slice_id=slice_id)
            slice_run_dir = Path(run_dir) / slice_id
            slice_data_dir = data_dir / slice_id
            slice_data_dir.mkdir(parents=True, exist_ok=True)

            in_file = slice_run_dir / "input.json"
            out_file = slice_run_dir / "output.json"
            if in_file.exists():
                dest = slice_data_dir / "input.json"
                shutil.copy2(str(in_file), str(dest))
                res.inputs_path = str(dest)
            if out_file.exists():
                dest = slice_data_dir / "output.json"
                shutil.copy2(str(out_file), str(dest))
                res.outputs_path = str(dest)
            if hasattr(exec_info, "tiles") and exec_info.tiles:
                res.tiling = {"num_tiles": len(exec_info.tiles)}
            run.slice_results[slice_id] = res
            _build_work_item_for_slice(run, slice_id, res)
            print(f"[slice_complete] {slice_id}, inputs={res.inputs_path}, outputs={res.outputs_path}", flush=True)

        run_output_dir = Path(tmp_dir) / "run"
        runner = Runner(
            run_dir=str(run_output_dir),
            on_slice_complete=on_slice_complete,
            lazy=True,
        )
        await asyncio.to_thread(
            runner.run,
            input_json_path=str(input_path),
            slice_path=str(slices_dir),
        )
        run.status = RunStatus.COMPLETE
        print(f"[run_complete] {run.run_uid}, work_items={len(run.work_items)}", flush=True)
    except Exception as e:
        print(f"[run_failed] {run.run_uid}: {e}", flush=True)
        run.status = RunStatus.FAILED
        run.error = str(e)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _build_work_item_for_slice(run, slice_id: str, exec_res: SliceExecResult):
    proof_system = "JSTPROVE"
    tiling = exec_res.tiling
    if tiling and tiling.get("num_tiles", 1) > 1:
        num_tiles = tiling["num_tiles"]
        if run.max_tiles:
            num_tiles = min(num_tiles, run.max_tiles)
        for tile_idx in range(num_tiles):
            task_id = f"{run.run_uid}_{slice_id}_tile_{tile_idx}"
            run.work_items.append(WorkItem(
                circuit_id=run.circuit_id,
                run_uid=run.run_uid,
                slice_num=slice_id,
                task_id=task_id,
                is_tile=True,
                tile_idx=tile_idx,
                proof_system=proof_system,
                inputs_path=exec_res.inputs_path,
                outputs_path=exec_res.outputs_path,
                run_source=run.run_source,
            ))
    else:
        task_id = f"{run.run_uid}_{slice_id}"
        run.work_items.append(WorkItem(
            circuit_id=run.circuit_id,
            run_uid=run.run_uid,
            slice_num=slice_id,
            task_id=task_id,
            is_tile=False,
            tile_idx=None,
            proof_system=proof_system,
            inputs_path=exec_res.inputs_path,
            outputs_path=exec_res.outputs_path,
            run_source=run.run_source,
        ))


async def handle_get_run_status(ctx: ServerContext, req: dict) -> dict:
    run_uid = req.get("run_uid", "")
    run = ctx.state.get_run(run_uid)
    if not run:
        return {"error": f"unknown run_uid: {run_uid}"}
    return {
        "run_uid": run.run_uid,
        "status": run.status.value,
        "total_items": len(run.work_items),
        "completed_items": sum(1 for w in run.work_items if w.completed),
    }


async def handle_get_next_work(ctx: ServerContext, req: dict) -> dict:
    run_uid = req.get("run_uid", "")
    limit = req.get("limit", 50)
    run = ctx.state.get_run(run_uid)
    if not run:
        return {"error": f"unknown run_uid: {run_uid}", "items": []}
    pending = ctx.state.get_pending_work(run_uid)
    items = []
    for w in pending:
        item = {
            "slice_num": w.slice_num,
            "task_id": w.task_id,
            "is_tile": w.is_tile,
            "tile_idx": w.tile_idx,
            "proof_system": w.proof_system,
            "inputs_path": w.inputs_path,
            "outputs_path": w.outputs_path,
            "run_source": w.run_source,
        }
        items.append(item)
        if limit and len(items) >= limit:
            break
    return {"items": items}


async def handle_apply_slice_result(ctx: ServerContext, req: dict) -> dict:
    run_uid = req.get("run_uid", "")
    slice_num = req.get("slice_num", "")
    run = ctx.state.get_run(run_uid)
    if not run:
        return {"error": f"unknown run_uid: {run_uid}"}

    run.applied_results[f"slice_{slice_num}"] = {
        "success": req.get("success", False),
        "computed_outputs": req.get("computed_outputs"),
        "proof": req.get("proof"),
        "proof_system": req.get("proof_system"),
        "response_time_sec": req.get("response_time_sec", 0),
        "verification_time_sec": req.get("verification_time_sec", 0),
    }
    ctx.state.mark_work_completed(run_uid, slice_num)
    if ctx.state.all_work_done(run_uid):
        run.status = RunStatus.COMPLETE
    return {"status": run.status.value}


async def handle_apply_tile_result(ctx: ServerContext, req: dict) -> dict:
    run_uid = req.get("run_uid", "")
    slice_id = req.get("slice_id", "")
    tile_idx = req.get("tile_idx")
    run = ctx.state.get_run(run_uid)
    if not run:
        return {"error": f"unknown run_uid: {run_uid}"}

    key = f"tile_{slice_id}_{tile_idx}"
    run.applied_results[key] = {
        "task_id": req.get("task_id"),
        "success": req.get("success", False),
        "computed_outputs": req.get("computed_outputs"),
        "proof": req.get("proof"),
        "witness": req.get("witness"),
        "proof_system": req.get("proof_system"),
        "response_time_sec": req.get("response_time_sec", 0),
        "verification_time_sec": req.get("verification_time_sec", 0),
    }
    ctx.state.mark_work_completed(run_uid, slice_id, tile_idx)
    if ctx.state.all_work_done(run_uid):
        run.status = RunStatus.COMPLETE
    return {"status": run.status.value}


def _flatten_list(lst: list) -> list:
    result = []
    for item in lst:
        if isinstance(item, list):
            result.extend(_flatten_list(item))
        else:
            result.append(item)
    return result


def _flatten_inputs(original_inputs) -> list | None:
    if original_inputs is None:
        return None
    if isinstance(original_inputs, list):
        return _flatten_list(original_inputs)
    if isinstance(original_inputs, dict):
        vals = original_inputs.get("input_data") if "input_data" in original_inputs else original_inputs.get("input")
        if vals is None or not isinstance(vals, list):
            return None
        return _flatten_list(vals)
    return None


async def handle_verify_incremental_slice_with_witness(ctx: ServerContext, req: dict) -> dict:
    circuit_id = req.get("circuit_id", "")
    slice_num = req.get("slice_num", "")
    original_inputs = req.get("original_inputs")
    witness_hex = req.get("witness_hex", "")
    proof_hex = req.get("proof_hex", "")

    model_dir = ctx.resolve_model_dir(circuit_id)
    slices_dir = model_dir / "slices"
    slice_dir = slices_dir / slice_num
    circuit_path = None
    for candidate in slice_dir.glob("*_jstprove_circuit.txt"):
        circuit_path = candidate
        break
    if not circuit_path:
        return {"error": f"circuit not found for {circuit_id}/{slice_num}"}

    try:
        witness_bytes = bytes.fromhex(witness_hex)
    except ValueError:
        return {"error": "invalid hex for witness_hex", "success": False}
    try:
        proof_bytes = bytes.fromhex(proof_hex)
    except ValueError:
        return {"error": "invalid hex for proof_hex", "success": False}

    flat_inputs = _flatten_inputs(original_inputs)

    metadata_path = circuit_path.parent / f"{circuit_path.stem}_metadata.json"
    num_inputs = 0
    if metadata_path.exists():
        with open(metadata_path) as f:
            meta = json.load(f)
        num_inputs = meta.get("num_inputs", 0)

    success, extracted = await asyncio.to_thread(
        ctx.jstprove.verify_with_io_extraction,
        circuit_path=circuit_path,
        witness_bytes=witness_bytes,
        proof_bytes=proof_bytes,
        num_inputs=num_inputs,
        expected_inputs=flat_inputs,
    )

    result = {"success": success}
    if extracted:
        result["computed_outputs"] = extracted.get("rescaled_outputs")
    return result


async def handle_prove(ctx: ServerContext, req: dict) -> dict:
    model_id = req.get("model_id", "")
    inputs = req.get("inputs")

    slices_dir = ctx.resolve_slices_dir(model_id)
    if not slices_dir.exists():
        return {"error": f"slices directory not found: {slices_dir}"}

    with tempfile.TemporaryDirectory() as tmp_dir:
        input_path = Path(tmp_dir) / "input.json"
        with open(input_path, "w") as f:
            json.dump(inputs, f)

        run_output_dir = Path(tmp_dir) / "run"
        runner = Runner(run_dir=str(run_output_dir))
        results = await asyncio.to_thread(
            runner.run,
            input_json_path=str(input_path),
            slice_path=str(slices_dir),
        )
        return {"output": results.get("output"), "proof_system": "JSTPROVE"}


async def handle_prove_slice(ctx: ServerContext, req: dict) -> dict:
    circuit_id = req.get("circuit_id", "")
    slice_num = req.get("slice_num", "")
    inputs = req.get("inputs")
    proof_system = req.get("proof_system", "JSTPROVE")

    model_dir = ctx.resolve_model_dir(circuit_id)
    slices_dir = model_dir / "slices"
    slice_dir = slices_dir / slice_num

    circuit_suffix = "_jstprove_circuit.txt" if "jstprove" in proof_system.lower() else "_ezkl_circuit"
    circuit_path = None
    for candidate in slice_dir.glob(f"*{circuit_suffix}"):
        circuit_path = candidate
        break
    if not circuit_path:
        return {"error": f"circuit not found for {circuit_id}/{slice_num}"}

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        input_file = tmp / "input.json"
        output_file = tmp / "output.json"
        witness_path = tmp / "output_witness.bin"
        proof_path = tmp / "proof.bin"

        with open(input_file, "w") as f:
            json.dump(inputs, f)

        ok, witness_result = await asyncio.to_thread(
            ctx.jstprove.generate_witness,
            input_file=str(input_file),
            model_path=str(circuit_path),
            output_file=str(output_file),
        )
        if not ok:
            return {"error": f"witness generation failed: {witness_result}", "success": False}

        if not witness_path.exists():
            return {"error": "witness file not created", "success": False}

        ok, proof_result = await asyncio.to_thread(
            ctx.jstprove.prove,
            witness_path=str(witness_path),
            circuit_path=str(circuit_path),
            proof_path=str(proof_path),
        )
        if not ok:
            return {"error": f"proof generation failed: {proof_result}", "success": False}

        proof_hex = proof_path.read_bytes().hex()
        witness_hex = witness_path.read_bytes().hex()

        return {
            "success": True,
            "proof": proof_hex,
            "witness": witness_hex,
            "proof_system": proof_system,
        }


async def handle_generate_requests(ctx: ServerContext, req: dict) -> dict:
    limit = req.get("limit", 50)
    items = ctx.state.generate_all_requests(limit=limit)
    return {"items": items}


async def handle_get_work_data(ctx: ServerContext, req: dict) -> dict:
    task_id = req.get("task_id", "")
    item = ctx.state.get_work_item_by_task_id(task_id)
    if not item:
        return {"error": f"unknown task_id: {task_id}"}
    inputs = None
    outputs = None
    if item.inputs_path and Path(item.inputs_path).exists():
        with open(item.inputs_path) as f:
            inputs = json.load(f)
    if item.outputs_path and Path(item.outputs_path).exists():
        with open(item.outputs_path) as f:
            outputs = json.load(f)
    return {
        "task_id": item.task_id,
        "inputs": inputs,
        "outputs": outputs,
    }


HANDLERS = {
    "start_incremental_run": handle_start_incremental_run,
    "get_run_status": handle_get_run_status,
    "get_next_work": handle_get_next_work,
    "apply_slice_result": handle_apply_slice_result,
    "apply_tile_result": handle_apply_tile_result,
    "verify_incremental_slice_with_witness": handle_verify_incremental_slice_with_witness,
    "prove": handle_prove,
    "prove_slice": handle_prove_slice,
    "generate_requests": handle_generate_requests,
    "get_work_data": handle_get_work_data,
}
