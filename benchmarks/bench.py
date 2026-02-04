import argparse
import json
import os
import resource
import shutil
import statistics
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_ROOT = PROJECT_ROOT / "tests" / "models"
ITERATIONS = 3


def measure(fn):
    start = time.perf_counter()
    fn()
    elapsed = time.perf_counter() - start
    usage = resource.getrusage(resource.RUSAGE_SELF)
    peak_kb = usage.ru_maxrss
    if os.uname().sysname == "Darwin":
        peak_mb = peak_kb / (1024 * 1024)
    else:
        peak_mb = peak_kb / 1024
    return elapsed, peak_mb


def bench_slice(model_name, tile_size=None):
    from dsperse.src.cli.slice import slice_model

    model_dir = str(MODELS_ROOT / model_name)

    def run():
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(
                model_dir=model_dir,
                output_dir=tmp,
                save_file=None,
                output_type="dirs",
                tile_size=tile_size,
            )
            slice_model(args)

    return run


def bench_run(model_name, slices_dir):
    from dsperse.src.cli.run import run_inference

    input_file = str(MODELS_ROOT / model_name / "input.json")

    def run():
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp) / "slices"
            shutil.copytree(slices_dir, work_dir)
            args = SimpleNamespace(
                path=str(work_dir),
                input_file=input_file,
                output_file=None,
                force_backend="onnx",
                threads=1,
                batch=False,
                run_dir=str(Path(tmp) / "run"),
            )
            run_inference(args)

    return run


def pre_slice(model_name, tile_size=None):
    from dsperse.src.cli.slice import slice_model

    tmp = tempfile.mkdtemp()
    args = SimpleNamespace(
        model_dir=str(MODELS_ROOT / model_name),
        output_dir=tmp,
        save_file=None,
        output_type="dirs",
        tile_size=tile_size,
    )
    slice_model(args)
    return tmp


BENCHMARKS = [
    {"name": "slice (net)", "factory": lambda: bench_slice("net")},
    {"name": "slice (doom)", "factory": lambda: bench_slice("doom")},
    {"name": "slice+tile (doom, t=1000)", "factory": lambda: bench_slice("doom", tile_size=1000)},
]


def build_run_benchmarks():
    net_slices = pre_slice("net")
    doom_slices = pre_slice("doom")

    return [
        {"name": "run (net, onnx)", "factory": lambda: bench_run("net", net_slices)},
        {"name": "run (doom, onnx)", "factory": lambda: bench_run("doom", doom_slices)},
    ], [net_slices, doom_slices]


def run_benchmarks():
    results = []
    run_benches, tmp_dirs = build_run_benchmarks()
    all_benches = BENCHMARKS + run_benches

    for b in all_benches:
        fn = b["factory"]()
        times = []
        peak_rss = 0.0
        for i in range(ITERATIONS):
            elapsed, rss = measure(fn)
            times.append(round(elapsed, 4))
            peak_rss = max(peak_rss, rss)
            print(f"  {b['name']} iter {i+1}/{ITERATIONS}: {elapsed:.3f}s")

        results.append({
            "name": b["name"],
            "times": times,
            "median": round(statistics.median(times), 4),
            "peak_rss_mb": round(peak_rss, 1),
        })

    for d in tmp_dirs:
        shutil.rmtree(d, ignore_errors=True)

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark dsperse commands")
    parser.add_argument("--output", "-o", default="benchmark_results.json")
    args = parser.parse_args()

    print(f"Running benchmarks ({ITERATIONS} iterations each)...")
    results = run_benchmarks()

    output = {"benchmarks": results}
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults written to {args.output}")
    for r in results:
        print(f"  {r['name']}: {r['median']}s (peak RSS: {r['peak_rss_mb']}MB)")


if __name__ == "__main__":
    main()
