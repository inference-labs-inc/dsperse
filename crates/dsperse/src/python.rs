use std::path::PathBuf;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use crate::backend::jstprove::JstproveBackend;
use crate::error::DsperseError;
use crate::pipeline::{self, RunConfig};

use jstprove_circuits::{ProofSystem, ProofSystemParseError};

fn to_py_err(e: DsperseError) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

fn to_pretty_json<T: serde::Serialize>(value: &T) -> PyResult<String> {
    serde_json::to_string_pretty(value)
        .map_err(|e| to_py_err(DsperseError::Other(format!("pretty-json serialization failed: {e}"))))
}

fn resolve_ops(proof_system: &str, circuit_ops: Option<&[String]>) -> PyResult<Vec<String>> {
    let ps: ProofSystem = proof_system
        .parse()
        .map_err(|e: ProofSystemParseError| PyRuntimeError::new_err(e.to_string()))?;
    let supported = ps.supported_ops();
    match circuit_ops {
        None => Ok(supported.iter().map(|s| (*s).to_string()).collect()),
        Some(ops) => {
            for op in ops {
                if !supported.contains(&op.as_str()) {
                    return Err(PyRuntimeError::new_err(format!(
                        "op {op:?} not supported by proof system {ps}. Supported: {supported:?}"
                    )));
                }
            }
            Ok(ops.to_vec())
        }
    }
}

fn require_nonzero(parallel: usize) -> PyResult<()> {
    if parallel == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("parallel must be > 0"));
    }
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (model_path, output_dir=None, tile_size=None, proof_system="expander", circuit_ops=None))]
fn slice_model(py: Python<'_>, model_path: &str, output_dir: Option<&str>, tile_size: Option<usize>, proof_system: &str, circuit_ops: Option<Vec<String>>) -> PyResult<String> {
    let model = PathBuf::from(model_path);
    let out = output_dir.map(PathBuf::from);
    let ops = resolve_ops(proof_system, circuit_ops.as_deref())?;
    let ops_refs: Vec<&str> = ops.iter().map(String::as_str).collect();
    let metadata = py.allow_threads(|| {
        crate::slicer::slice_model(&model, out.as_deref(), tile_size, &ops_refs)
    }).map_err(to_py_err)?;
    to_pretty_json(&metadata)
}

#[pyfunction]
#[pyo3(signature = (slices_dir, parallel=1, weights_as_inputs=false, layers=None, proof_system="expander", circuit_ops=None, fast_compile=false))]
fn compile_slices(py: Python<'_>, slices_dir: &str, parallel: usize, weights_as_inputs: bool, layers: Option<Vec<usize>>, proof_system: &str, circuit_ops: Option<Vec<String>>, fast_compile: bool) -> PyResult<()> {
    require_nonzero(parallel)?;
    let backend = JstproveBackend::default().with_fast_compile(fast_compile);
    let dir = PathBuf::from(slices_dir);
    let ops = resolve_ops(proof_system, circuit_ops.as_deref())?;
    let ops_refs: Vec<&str> = ops.iter().map(String::as_str).collect();
    py.allow_threads(|| {
        pipeline::compile_slices(&dir, &backend, parallel, weights_as_inputs, layers.as_deref(), &ops_refs)
    }).map_err(to_py_err)
}

#[pyfunction]
#[pyo3(signature = (slices_dir, input_file, run_dir, parallel=1, batch=false, weights_onnx=None))]
fn run_inference(py: Python<'_>, slices_dir: &str, input_file: &str, run_dir: &str, parallel: usize, batch: bool, weights_onnx: Option<&str>) -> PyResult<String> {
    require_nonzero(parallel)?;
    let backend = JstproveBackend::default();
    let config = RunConfig { parallel, batch, weights_onnx: weights_onnx.map(PathBuf::from) };
    let sd = PathBuf::from(slices_dir);
    let inf = PathBuf::from(input_file);
    let rd = PathBuf::from(run_dir);
    let metadata = py.allow_threads(|| {
        pipeline::run_inference(&sd, &inf, &rd, &backend, &config)
    }).map_err(to_py_err)?;
    to_pretty_json(&metadata)
}

#[pyfunction]
#[pyo3(signature = (run_dir, slices_dir, parallel=1))]
fn prove_run(py: Python<'_>, run_dir: &str, slices_dir: &str, parallel: usize) -> PyResult<String> {
    require_nonzero(parallel)?;
    let backend = JstproveBackend::default();
    let rd = PathBuf::from(run_dir);
    let sd = PathBuf::from(slices_dir);
    let metadata = py.allow_threads(|| {
        pipeline::prove_run(&rd, &sd, &backend, parallel)
    }).map_err(to_py_err)?;
    to_pretty_json(&metadata)
}

#[pyfunction]
#[pyo3(signature = (run_dir, slices_dir, parallel=1))]
fn verify_run(py: Python<'_>, run_dir: &str, slices_dir: &str, parallel: usize) -> PyResult<String> {
    require_nonzero(parallel)?;
    let backend = JstproveBackend::default();
    let rd = PathBuf::from(run_dir);
    let sd = PathBuf::from(slices_dir);
    let metadata = py.allow_threads(|| {
        pipeline::verify_run(&rd, &sd, &backend, parallel)
    }).map_err(to_py_err)?;
    to_pretty_json(&metadata)
}

#[pyfunction]
#[pyo3(signature = (argv=None))]
fn cli_main(py: Python<'_>, argv: Option<Vec<String>>) -> PyResult<()> {
    use clap::Parser;
    use tracing_subscriber::EnvFilter;

    let cli = match argv {
        Some(args) => crate::cli::Cli::try_parse_from(args.clone()).or_else(|_| {
            let mut with_prog = vec!["dsperse".to_string()];
            with_prog.extend(args);
            crate::cli::Cli::try_parse_from(with_prog)
        }),
        None => crate::cli::Cli::try_parse(),
    }.map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(&cli.log_level)),
        )
        .try_init();

    let result = py.allow_threads(|| crate::cli::dispatch(cli.command));

    result.map_err(to_py_err)
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(slice_model, m)?)?;
    m.add_function(wrap_pyfunction!(compile_slices, m)?)?;
    m.add_function(wrap_pyfunction!(run_inference, m)?)?;
    m.add_function(wrap_pyfunction!(prove_run, m)?)?;
    m.add_function(wrap_pyfunction!(verify_run, m)?)?;
    m.add_function(wrap_pyfunction!(cli_main, m)?)?;
    Ok(())
}
