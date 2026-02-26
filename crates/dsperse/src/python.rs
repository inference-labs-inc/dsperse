use std::path::PathBuf;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use crate::backend::jstprove::JstproveBackend;
use crate::error::DsperseError;
use crate::pipeline::{self, RunConfig};

fn to_py_err(e: DsperseError) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

fn to_pretty_json<T: serde::Serialize>(value: &T) -> PyResult<String> {
    serde_json::to_string_pretty(value)
        .map_err(|e| to_py_err(DsperseError::Other(format!("pretty-json serialization failed: {e}"))))
}

fn require_nonzero(parallel: usize) -> PyResult<()> {
    if parallel == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("parallel must be > 0"));
    }
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (model_path, output_dir=None, tile_size=None))]
fn slice_model(py: Python<'_>, model_path: &str, output_dir: Option<&str>, tile_size: Option<usize>) -> PyResult<String> {
    let model = PathBuf::from(model_path);
    let out = output_dir.map(PathBuf::from);
    let metadata = py.allow_threads(|| {
        crate::slicer::slice_model(&model, out.as_deref(), tile_size)
    }).map_err(to_py_err)?;
    to_pretty_json(&metadata)
}

#[pyfunction]
#[pyo3(signature = (slices_dir, parallel=1, weights_as_inputs=false, layers=None))]
fn compile_slices(py: Python<'_>, slices_dir: &str, parallel: usize, weights_as_inputs: bool, layers: Option<Vec<usize>>) -> PyResult<()> {
    require_nonzero(parallel)?;
    let backend = JstproveBackend::default();
    let dir = PathBuf::from(slices_dir);
    py.allow_threads(|| {
        pipeline::compile_slices(&dir, &backend, parallel, weights_as_inputs, layers.as_deref())
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
