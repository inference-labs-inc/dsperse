use std::path::PathBuf;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use crate::archive::converter::{self, FormatType};
use crate::backend::jstprove::JstproveBackend;
use crate::error::DsperseError;
use crate::pipeline::{self, RunConfig};

fn to_py_err(e: DsperseError) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

#[pyfunction]
#[pyo3(signature = (model_path, output_dir=None, tile_size=None))]
fn slice_model(model_path: &str, output_dir: Option<&str>, tile_size: Option<usize>) -> PyResult<String> {
    let model = PathBuf::from(model_path);
    let out = output_dir.map(PathBuf::from);
    let metadata = crate::slicer::slice_model(&model, out.as_deref(), tile_size).map_err(to_py_err)?;
    serde_json::to_string_pretty(&metadata).map_err(|e| to_py_err(DsperseError::Json(e)))
}

#[pyfunction]
#[pyo3(signature = (slices_dir, parallel=1, weights_as_inputs=false, layers=None))]
fn compile_slices(slices_dir: &str, parallel: usize, weights_as_inputs: bool, layers: Option<Vec<usize>>) -> PyResult<()> {
    let backend = JstproveBackend::default();
    pipeline::compile_slices(
        &PathBuf::from(slices_dir),
        &backend,
        parallel,
        weights_as_inputs,
        layers.as_deref(),
    )
    .map_err(to_py_err)
}

#[pyfunction]
#[pyo3(signature = (slices_dir, input_file, run_dir, parallel=1, batch=false))]
fn run_inference(slices_dir: &str, input_file: &str, run_dir: &str, parallel: usize, batch: bool) -> PyResult<String> {
    let backend = JstproveBackend::default();
    let config = RunConfig { parallel, batch };
    let metadata = pipeline::run_inference(
        &PathBuf::from(slices_dir),
        &PathBuf::from(input_file),
        &PathBuf::from(run_dir),
        &backend,
        &config,
    )
    .map_err(to_py_err)?;
    serde_json::to_string_pretty(&metadata).map_err(|e| to_py_err(DsperseError::Json(e)))
}

#[pyfunction]
#[pyo3(signature = (run_dir, slices_dir, parallel=1))]
fn prove_run(run_dir: &str, slices_dir: &str, parallel: usize) -> PyResult<String> {
    let backend = JstproveBackend::default();
    let metadata = pipeline::prove_run(
        &PathBuf::from(run_dir),
        &PathBuf::from(slices_dir),
        &backend,
        parallel,
    )
    .map_err(to_py_err)?;
    serde_json::to_string_pretty(&metadata).map_err(|e| to_py_err(DsperseError::Json(e)))
}

#[pyfunction]
#[pyo3(signature = (run_dir, slices_dir, parallel=1))]
fn verify_run(run_dir: &str, slices_dir: &str, parallel: usize) -> PyResult<String> {
    let backend = JstproveBackend::default();
    let metadata = pipeline::verify_run(
        &PathBuf::from(run_dir),
        &PathBuf::from(slices_dir),
        &backend,
        parallel,
    )
    .map_err(to_py_err)?;
    serde_json::to_string_pretty(&metadata).map_err(|e| to_py_err(DsperseError::Json(e)))
}

#[pyfunction]
#[pyo3(signature = (input, to, output=None, expand_slices=false, cleanup=false))]
fn convert(input: &str, to: &str, output: Option<&str>, expand_slices: bool, cleanup: bool) -> PyResult<String> {
    let format: FormatType = match to {
        "dirs" => FormatType::Dirs,
        "dslice" => FormatType::Dslice,
        "dsperse" => FormatType::Dsperse,
        other => return Err(PyRuntimeError::new_err(format!("unknown format: {other}"))),
    };
    let result = converter::convert(
        &PathBuf::from(input),
        format,
        output.map(PathBuf::from).as_deref(),
        cleanup,
        expand_slices,
    )
    .map_err(to_py_err)?;
    Ok(result.to_string_lossy().to_string())
}

#[pyfunction]
fn cli_main() -> PyResult<()> {
    use clap::Parser;
    use tracing_subscriber::EnvFilter;

    #[derive(Parser)]
    #[command(name = "dsperse", about = "Distributed zkML Toolkit")]
    struct Cli {
        #[command(subcommand)]
        command: Commands,
        #[arg(long, default_value = "warn", global = true)]
        log_level: String,
    }

    #[derive(clap::Subcommand)]
    enum Commands {
        Slice(crate::cli::SliceArgs),
        Compile(crate::cli::CompileArgs),
        Run(crate::cli::RunArgs),
        Prove(crate::cli::ProveArgs),
        Verify(crate::cli::VerifyArgs),
        #[command(name = "full-run")]
        FullRun(crate::cli::FullRunArgs),
        Convert(crate::cli::ConvertArgs),
    }

    let cli = Cli::parse();

    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(&cli.log_level)),
        )
        .init();

    let result = match cli.command {
        Commands::Slice(args) => crate::cli::cmd_slice(args),
        Commands::Compile(args) => crate::cli::cmd_compile(args),
        Commands::Run(args) => crate::cli::cmd_run(args),
        Commands::Prove(args) => crate::cli::cmd_prove(args),
        Commands::Verify(args) => crate::cli::cmd_verify(args),
        Commands::FullRun(args) => crate::cli::cmd_full_run(args),
        Commands::Convert(args) => crate::cli::cmd_convert(args),
    };

    result.map_err(to_py_err)
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(slice_model, m)?)?;
    m.add_function(wrap_pyfunction!(compile_slices, m)?)?;
    m.add_function(wrap_pyfunction!(run_inference, m)?)?;
    m.add_function(wrap_pyfunction!(prove_run, m)?)?;
    m.add_function(wrap_pyfunction!(verify_run, m)?)?;
    m.add_function(wrap_pyfunction!(convert, m)?)?;
    m.add_function(wrap_pyfunction!(cli_main, m)?)?;
    Ok(())
}
