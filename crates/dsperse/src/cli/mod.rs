use std::num::NonZeroUsize;
use std::path::PathBuf;

use clap::{Args, Parser, Subcommand};

use crate::backend::jstprove::JstproveBackend;
use crate::error::{DsperseError, Result};
use crate::pipeline::{self, RunConfig};

use jstprove_circuits::ProofSystem;

#[derive(Parser)]
#[command(name = "dsperse", about = "Distributed zkML Toolkit")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
    #[arg(long, default_value = "warn", global = true)]
    pub log_level: String,
}

#[derive(Subcommand)]
pub enum Commands {
    Slice(SliceArgs),
    Compile(CompileArgs),
    Run(RunArgs),
    Prove(ProveArgs),
    Verify(VerifyArgs),
    #[command(name = "full-run")]
    FullRun(FullRunArgs),
}

pub fn dispatch(command: Commands) -> Result<()> {
    match command {
        Commands::Slice(args) => cmd_slice(args),
        Commands::Compile(args) => cmd_compile(args),
        Commands::Run(args) => cmd_run(args),
        Commands::Prove(args) => cmd_prove(args),
        Commands::Verify(args) => cmd_verify(args),
        Commands::FullRun(args) => cmd_full_run(args),
    }
}

#[derive(Args)]
pub struct SliceArgs {
    #[arg(long)]
    pub model_dir: PathBuf,
    #[arg(long)]
    pub output_dir: Option<PathBuf>,
    #[arg(long)]
    pub tile_size: Option<usize>,
    #[arg(long, default_value = "expander", help = "Proof system backend (expander or remainder)")]
    pub proof_system: String,
    #[arg(long, help = "Comma-separated ONNX op names to compile via the proof backend (default: all supported)")]
    pub circuit_ops: Option<String>,
}

#[derive(Args)]
pub struct CompileArgs {
    #[arg(long)]
    pub model_dir: PathBuf,
    #[arg(long)]
    pub slices_dir: Option<PathBuf>,
    #[arg(long)]
    pub layers: Option<String>,
    #[arg(long, default_value = "1")]
    pub parallel: NonZeroUsize,
    #[arg(long)]
    pub weights_as_inputs: bool,
    #[arg(long, default_value = "expander", help = "Proof system backend (expander or remainder)")]
    pub proof_system: String,
    #[arg(long, help = "Comma-separated ONNX op names to compile via the proof backend (default: all supported)")]
    pub circuit_ops: Option<String>,
}

#[derive(Args)]
pub struct RunArgs {
    #[arg(long)]
    pub model_dir: PathBuf,
    #[arg(long)]
    pub input_file: PathBuf,
    #[arg(long)]
    pub run_dir: Option<PathBuf>,
    #[arg(long)]
    pub slices_dir: Option<PathBuf>,
    #[arg(long, default_value = "1")]
    pub parallel: NonZeroUsize,
    #[arg(long)]
    pub batch: bool,
    #[arg(long, help = "Path to consumer ONNX with fine-tuned weights to inject at inference time")]
    pub weights: Option<PathBuf>,
}

#[derive(Args)]
pub struct ProveArgs {
    #[arg(long)]
    pub run_dir: PathBuf,
    #[arg(long)]
    pub model_dir: PathBuf,
    #[arg(long)]
    pub slices_dir: Option<PathBuf>,
    #[arg(long, default_value = "1")]
    pub parallel: NonZeroUsize,
}

#[derive(Args)]
pub struct VerifyArgs {
    #[arg(long)]
    pub run_dir: PathBuf,
    #[arg(long)]
    pub model_dir: PathBuf,
    #[arg(long)]
    pub slices_dir: Option<PathBuf>,
    #[arg(long, default_value = "1")]
    pub parallel: NonZeroUsize,
}

#[derive(Args)]
pub struct FullRunArgs {
    #[arg(long)]
    pub model_dir: PathBuf,
    #[arg(long)]
    pub input_file: Option<PathBuf>,
    #[arg(long)]
    pub slices_dir: Option<PathBuf>,
    #[arg(long)]
    pub layers: Option<String>,
    #[arg(long)]
    pub weights_as_inputs: bool,
    #[arg(long, default_value = "1")]
    pub parallel: NonZeroUsize,
    #[arg(long)]
    pub batch: bool,
    #[arg(long, help = "Path to consumer ONNX with fine-tuned weights to inject at inference time")]
    pub weights: Option<PathBuf>,
    #[arg(long, default_value = "expander", help = "Proof system backend (expander or remainder)")]
    pub proof_system: String,
    #[arg(long, help = "Comma-separated ONNX op names to compile via the proof backend (default: all supported)")]
    pub circuit_ops: Option<String>,
}

fn resolve_circuit_ops(proof_system_str: &str, circuit_ops: Option<&str>) -> Result<Vec<String>> {
    let ps: ProofSystem = proof_system_str
        .parse()
        .map_err(|e: jstprove_circuits::ProofSystemParseError| DsperseError::Other(e.to_string()))?;

    let supported = ps.supported_ops();

    match circuit_ops {
        None => Ok(supported.iter().map(|s| (*s).to_string()).collect()),
        Some(spec) => {
            let requested: Vec<String> = spec.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
            for op in &requested {
                if !supported.contains(&op.as_str()) {
                    return Err(DsperseError::Other(format!(
                        "op {op:?} is not supported by proof system {ps}. Supported: {supported:?}"
                    )));
                }
            }
            Ok(requested)
        }
    }
}

pub fn cmd_slice(args: SliceArgs) -> Result<()> {
    let model_path = args.model_dir.join("model.onnx");
    if !model_path.exists() {
        return Err(DsperseError::Slicer(format!(
            "model.onnx not found in {}",
            args.model_dir.display()
        )));
    }
    let ops = resolve_circuit_ops(&args.proof_system, args.circuit_ops.as_deref())?;
    let ops_refs: Vec<&str> = ops.iter().map(String::as_str).collect();
    let metadata =
        crate::slicer::slice_model(&model_path, args.output_dir.as_deref(), args.tile_size, &ops_refs)?;
    tracing::info!(slices = metadata.slices.len(), "slicing complete");
    Ok(())
}

pub fn cmd_compile(args: CompileArgs) -> Result<()> {
    let backend = JstproveBackend::default();
    let slices_dir = args
        .slices_dir
        .unwrap_or_else(|| args.model_dir.join("slices"));

    let layers = args
        .layers
        .as_ref()
        .map(|s| parse_index_spec(s))
        .transpose()?;

    let ops = resolve_circuit_ops(&args.proof_system, args.circuit_ops.as_deref())?;
    let ops_refs: Vec<&str> = ops.iter().map(String::as_str).collect();

    pipeline::compile_slices(
        &slices_dir,
        &backend,
        args.parallel.get(),
        args.weights_as_inputs,
        layers.as_deref(),
        &ops_refs,
    )
}

pub fn cmd_run(args: RunArgs) -> Result<()> {
    if !args.input_file.is_file() {
        return Err(DsperseError::Other(format!(
            "input file not found: {}",
            args.input_file.display()
        )));
    }

    let backend = JstproveBackend::default();
    let slices_dir = args
        .slices_dir
        .unwrap_or_else(|| args.model_dir.join("slices"));

    let run_dir = args
        .run_dir
        .unwrap_or_else(|| args.model_dir.join("run").join(format!("run_{}", run_id())));

    let config = RunConfig {
        parallel: args.parallel.get(),
        batch: args.batch,
        weights_onnx: args.weights,
    };

    pipeline::run_inference(&slices_dir, &args.input_file, &run_dir, &backend, &config)?;
    Ok(())
}

pub fn cmd_prove(args: ProveArgs) -> Result<()> {
    let backend = JstproveBackend::default();
    let slices_dir = args
        .slices_dir
        .unwrap_or_else(|| args.model_dir.join("slices"));

    pipeline::prove_run(
        &args.run_dir,
        &slices_dir,
        &backend,
        args.parallel.get(),
    )?;
    Ok(())
}

pub fn cmd_verify(args: VerifyArgs) -> Result<()> {
    let backend = JstproveBackend::default();
    let slices_dir = args
        .slices_dir
        .unwrap_or_else(|| args.model_dir.join("slices"));

    pipeline::verify_run(&args.run_dir, &slices_dir, &backend, args.parallel.get())?;
    Ok(())
}

pub fn cmd_full_run(args: FullRunArgs) -> Result<()> {
    let backend = JstproveBackend::default();

    let slices_dir = args
        .slices_dir
        .unwrap_or_else(|| args.model_dir.join("slices"));

    let input_file = args
        .input_file
        .unwrap_or_else(|| args.model_dir.join(crate::utils::paths::INPUT_FILE));

    if !input_file.is_file() {
        return Err(DsperseError::Other(format!(
            "input file not found: {}",
            input_file.display()
        )));
    }

    if args.weights.is_some() && !args.weights_as_inputs {
        return Err(DsperseError::Other(
            "--weights requires --weights-as-inputs during compilation".into(),
        ));
    }

    let layers = args
        .layers
        .as_ref()
        .map(|s| parse_index_spec(s))
        .transpose()?;

    let ops = resolve_circuit_ops(&args.proof_system, args.circuit_ops.as_deref())?;
    let ops_refs: Vec<&str> = ops.iter().map(String::as_str).collect();

    tracing::info!("compiling slices");
    pipeline::compile_slices(
        &slices_dir,
        &backend,
        args.parallel.get(),
        args.weights_as_inputs,
        layers.as_deref(),
        &ops_refs,
    )?;

    let run_dir = args.model_dir.join("run").join(format!("run_{}", run_id()));

    let config = RunConfig {
        parallel: args.parallel.get(),
        batch: args.batch,
        weights_onnx: args.weights,
    };

    tracing::info!("running inference");
    pipeline::run_inference(&slices_dir, &input_file, &run_dir, &backend, &config)?;

    tracing::info!("proving");
    pipeline::prove_run(&run_dir, &slices_dir, &backend, args.parallel.get())?;

    tracing::info!("verifying");
    pipeline::verify_run(&run_dir, &slices_dir, &backend, args.parallel.get())?;

    tracing::info!(run_dir = %run_dir.display(), "full run complete");
    Ok(())
}

fn parse_index_spec(spec: &str) -> Result<Vec<usize>> {
    let mut layers = Vec::new();
    for part in spec.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        if let Some((start, end)) = part.split_once('-') {
            let s: usize = start.trim().parse().map_err(|_| {
                DsperseError::Other(format!("invalid index spec range start: {start:?}"))
            })?;
            let e: usize = end.trim().parse().map_err(|_| {
                DsperseError::Other(format!("invalid index spec range end: {end:?}"))
            })?;
            if s > e {
                return Err(DsperseError::Other(format!(
                    "invalid index spec range: start {s} > end {e}"
                )));
            }
            layers.extend(s..=e);
        } else {
            let n: usize = part
                .parse()
                .map_err(|_| DsperseError::Other(format!("invalid index spec token: {part:?}")))?;
            layers.push(n);
        }
    }
    if layers.is_empty() {
        return Err(DsperseError::Other("empty index spec".into()));
    }
    Ok(layers)
}

fn run_id() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let uuid = uuid::Uuid::new_v4();
    format!("{}_{}", now.as_secs(), uuid.as_simple())
}
