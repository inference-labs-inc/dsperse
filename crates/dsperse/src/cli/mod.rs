use std::path::PathBuf;

use clap::Args;

use crate::archive::converter::{self, FormatType};
use crate::backend::JstproveBackend;
use crate::error::{DsperseError, Result};
use crate::pipeline::{self, RunConfig};

#[derive(Args)]
pub struct SliceArgs {
    #[arg(long)]
    pub model_dir: PathBuf,
    #[arg(long)]
    pub output_dir: Option<PathBuf>,
    #[arg(long, default_value = "dirs")]
    pub format: String,
    #[arg(long)]
    pub tile_size: Option<usize>,
}

#[derive(Args)]
pub struct CompileArgs {
    #[arg(long)]
    pub model_dir: PathBuf,
    #[arg(long)]
    pub input_file: Option<PathBuf>,
    #[arg(long)]
    pub layers: Option<String>,
    #[arg(long, default_value_t = 1)]
    pub parallel: usize,
    #[arg(long)]
    pub weights_as_inputs: bool,
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
    pub backend: Option<String>,
    #[arg(long, default_value_t = 1)]
    pub parallel: usize,
    #[arg(long)]
    pub batch: bool,
}

#[derive(Args)]
pub struct ProveArgs {
    #[arg(long)]
    pub run_dir: PathBuf,
    #[arg(long)]
    pub model_dir: PathBuf,
    #[arg(long)]
    pub backend: Option<String>,
    #[arg(long, default_value_t = 1)]
    pub parallel: usize,
    #[arg(long)]
    pub tiles: Option<String>,
}

#[derive(Args)]
pub struct VerifyArgs {
    #[arg(long)]
    pub run_dir: PathBuf,
    #[arg(long)]
    pub model_dir: PathBuf,
    #[arg(long)]
    pub backend: Option<String>,
    #[arg(long, default_value_t = 1)]
    pub parallel: usize,
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
    #[arg(long, default_value_t = 1)]
    pub parallel: usize,
    #[arg(long)]
    pub batch: bool,
}

#[derive(Args)]
pub struct ConvertArgs {
    #[arg(long)]
    pub input: PathBuf,
    #[arg(long)]
    pub to: String,
    #[arg(long)]
    pub output: Option<PathBuf>,
    #[arg(long)]
    pub expand_slices: bool,
    #[arg(long)]
    pub cleanup: bool,
}

pub fn cmd_slice(args: SliceArgs) -> Result<()> {
    let model_path = args.model_dir.join("model.onnx");
    if !model_path.exists() {
        return Err(DsperseError::Slicer(format!(
            "model.onnx not found in {}",
            args.model_dir.display()
        )));
    }
    let metadata = crate::slicer::slice_model(
        &model_path,
        args.output_dir.as_deref(),
        args.tile_size,
    )?;
    tracing::info!(slices = metadata.slices.len(), "slicing complete");
    Ok(())
}

pub fn cmd_compile(args: CompileArgs) -> Result<()> {
    let backend = JstproveBackend::new();
    let slices_dir = args.model_dir.join("slices");

    let layers: Option<Vec<usize>> = args.layers.as_ref().map(|s| parse_layer_spec(s));

    pipeline::compile_slices(
        &slices_dir,
        &backend,
        args.parallel,
        args.weights_as_inputs,
        layers.as_deref(),
    )
}

pub fn cmd_run(args: RunArgs) -> Result<()> {
    let backend = JstproveBackend::new();
    let slices_dir = args.model_dir.join("slices");

    let run_dir = args.run_dir.unwrap_or_else(|| {
        let ts = chrono_timestamp();
        args.model_dir.join("run").join(format!("run_{ts}"))
    });

    let config = RunConfig {
        parallel: args.parallel,
        batch: args.batch,
    };

    pipeline::run_inference(&slices_dir, &args.input_file, &run_dir, &backend, &config)?;
    Ok(())
}

pub fn cmd_prove(args: ProveArgs) -> Result<()> {
    let backend = JstproveBackend::new();
    let slices_dir = args.model_dir.join("slices");

    let tiles: Option<Vec<usize>> = args.tiles.as_ref().map(|s| parse_layer_spec(s));

    pipeline::prove_run(
        &args.run_dir,
        &slices_dir,
        &backend,
        args.parallel,
        tiles.as_deref(),
    )?;
    Ok(())
}

pub fn cmd_verify(args: VerifyArgs) -> Result<()> {
    let backend = JstproveBackend::new();
    let slices_dir = args.model_dir.join("slices");

    pipeline::verify_run(&args.run_dir, &slices_dir, &backend, args.parallel)?;
    Ok(())
}

pub fn cmd_full_run(args: FullRunArgs) -> Result<()> {
    let backend = JstproveBackend::new();

    let slices_dir = args
        .slices_dir
        .unwrap_or_else(|| args.model_dir.join("slices"));

    let input_file = args
        .input_file
        .unwrap_or_else(|| args.model_dir.join("input.json"));

    let layers: Option<Vec<usize>> = args.layers.as_ref().map(|s| parse_layer_spec(s));

    tracing::info!("compiling slices");
    pipeline::compile_slices(
        &slices_dir,
        &backend,
        args.parallel,
        args.weights_as_inputs,
        layers.as_deref(),
    )?;

    let ts = chrono_timestamp();
    let run_dir = args.model_dir.join("run").join(format!("run_{ts}"));

    let config = RunConfig {
        parallel: args.parallel,
        batch: args.batch,
    };

    tracing::info!("running inference");
    pipeline::run_inference(&slices_dir, &input_file, &run_dir, &backend, &config)?;

    tracing::info!("proving");
    pipeline::prove_run(&run_dir, &slices_dir, &backend, args.parallel, None)?;

    tracing::info!("verifying");
    pipeline::verify_run(&run_dir, &slices_dir, &backend, args.parallel)?;

    tracing::info!(run_dir = %run_dir.display(), "full run complete");
    Ok(())
}

pub fn cmd_convert(args: ConvertArgs) -> Result<()> {
    let target = match args.to.as_str() {
        "dirs" => FormatType::Dirs,
        "dslice" => FormatType::Dslice,
        "dsperse" => FormatType::Dsperse,
        other => return Err(DsperseError::Other(format!("unknown format: {other}"))),
    };

    converter::convert(&args.input, target, args.output.as_deref(), args.cleanup)?;
    Ok(())
}

fn parse_layer_spec(spec: &str) -> Vec<usize> {
    let mut layers = Vec::new();
    for part in spec.split(',') {
        let part = part.trim();
        if let Some((start, end)) = part.split_once('-') {
            if let (Ok(s), Ok(e)) = (start.trim().parse::<usize>(), end.trim().parse::<usize>()) {
                layers.extend(s..=e);
            }
        } else if let Ok(n) = part.parse::<usize>() {
            layers.push(n);
        }
    }
    layers
}

fn chrono_timestamp() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    format!("{}", now.as_secs())
}
