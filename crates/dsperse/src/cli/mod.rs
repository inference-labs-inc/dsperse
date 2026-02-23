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
    #[arg(long, value_enum, default_value = "dirs")]
    pub format: FormatType,
    #[arg(long)]
    pub tile_size: Option<usize>,
}

#[derive(Args)]
pub struct CompileArgs {
    #[arg(long)]
    pub model_dir: PathBuf,
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
    pub slices_dir: Option<PathBuf>,
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
    pub slices_dir: Option<PathBuf>,
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
    pub slices_dir: Option<PathBuf>,
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
    #[arg(long, value_enum)]
    pub to: FormatType,
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

    if args.format != FormatType::Dirs {
        let default_dir = args.model_dir.join("slices");
        let slices_dir = args.output_dir.as_deref().unwrap_or(&default_dir);
        converter::convert(slices_dir, args.format, None, false, true)?;
    }

    Ok(())
}

pub fn cmd_compile(args: CompileArgs) -> Result<()> {
    let backend = JstproveBackend::default();
    let slices_dir = args.model_dir.join("slices");

    let layers = args.layers.as_ref().map(|s| parse_layer_spec(s)).transpose()?;

    pipeline::compile_slices(
        &slices_dir,
        &backend,
        args.parallel,
        args.weights_as_inputs,
        layers.as_deref(),
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
    let slices_dir = args.slices_dir.unwrap_or_else(|| args.model_dir.join("slices"));

    let run_dir = args.run_dir.unwrap_or_else(|| {
        args.model_dir.join("run").join(format!("run_{}", run_id()))
    });

    let config = RunConfig {
        parallel: args.parallel,
        batch: args.batch,
    };

    pipeline::run_inference(&slices_dir, &args.input_file, &run_dir, &backend, &config)?;
    Ok(())
}

pub fn cmd_prove(args: ProveArgs) -> Result<()> {
    let backend = JstproveBackend::default();
    let slices_dir = args.slices_dir.unwrap_or_else(|| args.model_dir.join("slices"));

    let tiles = args.tiles.as_ref().map(|s| parse_layer_spec(s)).transpose()?;

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
    let backend = JstproveBackend::default();
    let slices_dir = args.slices_dir.unwrap_or_else(|| args.model_dir.join("slices"));

    pipeline::verify_run(&args.run_dir, &slices_dir, &backend, args.parallel)?;
    Ok(())
}

pub fn cmd_full_run(args: FullRunArgs) -> Result<()> {
    let backend = JstproveBackend::default();

    let slices_dir = args
        .slices_dir
        .unwrap_or_else(|| args.model_dir.join("slices"));

    let input_file = args
        .input_file
        .unwrap_or_else(|| args.model_dir.join("input.json"));

    if !input_file.is_file() {
        return Err(DsperseError::Other(format!(
            "input file not found: {}",
            input_file.display()
        )));
    }

    let layers = args.layers.as_ref().map(|s| parse_layer_spec(s)).transpose()?;

    tracing::info!("compiling slices");
    pipeline::compile_slices(
        &slices_dir,
        &backend,
        args.parallel,
        args.weights_as_inputs,
        layers.as_deref(),
    )?;

    let run_dir = args.model_dir.join("run").join(format!("run_{}", run_id()));

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
    converter::convert(&args.input, args.to, args.output.as_deref(), args.cleanup, args.expand_slices)?;
    Ok(())
}

fn parse_layer_spec(spec: &str) -> Result<Vec<usize>> {
    let mut layers = Vec::new();
    for part in spec.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        if let Some((start, end)) = part.split_once('-') {
            let s: usize = start.trim().parse().map_err(|_| {
                DsperseError::Other(format!("invalid layer spec range start: {start:?}"))
            })?;
            let e: usize = end.trim().parse().map_err(|_| {
                DsperseError::Other(format!("invalid layer spec range end: {end:?}"))
            })?;
            if s > e {
                return Err(DsperseError::Other(format!(
                    "invalid layer spec range: start {s} > end {e}"
                )));
            }
            layers.extend(s..=e);
        } else {
            let n: usize = part.parse().map_err(|_| {
                DsperseError::Other(format!("invalid layer spec token: {part:?}"))
            })?;
            layers.push(n);
        }
    }
    Ok(layers)
}

fn run_id() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let uuid = uuid::Uuid::new_v4();
    format!("{}_{}", now.as_secs(), uuid)
}
