use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};

use clap::{Args, Parser, Subcommand};

use crate::backend::jstprove::{Curve, JstproveBackend};
use crate::error::{DsperseError, Result};
use crate::pipeline::{self, RunConfig};

use jstprove_circuits::api::{CurveParseError, ProofSystemType as ProofSystem};

fn parse_curve(curve: &str) -> Result<Curve> {
    curve
        .parse()
        .map_err(|e: CurveParseError| DsperseError::Other(e.to_string()))
}

pub const VERSION: &str = env!("DSPERSE_DISPLAY_VERSION");

#[derive(Parser)]
#[command(name = "dsperse", about = "Distributed zkML Toolkit", version = VERSION)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
    #[arg(long, default_value = "warn", global = true)]
    pub log_level: String,
}

#[derive(Subcommand)]
pub enum Commands {
    Slice(SliceArgs),
    Combine(CombineArgs),
    Compile(CompileArgs),
    Run(RunArgs),
    Prove(ProveArgs),
    Verify(VerifyArgs),
    Package(PackageArgs),
    Publish(PublishArgs),
    #[command(name = "full-run")]
    FullRun(FullRunArgs),
}

pub fn dispatch(command: Commands) -> Result<()> {
    match command {
        Commands::Slice(args) => cmd_slice(args),
        Commands::Combine(args) => cmd_combine(args),
        Commands::Compile(args) => cmd_compile(args),
        Commands::Run(args) => cmd_run(args),
        Commands::Prove(args) => cmd_prove(args),
        Commands::Verify(args) => cmd_verify(args),
        Commands::Package(args) => cmd_package(args),
        Commands::Publish(args) => cmd_publish(args),
        Commands::FullRun(args) => cmd_full_run(args),
    }
}

#[derive(Args)]
pub struct SliceArgs {
    #[arg(long)]
    pub model_dir: PathBuf,
    #[arg(long)]
    pub output_dir: Option<PathBuf>,
    #[arg(long, default_value = "512")]
    pub tile_size: Option<usize>,
    #[arg(
        long,
        default_value = "expander",
        help = "Proof system backend (expander or remainder)"
    )]
    pub proof_system: String,
    #[arg(
        long,
        help = "Comma-separated ONNX op names to compile via the proof backend (default: all supported)"
    )]
    pub circuit_ops: Option<String>,
    #[arg(
        long,
        value_delimiter = ',',
        help = "Concrete input shape as comma-separated dims (e.g. 1,3,560,560). Overrides dynamic dimensions."
    )]
    pub input_shape: Option<Vec<i64>>,
}

#[derive(Args)]
pub struct CombineArgs {
    #[arg(long)]
    pub model_dir: PathBuf,
    #[arg(long)]
    pub slices_dir: Option<PathBuf>,
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
    #[arg(
        long,
        default_value_t = true,
        action = clap::ArgAction::Set,
        help = "Compile circuits with weights as inputs for shared circuit reuse (default: true)"
    )]
    pub weights_as_inputs: bool,
    #[arg(
        long,
        default_value = "expander",
        help = "Proof system backend (expander or remainder)"
    )]
    pub proof_system: String,
    #[arg(
        long,
        help = "Comma-separated ONNX op names to compile via the proof backend (default: all supported)"
    )]
    pub circuit_ops: Option<String>,
    #[arg(
        long,
        default_value = "bn254",
        help = "Finite field (bn254, goldilocks, goldilocks_basefold)"
    )]
    pub curve: String,
    #[arg(
        long,
        help = "Skip compilation of slices whose estimated constraint count exceeds this threshold"
    )]
    pub skip_compile_over_size: Option<u64>,
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
    #[arg(
        long,
        help = "Path to consumer ONNX with fine-tuned weights to inject at inference time"
    )]
    pub weights: Option<PathBuf>,
    #[arg(
        long,
        default_value_t = true,
        action = clap::ArgAction::Set,
        help = "Run inference on combined monolithic ONNX instead of per-slice execution"
    )]
    pub combined: bool,
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
pub struct PackageArgs {
    #[arg(long)]
    pub model_dir: PathBuf,
    #[arg(long)]
    pub slices_dir: Option<PathBuf>,
    #[arg(long)]
    pub output_dir: Option<PathBuf>,
    #[arg(long)]
    pub author: Option<String>,
    #[arg(long)]
    pub model_version: Option<String>,
    #[arg(long)]
    pub model_name: Option<String>,
    #[arg(long)]
    pub timeout: Option<u64>,
    #[arg(
        long,
        help = "Finite field curve used as domain separator in content hashes (bn254, goldilocks, goldilocks_basefold)"
    )]
    pub curve: Option<String>,
}

#[derive(Args)]
pub struct PublishArgs {
    #[arg(long, help = "Package directory containing manifest.msgpack")]
    pub dir: PathBuf,
    #[arg(long, help = "Registry base URL")]
    pub url: String,
    #[arg(long, env = "REGISTRY_AUTH_TOKEN", hide_env_values = true)]
    pub auth_token: String,
    #[arg(long)]
    pub name: String,
    #[arg(long, default_value = "")]
    pub description: String,
    #[arg(long)]
    pub author: String,
    #[arg(long, default_value = "1.0.0")]
    pub version: String,
    #[arg(long, default_value = "JSTPROVE")]
    pub proof_system: String,
    #[arg(long, default_value = "3600")]
    pub timeout: u64,
    #[arg(long, default_value_t = false, help = "Activate model after upload")]
    pub activate: bool,
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
    #[arg(
        long,
        default_value_t = true,
        action = clap::ArgAction::Set,
        help = "Compile circuits with weights as inputs for shared circuit reuse (default: true)"
    )]
    pub weights_as_inputs: bool,
    #[arg(long, default_value = "1")]
    pub parallel: NonZeroUsize,
    #[arg(long)]
    pub batch: bool,
    #[arg(
        long,
        help = "Path to consumer ONNX with fine-tuned weights to inject at inference time"
    )]
    pub weights: Option<PathBuf>,
    #[arg(
        long,
        default_value = "expander",
        help = "Proof system backend (expander or remainder)"
    )]
    pub proof_system: String,
    #[arg(
        long,
        help = "Comma-separated ONNX op names to compile via the proof backend (default: all supported)"
    )]
    pub circuit_ops: Option<String>,
    #[arg(
        long,
        default_value_t = true,
        action = clap::ArgAction::Set,
        help = "Run inference on combined monolithic ONNX instead of per-slice execution"
    )]
    pub combined: bool,
    #[arg(
        long,
        default_value = "bn254",
        help = "Finite field (bn254, goldilocks, goldilocks_basefold)"
    )]
    pub curve: String,
    #[arg(
        long,
        help = "Skip compilation of slices whose estimated constraint count exceeds this threshold"
    )]
    pub skip_compile_over_size: Option<u64>,
}

struct CircuitOps(Vec<String>);

impl CircuitOps {
    fn as_refs(&self) -> Vec<&str> {
        self.0.iter().map(String::as_str).collect()
    }
}

fn resolve_circuit_ops(proof_system_str: &str, circuit_ops: Option<&str>) -> Result<CircuitOps> {
    let ps: ProofSystem =
        proof_system_str
            .parse()
            .map_err(|e: jstprove_circuits::api::ProofSystemParseError| {
                DsperseError::Other(e.to_string())
            })?;

    let supported = ps.supported_ops();

    let ops = match circuit_ops {
        None => supported.iter().map(|s| (*s).to_string()).collect(),
        Some(spec) => {
            let requested: Vec<String> = spec
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            if requested.is_empty() {
                return Err(DsperseError::Other(
                    "empty --circuit-ops; provide at least one op or omit the flag to use all supported ops".into(),
                ));
            }
            for op in &requested {
                if !supported.contains(&op.as_str()) {
                    return Err(DsperseError::Other(format!(
                        "op {op:?} is not supported by proof system {ps}. Supported: {supported:?}"
                    )));
                }
            }
            requested
        }
    };
    Ok(CircuitOps(ops))
}

fn resolve_slices_dir(slices_dir: Option<PathBuf>, model_dir: &Path) -> PathBuf {
    slices_dir.unwrap_or_else(|| model_dir.join("slices"))
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
    let metadata = crate::slicer::slice_model(
        &model_path,
        args.output_dir.as_deref(),
        args.tile_size,
        &ops.as_refs(),
        args.input_shape.as_deref(),
    )?;
    tracing::info!(slices = metadata.slices.len(), "slicing complete");
    Ok(())
}

pub fn cmd_combine(args: CombineArgs) -> Result<()> {
    let slices_dir = resolve_slices_dir(args.slices_dir, &args.model_dir);
    let meta = pipeline::runner::load_model_metadata(&slices_dir)?;
    let path = crate::slicer::combiner::materialize_combined_to_disk(&slices_dir, &meta)?;
    tracing::info!(path = %path.display(), "combined ONNX materialized");
    Ok(())
}

pub fn cmd_compile(args: CompileArgs) -> Result<()> {
    let curve = parse_curve(&args.curve)?;
    let backend = JstproveBackend::new();
    let slices_dir = resolve_slices_dir(args.slices_dir, &args.model_dir);

    let layers = args
        .layers
        .as_ref()
        .map(|s| parse_index_spec(s))
        .transpose()?;

    let ops = resolve_circuit_ops(&args.proof_system, args.circuit_ops.as_deref())?;

    pipeline::compile_slices(
        &slices_dir,
        &backend,
        curve,
        args.parallel.get(),
        args.weights_as_inputs,
        layers.as_deref(),
        &ops.as_refs(),
        args.skip_compile_over_size,
    )
}

pub fn cmd_run(args: RunArgs) -> Result<()> {
    if !args.input_file.is_file() {
        return Err(DsperseError::Other(format!(
            "input file not found: {}",
            args.input_file.display()
        )));
    }

    let backend = JstproveBackend::new();
    let slices_dir = resolve_slices_dir(args.slices_dir, &args.model_dir);

    let run_dir = args
        .run_dir
        .unwrap_or_else(|| args.model_dir.join("run").join(format!("run_{}", run_id())));

    let config = RunConfig {
        parallel: args.parallel.get(),
        batch: args.batch,
        weights_onnx: args.weights,
        combined: args.combined,
    };

    pipeline::run_inference(&slices_dir, &args.input_file, &run_dir, &backend, &config)?;
    Ok(())
}

pub fn cmd_prove(args: ProveArgs) -> Result<()> {
    let backend = JstproveBackend::new();
    let slices_dir = resolve_slices_dir(args.slices_dir, &args.model_dir);

    pipeline::prove_run(&args.run_dir, &slices_dir, &backend, args.parallel.get())?;
    Ok(())
}

pub fn cmd_verify(args: VerifyArgs) -> Result<()> {
    let backend = JstproveBackend::new();
    let slices_dir = resolve_slices_dir(args.slices_dir, &args.model_dir);

    pipeline::verify_run(&args.run_dir, &slices_dir, &backend, args.parallel.get())?;
    Ok(())
}

pub fn cmd_package(args: PackageArgs) -> Result<()> {
    let slices_dir = resolve_slices_dir(args.slices_dir, &args.model_dir);
    let output_dir = args
        .output_dir
        .unwrap_or_else(|| args.model_dir.join("package"));

    let config = pipeline::packager::PackageConfig {
        output_dir,
        author: args.author,
        model_version: args.model_version,
        model_name: args.model_name,
        timeout: args.timeout,
        curve: args.curve,
    };

    let result = pipeline::packager::package_content_addressed(&slices_dir, &config)?;

    tracing::info!(
        components = result.component_count,
        weight_biases = result.wb_count,
        total_bytes = result.total_size,
        manifest = %result.manifest_path.display(),
        "content-addressed packaging complete"
    );

    Ok(())
}

pub fn cmd_publish(args: PublishArgs) -> Result<()> {
    let config = pipeline::publisher::PublishConfig {
        api_url: args.url,
        auth_token: args.auth_token,
        name: args.name,
        description: args.description,
        author: args.author,
        version: args.version,
        proof_system: args.proof_system,
        timeout: args.timeout,
        activate: args.activate,
    };

    let result = match pipeline::publisher::publish(&args.dir, &config) {
        Ok(r) => r,
        Err(e) => {
            tracing::error!(error = %e, "publish failed");
            return Err(e);
        }
    };

    tracing::info!(
        model_id = %result.model_id,
        components_uploaded = result.components_uploaded,
        components_skipped = result.components_skipped,
        weights_uploaded = result.weights_uploaded,
        weights_skipped = result.weights_skipped,
        "publish complete"
    );

    Ok(())
}

pub fn cmd_full_run(args: FullRunArgs) -> Result<()> {
    let curve = parse_curve(&args.curve)?;
    let backend = JstproveBackend::new();

    let slices_dir = resolve_slices_dir(args.slices_dir, &args.model_dir);

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

    tracing::info!("compiling slices");
    pipeline::compile_slices(
        &slices_dir,
        &backend,
        curve,
        args.parallel.get(),
        args.weights_as_inputs,
        layers.as_deref(),
        &ops.as_refs(),
        args.skip_compile_over_size,
    )?;

    let run_dir = args.model_dir.join("run").join(format!("run_{}", run_id()));

    let config = RunConfig {
        parallel: args.parallel.get(),
        batch: args.batch,
        weights_onnx: args.weights,
        combined: args.combined,
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

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn parse_index_spec_single() {
        assert_eq!(parse_index_spec("3").unwrap(), vec![3]);
    }

    #[test]
    fn parse_index_spec_multiple() {
        assert_eq!(parse_index_spec("1,3,5").unwrap(), vec![1, 3, 5]);
    }

    #[test]
    fn parse_index_spec_range() {
        assert_eq!(parse_index_spec("2-5").unwrap(), vec![2, 3, 4, 5]);
    }

    #[test]
    fn parse_index_spec_mixed() {
        assert_eq!(parse_index_spec("0,2-4,7").unwrap(), vec![0, 2, 3, 4, 7]);
    }

    #[test]
    fn parse_index_spec_whitespace_tolerance() {
        assert_eq!(parse_index_spec(" 1 , 2 - 3 ").unwrap(), vec![1, 2, 3]);
    }

    #[test]
    fn parse_index_spec_empty_rejected() {
        assert!(parse_index_spec("").is_err());
    }

    #[test]
    fn parse_index_spec_invalid_token() {
        assert!(parse_index_spec("abc").is_err());
    }

    #[test]
    fn parse_index_spec_reversed_range() {
        assert!(parse_index_spec("5-2").is_err());
    }

    #[test]
    fn parse_index_spec_trailing_comma() {
        assert_eq!(parse_index_spec("1,2,").unwrap(), vec![1, 2]);
    }

    #[test]
    fn run_id_format() {
        let id = run_id();
        let parts: Vec<&str> = id.splitn(2, '_').collect();
        assert_eq!(parts.len(), 2);
        assert!(parts[0].parse::<u64>().is_ok());
        assert_eq!(parts[1].len(), 32);
    }

    #[test]
    fn run_id_unique() {
        let id1 = run_id();
        let id2 = run_id();
        assert_ne!(id1, id2);
    }

    #[test]
    fn cli_parse_slice_command() {
        let cli = Cli::parse_from(["dsperse", "slice", "--model-dir", "/tmp/model"]);
        assert!(matches!(cli.command, Commands::Slice(_)));
    }

    #[test]
    fn cli_parse_run_command() {
        let cli = Cli::parse_from([
            "dsperse",
            "run",
            "--model-dir",
            "/tmp/model",
            "--input-file",
            "/tmp/input.json",
        ]);
        assert!(matches!(cli.command, Commands::Run(_)));
    }

    #[test]
    fn cli_log_level_default() {
        let cli = Cli::parse_from(["dsperse", "slice", "--model-dir", "/tmp"]);
        assert_eq!(cli.log_level, "warn");
    }

    #[test]
    fn cli_log_level_override() {
        let cli = Cli::parse_from([
            "dsperse",
            "--log-level",
            "debug",
            "slice",
            "--model-dir",
            "/tmp",
        ]);
        assert_eq!(cli.log_level, "debug");
    }

    #[test]
    fn cli_compile_with_layers() {
        let cli = Cli::parse_from([
            "dsperse",
            "compile",
            "--model-dir",
            "/tmp",
            "--layers",
            "0,2-4",
        ]);
        if let Commands::Compile(args) = cli.command {
            assert_eq!(args.layers.as_deref(), Some("0,2-4"));
        } else {
            panic!("expected Compile");
        }
    }

    #[test]
    fn cli_run_parallel() {
        let cli = Cli::parse_from([
            "dsperse",
            "run",
            "--model-dir",
            "/tmp",
            "--input-file",
            "/tmp/in.json",
            "--parallel",
            "4",
        ]);
        if let Commands::Run(args) = cli.command {
            assert_eq!(args.parallel.get(), 4);
        } else {
            panic!("expected Run");
        }
    }

    #[test]
    fn cli_slice_with_tile_size() {
        let cli = Cli::parse_from([
            "dsperse",
            "slice",
            "--model-dir",
            "/tmp",
            "--tile-size",
            "1024",
        ]);
        if let Commands::Slice(args) = cli.command {
            assert_eq!(args.tile_size, Some(1024));
        } else {
            panic!("expected Slice");
        }
    }

    #[test]
    fn cli_parse_combine_command() {
        let cli = Cli::parse_from(["dsperse", "combine", "--model-dir", "/tmp/model"]);
        assert!(matches!(cli.command, Commands::Combine(_)));
    }

    #[test]
    fn cli_parse_combine_with_slices_dir() {
        let cli = Cli::parse_from([
            "dsperse",
            "combine",
            "--model-dir",
            "/tmp/model",
            "--slices-dir",
            "/tmp/slices",
        ]);
        if let Commands::Combine(args) = cli.command {
            assert_eq!(
                args.slices_dir,
                Some(std::path::PathBuf::from("/tmp/slices"))
            );
        } else {
            panic!("expected Combine");
        }
    }

    #[test]
    fn cli_run_combined_default_true() {
        let cli = Cli::parse_from([
            "dsperse",
            "run",
            "--model-dir",
            "/tmp",
            "--input-file",
            "/tmp/in.json",
        ]);
        if let Commands::Run(args) = cli.command {
            assert!(args.combined);
        } else {
            panic!("expected Run");
        }
    }

    #[test]
    fn cli_run_combined_explicit_false() {
        let cli = Cli::parse_from([
            "dsperse",
            "run",
            "--model-dir",
            "/tmp",
            "--input-file",
            "/tmp/in.json",
            "--combined",
            "false",
        ]);
        if let Commands::Run(args) = cli.command {
            assert!(!args.combined);
        } else {
            panic!("expected Run");
        }
    }

    #[test]
    fn cli_compile_wai_default_true() {
        let cli = Cli::parse_from(["dsperse", "compile", "--model-dir", "/tmp"]);
        if let Commands::Compile(args) = cli.command {
            assert!(args.weights_as_inputs);
        } else {
            panic!("expected Compile");
        }
    }

    #[test]
    fn cli_compile_wai_explicit_false() {
        let cli = Cli::parse_from([
            "dsperse",
            "compile",
            "--model-dir",
            "/tmp",
            "--weights-as-inputs",
            "false",
        ]);
        if let Commands::Compile(args) = cli.command {
            assert!(!args.weights_as_inputs);
        } else {
            panic!("expected Compile");
        }
    }

    #[test]
    fn resolve_circuit_ops_invalid_proof_system() {
        let result = resolve_circuit_ops("nonexistent", None);
        assert!(result.is_err());
    }

    #[test]
    fn resolve_circuit_ops_unsupported_op() {
        let result = resolve_circuit_ops("expander", Some("FakeOp"));
        assert!(result.is_err());
    }

    #[test]
    fn resolve_circuit_ops_empty_spec_rejected() {
        let result = resolve_circuit_ops("expander", Some(""));
        assert!(result.is_err());
    }

    #[test]
    fn resolve_circuit_ops_whitespace_only_spec_rejected() {
        let result = resolve_circuit_ops("expander", Some(" ,  , "));
        assert!(result.is_err());
    }

    #[test]
    fn resolve_circuit_ops_valid_specific_ops() {
        let supported = ProofSystem::Expander.supported_ops();
        assert!(!supported.is_empty());
        let first_op = supported[0];
        let ops = resolve_circuit_ops("expander", Some(first_op)).unwrap();
        assert_eq!(ops.as_refs(), vec![first_op]);
    }

    #[test]
    fn resolve_circuit_ops_none_returns_all() {
        let ops = resolve_circuit_ops("expander", None).unwrap();
        let expected: Vec<&str> = ProofSystem::Expander.supported_ops().to_vec();
        assert_eq!(ops.as_refs(), expected);
    }

    #[test]
    fn resolve_slices_dir_custom_path() {
        let result = resolve_slices_dir(Some(PathBuf::from("/custom")), Path::new("/model"));
        assert_eq!(result, PathBuf::from("/custom"));
    }

    #[test]
    fn resolve_slices_dir_default_fallback() {
        let model_dir = Path::new("/model");
        let result = resolve_slices_dir(None, model_dir);
        assert_eq!(result, model_dir.join("slices"));
    }
}
