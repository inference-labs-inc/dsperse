use clap::{Parser, Subcommand};
use tracing_subscriber::EnvFilter;

use dsperse::cli;

#[derive(Parser)]
#[command(name = "dsperse", about = "Distributed zkML Toolkit")]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    #[arg(long, default_value = "warn", global = true)]
    log_level: String,
}

#[derive(Subcommand)]
enum Commands {
    Slice(cli::SliceArgs),
    Compile(cli::CompileArgs),
    Run(cli::RunArgs),
    Prove(cli::ProveArgs),
    Verify(cli::VerifyArgs),
    #[command(name = "full-run")]
    FullRun(cli::FullRunArgs),
    Convert(cli::ConvertArgs),
}

fn main() {
    let cli = Cli::parse();

    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new(&cli.log_level)),
        )
        .init();

    let result = match cli.command {
        Commands::Slice(args) => cli::cmd_slice(args),
        Commands::Compile(args) => cli::cmd_compile(args),
        Commands::Run(args) => cli::cmd_run(args),
        Commands::Prove(args) => cli::cmd_prove(args),
        Commands::Verify(args) => cli::cmd_verify(args),
        Commands::FullRun(args) => cli::cmd_full_run(args),
        Commands::Convert(args) => cli::cmd_convert(args),
    };

    if let Err(e) = result {
        tracing::error!("{e}");
        std::process::exit(1);
    }
}
