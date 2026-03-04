use clap::Parser;
use tracing_subscriber::EnvFilter;

use dsperse::cli;

fn main() {
    let parsed = cli::Cli::parse();

    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(&parsed.log_level)),
        )
        .init();

    if let Err(e) = cli::dispatch(parsed.command) {
        tracing::error!("{e}");
        std::process::exit(1);
    }
}
