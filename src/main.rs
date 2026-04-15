mod agent;
mod app;
mod citation;
mod cli;
mod commands;
mod config;
mod fsutil;
mod graph;
mod ingest;
mod jsonutil;
mod models;
mod retrieval;
mod rewrite;
mod source_kind;
mod store;
#[cfg(test)]
mod test_support;

use anyhow::Result;
use clap::Parser;
use cli::Cli;
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

static INIT: std::sync::Once = std::sync::Once::new();

fn init_tracing() {
    INIT.call_once(|| {
        let filter = EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| EnvFilter::new("info"));
        tracing_subscriber::registry()
            .with(fmt::layer().with_writer(std::io::stderr))
            .with(filter)
            .init();
    });
}

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing();
    let cli = Cli::parse();
    app::run(cli).await
}
