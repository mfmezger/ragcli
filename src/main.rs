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
mod query;
mod retrieval;
mod rewrite;
mod source_kind;
mod store;
mod telemetry;
#[cfg(test)]
mod test_support;

use anyhow::Result;
use clap::Parser;
use cli::Cli;

fn main() -> Result<()> {
    let mut telemetry = telemetry::init()?;
    let cli = Cli::parse();
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    let result = runtime.block_on(app::run(cli));
    drop(runtime);

    if let Err(err) = telemetry.shutdown() {
        eprintln!("telemetry shutdown error: {err}");
    }

    result
}
