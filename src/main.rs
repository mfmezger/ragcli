mod agent;
mod app;
mod citation;
mod cli;
mod commands;
mod config;
mod fsutil;
mod ingest;
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

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    app::run(cli).await
}
