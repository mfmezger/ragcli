mod app;
mod cli;
mod commands;
mod config;
mod fsutil;
mod ingest;
mod models;
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
