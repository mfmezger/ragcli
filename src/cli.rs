use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "ragcli", about = "Local RAG CLI")]
pub struct Cli {
    /// Store name under ~/.config/ragcli (default: default)
    #[arg(long, global = true)]
    pub name: Option<String>,

    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Index a folder into the local store
    Index {
        path: PathBuf,
        /// Override chunk size (chars)
        #[arg(long)]
        chunk_size: Option<usize>,
        /// Override chunk overlap (chars)
        #[arg(long)]
        chunk_overlap: Option<usize>,
        /// Override embedding model path
        #[arg(long)]
        embed_model: Option<String>,
    },
    /// Query the local store
    Query {
        question: String,
        /// Top-K results to retrieve
        #[arg(long, default_value_t = 5)]
        top_k: usize,
        /// Print retrieved context snippets before answering
        #[arg(long, default_value_t = false)]
        show_context: bool,
        /// Override generation model path
        #[arg(long)]
        gen_model: Option<String>,
        /// Max tokens to generate
        #[arg(long, default_value_t = 256)]
        max_tokens: usize,
    },
    /// Show or update config values
    Config {
        #[command(subcommand)]
        command: ConfigCommand,
    },
    /// Show a summary of indexed content and store usage
    Stat,
    /// Check store layout and status
    Doctor,
}

#[derive(Subcommand, Debug)]
pub enum ConfigCommand {
    /// Print the effective config for this store
    Show,
    /// Set a config key in ~/.config/ragcli/<name>/config.toml
    Set {
        /// Config key such as models.embed or ollama.base_url
        key: String,
        /// New value to write
        value: String,
    },
}
