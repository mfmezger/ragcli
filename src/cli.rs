//! Command-line interface definitions for `ragcli`.

use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

/// PDF extraction backend used during indexing.
#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
pub enum PdfParserArg {
    Native,
    Liteparse,
}

/// Top-level CLI arguments.
#[derive(Parser, Debug)]
#[command(name = "ragcli", about = "RAG CLI powered by Ollama.")]
pub struct Cli {
    /// Selects the store under `~/.config/ragcli` to operate on.
    #[arg(long, global = true)]
    pub name: Option<String>,

    /// Runs one of the supported subcommands.
    #[command(subcommand)]
    pub command: Command,
}

/// Supported `ragcli` subcommands.
#[derive(Subcommand, Debug)]
pub enum Command {
    /// Indexes a file or directory into the local store.
    Index {
        /// Path to a file or directory to index.
        path: PathBuf,
        /// Overrides the chunk size in characters.
        #[arg(long)]
        chunk_size: Option<usize>,
        /// Overrides the chunk overlap in characters.
        #[arg(long)]
        chunk_overlap: Option<usize>,
        /// Overrides the embedding model name.
        #[arg(long)]
        embed_model: Option<String>,
        /// Selects the PDF parser used for `.pdf` inputs.
        #[arg(long, value_enum)]
        pdf_parser: Option<PdfParserArg>,
        /// Excludes files or directories matching the provided glob.
        #[arg(long = "exclude")]
        exclude: Vec<String>,
        /// Includes hidden files and directories during traversal.
        #[arg(long, default_value_t = false)]
        include_hidden: bool,
    },
    /// Queries the local store and generates an answer.
    Query {
        /// Natural-language question to ask.
        question: String,
        /// Number of results to retrieve before generation.
        #[arg(long, default_value_t = 5)]
        top_k: usize,
        /// Prints retrieved context snippets before answering.
        #[arg(long, default_value_t = false)]
        show_context: bool,
        /// Overrides the generation model name.
        #[arg(long)]
        gen_model: Option<String>,
        /// Maximum number of tokens to generate.
        #[arg(long, default_value_t = 256)]
        max_tokens: usize,
    },
    /// Shows or updates configuration values.
    Config {
        /// Configuration subcommand to run.
        #[command(subcommand)]
        command: ConfigCommand,
    },
    /// Shows a summary of indexed content and store usage.
    Stat,
    /// Checks store layout and runtime dependencies.
    Doctor,
}

/// Configuration subcommands.
#[derive(Subcommand, Debug)]
pub enum ConfigCommand {
    /// Prints the effective configuration for the selected store.
    Show,
    /// Sets a config key in `~/.config/ragcli/<name>/config.toml`.
    Set {
        /// Configuration key such as `models.embed` or `ollama.base_url`.
        key: String,
        /// New value to write.
        value: String,
    },
}
