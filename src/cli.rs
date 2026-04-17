//! Command-line interface definitions for `ragcli`.

use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

/// PDF extraction backend used during indexing.
#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
pub enum PdfParserArg {
    Native,
    Liteparse,
}

/// Retrieval mode used during querying.
#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
pub enum QueryModeArg {
    Naive,
    Hybrid,
    Agentic,
    Local,
    Global,
    Mix,
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
        /// Re-embeds files even when their stored fingerprint is unchanged.
        #[arg(long, default_value_t = false)]
        force: bool,
    },
    /// Queries the local store and generates an answer.
    Query {
        /// Natural-language question to ask.
        question: String,
        /// Retrieval mode used for the query path.
        #[arg(long, default_value = "hybrid")]
        mode: QueryModeArg,
        /// Number of results to retrieve before generation.
        #[arg(long, default_value_t = 5)]
        top_k: usize,
        /// Number of candidates to overfetch before later pruning or reranking.
        #[arg(long, default_value_t = 20)]
        fetch_k: usize,
        /// Maximum number of Ralph-style retrieval iterations.
        #[arg(long, default_value_t = 2)]
        max_iterations: usize,
        /// Enables retrieval-oriented query rewriting.
        #[arg(long, default_value_t = false)]
        rewrite: bool,
        /// Enables reranking after retrieval.
        #[arg(long, default_value_t = false)]
        rerank: bool,
        /// Prints retrieved context snippets before answering.
        #[arg(long, default_value_t = false)]
        show_context: bool,
        /// Prints the selected retrieval plan.
        #[arg(long, default_value_t = false)]
        show_plan: bool,
        /// Prints retrieval scores for selected contexts.
        #[arg(long, default_value_t = false)]
        show_scores: bool,
        /// Prints selected evidence sources and labels.
        #[arg(long, default_value_t = false)]
        show_citations: bool,
        /// Prints per-iteration retrieval trace details.
        #[arg(long, default_value_t = false)]
        show_trace: bool,
        /// Restricts retrieval to a single indexed source path.
        #[arg(long)]
        source: Option<String>,
        /// Restricts retrieval to indexed source paths under the given prefix.
        #[arg(long)]
        path_prefix: Option<String>,
        /// Restricts retrieval to a specific page number.
        #[arg(long)]
        page: Option<i32>,
        /// Restricts retrieval to a specific indexed content format.
        #[arg(long)]
        format: Option<String>,
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
    /// Lists indexed source paths together with per-source metadata.
    #[command(visible_alias = "ls")]
    Sources {
        /// Prints machine-readable JSON instead of the default text report.
        #[arg(long, default_value_t = false)]
        json: bool,
    },
    /// Removes a single indexed source path from the store.
    Delete {
        /// Indexed source path to remove.
        path: String,
    },
    /// Removes all indexed content from the selected store.
    Clear {
        /// Confirms the destructive clear operation.
        #[arg(long, default_value_t = false)]
        yes: bool,
    },
    /// Removes indexed sources whose files no longer exist on disk.
    Prune {
        /// Applies the prune instead of printing a preview.
        #[arg(long, default_value_t = false)]
        apply: bool,
        /// Prints machine-readable JSON instead of the default text report.
        #[arg(long, default_value_t = false)]
        json: bool,
    },
    /// Shows a summary of indexed content and store usage.
    Stat {
        /// Prints machine-readable JSON instead of the default text report.
        #[arg(long, default_value_t = false)]
        json: bool,
    },
    /// Checks store layout and runtime dependencies.
    Doctor {
        /// Prints machine-readable JSON instead of the default text report.
        #[arg(long, default_value_t = false)]
        json: bool,
    },
}

/// Configuration subcommands.
#[derive(Subcommand, Debug)]
pub enum ConfigCommand {
    /// Prints the effective configuration for the selected store.
    Show {
        /// Prints machine-readable JSON instead of the default text report.
        #[arg(long, default_value_t = false)]
        json: bool,
    },
    /// Sets a config key in `~/.config/ragcli/<name>/config.toml`.
    Set {
        /// Configuration key such as `models.embed` or `ollama.base_url`.
        key: String,
        /// New value to write.
        value: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_query_mode_and_inspection_flags_parse() {
        let cli = Cli::try_parse_from([
            "ragcli",
            "query",
            "What is this?",
            "--mode",
            "agentic",
            "--rewrite",
            "--rerank",
            "--show-plan",
            "--show-scores",
            "--show-citations",
            "--show-trace",
            "--fetch-k",
            "12",
            "--max-iterations",
            "4",
        ])
        .unwrap();

        match cli.command {
            Command::Query {
                question,
                mode,
                rewrite,
                rerank,
                show_plan,
                show_scores,
                show_citations,
                show_trace,
                fetch_k,
                max_iterations,
                ..
            } => {
                assert_eq!(question, "What is this?");
                assert_eq!(mode, QueryModeArg::Agentic);
                assert!(rewrite);
                assert!(rerank);
                assert!(show_plan);
                assert!(show_scores);
                assert!(show_citations);
                assert!(show_trace);
                assert_eq!(fetch_k, 12);
                assert_eq!(max_iterations, 4);
            }
            command => panic!("expected query command, got {command:?}"),
        }
    }

    #[test]
    fn test_query_mode_defaults_to_hybrid() {
        let cli = Cli::try_parse_from(["ragcli", "query", "What is this?"]).unwrap();

        match cli.command {
            Command::Query {
                mode,
                fetch_k,
                max_iterations,
                ..
            } => {
                assert_eq!(mode, QueryModeArg::Hybrid);
                assert_eq!(fetch_k, 20);
                assert_eq!(max_iterations, 2);
            }
            command => panic!("expected query command, got {command:?}"),
        }
    }

    #[test]
    fn test_index_force_flag_parses() {
        let cli = Cli::try_parse_from(["ragcli", "index", "./docs", "--force"]).unwrap();
        match cli.command {
            Command::Index { path, force, .. } => {
                assert_eq!(path, PathBuf::from("./docs"));
                assert!(force);
            }
            command => panic!("expected index command, got {command:?}"),
        }
    }

    #[test]
    fn test_store_management_commands_parse() {
        let sources = Cli::try_parse_from(["ragcli", "ls", "--json"]).unwrap();
        match sources.command {
            Command::Sources { json } => assert!(json),
            command => panic!("expected sources command, got {command:?}"),
        }

        let delete = Cli::try_parse_from(["ragcli", "delete", "docs/a.md"]).unwrap();
        match delete.command {
            Command::Delete { path } => assert_eq!(path, "docs/a.md"),
            command => panic!("expected delete command, got {command:?}"),
        }

        let clear = Cli::try_parse_from(["ragcli", "clear", "--yes"]).unwrap();
        match clear.command {
            Command::Clear { yes } => assert!(yes),
            command => panic!("expected clear command, got {command:?}"),
        }

        let prune = Cli::try_parse_from(["ragcli", "prune", "--apply", "--json"]).unwrap();
        match prune.command {
            Command::Prune { apply, json } => {
                assert!(apply);
                assert!(json);
            }
            command => panic!("expected prune command, got {command:?}"),
        }
    }

    #[test]
    fn test_json_flags_parse_for_supported_commands() {
        let stat = Cli::try_parse_from(["ragcli", "stat", "--json"]).unwrap();
        match stat.command {
            Command::Stat { json } => assert!(json),
            command => panic!("expected stat command, got {command:?}"),
        }

        let doctor = Cli::try_parse_from(["ragcli", "doctor", "--json"]).unwrap();
        match doctor.command {
            Command::Doctor { json } => assert!(json),
            command => panic!("expected doctor command, got {command:?}"),
        }

        let config = Cli::try_parse_from(["ragcli", "config", "show", "--json"]).unwrap();
        match config.command {
            Command::Config {
                command: ConfigCommand::Show { json },
            } => assert!(json),
            command => panic!("expected config show command, got {command:?}"),
        }
    }
}
