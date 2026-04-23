use crate::cli::{Cli, Command, ConfigCommand};
use crate::commands;
use anyhow::Result;
use tracing::{field, Instrument};

pub async fn run(cli: Cli) -> Result<()> {
    let name = cli.name.as_deref();
    let span_name = name.unwrap_or("default");
    let command_name = command_name(&cli.command);
    tracing::info!(name = span_name, command = command_name, "starting ragcli");
    let span = tracing::info_span!(
        "command_dispatch",
        store_name = span_name,
        command = command_name
    );

    async move {
        match cli.command {
            Command::Index {
                path,
                chunk_size,
                chunk_overlap,
                embed_model,
                pdf_parser,
                exclude,
                include_hidden,
                force,
            } => {
                commands::index::run(
                    name,
                    path,
                    chunk_size,
                    chunk_overlap,
                    embed_model,
                    pdf_parser,
                    exclude,
                    include_hidden,
                    force,
                )
                .await?
            }
            Command::Query {
                question,
                mode,
                top_k,
                fetch_k,
                max_iterations,
                rewrite,
                rerank,
                show_context,
                show_plan,
                show_scores,
                show_citations,
                show_trace,
                source,
                path_prefix,
                page,
                format,
                gen_model,
                max_tokens,
            } => {
                let span = tracing::info_span!(
                    "query_command",
                    mode = ?mode,
                    top_k,
                    fetch_k,
                    max_iterations,
                    rewrite,
                    rerank,
                    has_source_filter = source.is_some(),
                    has_path_prefix_filter = path_prefix.is_some(),
                    page = field::debug(page),
                    has_format_filter = format.is_some(),
                    max_tokens
                );
                commands::query::run(
                    name,
                    commands::query::QueryCommand {
                        question,
                        mode,
                        top_k,
                        fetch_k,
                        max_iterations,
                        rewrite,
                        rerank,
                        show_context,
                        show_plan,
                        show_scores,
                        show_citations,
                        show_trace,
                        source,
                        path_prefix,
                        page,
                        format,
                        gen_model,
                        max_tokens,
                    },
                )
                .instrument(span)
                .await?
            }
            Command::Config { command } => match command {
                ConfigCommand::Show { json } => commands::config::show(name, json).await?,
                ConfigCommand::Set { key, value } => {
                    commands::config::set(name, key, value).await?
                }
            },
            Command::Sources { json } => commands::sources::run(name, json).await?,
            Command::Delete { path } => commands::maintenance::delete(name, path).await?,
            Command::Clear { yes } => commands::maintenance::clear(name, yes).await?,
            Command::Prune { apply, json } => {
                commands::maintenance::prune(name, apply, json).await?
            }
            Command::Stat { json } => commands::stat::run(name, json).await?,
            Command::Doctor { json } => commands::doctor::run(name, json).await?,
        }

        Ok(())
    }
    .instrument(span)
    .await
}

fn command_name(command: &Command) -> &'static str {
    match command {
        Command::Index { .. } => "index",
        Command::Query { .. } => "query",
        Command::Config { .. } => "config",
        Command::Sources { .. } => "sources",
        Command::Delete { .. } => "delete",
        Command::Clear { .. } => "clear",
        Command::Prune { .. } => "prune",
        Command::Stat { .. } => "stat",
        Command::Doctor { .. } => "doctor",
    }
}
