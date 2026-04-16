use crate::cli::{Cli, Command, ConfigCommand};
use crate::commands;
use anyhow::Result;
use tracing::Instrument;

pub async fn run(cli: Cli) -> Result<()> {
    let name = cli.name.as_deref();
    let span_name = name.unwrap_or("default");
    tracing::info!(name = span_name, "starting ragcli");
    let span = tracing::info_span!("command_dispatch", name = span_name);

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
                .await?
            }
            Command::Config { command } => match command {
                ConfigCommand::Show { json } => commands::config::show(name, json).await?,
                ConfigCommand::Set { key, value } => {
                    commands::config::set(name, key, value).await?
                }
            },
            Command::Stat { json } => commands::stat::run(name, json).await?,
            Command::Doctor { json } => commands::doctor::run(name, json).await?,
        }

        Ok(())
    }
    .instrument(span)
    .await
}
