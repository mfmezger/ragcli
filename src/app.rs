use crate::cli::{Cli, Command, ConfigCommand};
use crate::commands;
use anyhow::Result;

pub async fn run(cli: Cli) -> Result<()> {
    let name = cli.name.as_deref();

    match cli.command {
        Command::Index {
            path,
            chunk_size,
            chunk_overlap,
            embed_model,
            pdf_parser,
            exclude,
            include_hidden,
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
            )
            .await?
        }
        Command::Query {
            question,
            top_k,
            show_context,
            source,
            path_prefix,
            page,
            format,
            gen_model,
            max_tokens,
        } => {
            commands::query::run(
                name,
                question,
                top_k,
                show_context,
                source,
                path_prefix,
                page,
                format,
                gen_model,
                max_tokens,
            )
            .await?
        }
        Command::Config { command } => match command {
            ConfigCommand::Show => commands::config::show(name).await?,
            ConfigCommand::Set { key, value } => commands::config::set(name, key, value).await?,
        },
        Command::Stat => commands::stat::run(name).await?,
        Command::Doctor => commands::doctor::run(name).await?,
    }

    Ok(())
}
