mod cli;
mod config;
mod ingest;
mod models;
mod store;

use anyhow::{Context, Result};
use clap::Parser;
use cli::{Cli, Command};
use config::{
    ensure_store_layout, load_or_create_config, normalize_chunk_settings, resolve_model_name,
    status, store_dir,
};
use futures::TryStreamExt;
use ingest::ingest_path;
use lancedb::query::{ExecutableQuery, QueryBase};
use models::{Embedder, Generator, OllamaClient};
use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};
use store::{connect_db, ensure_metadata, extract_contexts, load_metadata, replace_source_rows};

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let name = cli.name.as_deref();

    match cli.command {
        Command::Index {
            path,
            chunk_size,
            chunk_overlap,
            embed_model,
        } => cmd_index(name, path, chunk_size, chunk_overlap, embed_model).await?,
        Command::Query {
            question,
            top_k,
            show_context,
            gen_model,
            max_tokens,
        } => cmd_query(name, question, top_k, show_context, gen_model, max_tokens).await?,
        Command::Doctor => cmd_doctor(name).await?,
    }

    Ok(())
}

async fn cmd_index(
    name: Option<&str>,
    path: std::path::PathBuf,
    chunk_size: Option<usize>,
    chunk_overlap: Option<usize>,
    embed_model: Option<String>,
) -> Result<()> {
    let store = store_dir(name)?;
    ensure_store_layout(&store)?;
    let cfg = load_or_create_config(&store)?;

    let (size, overlap) = normalize_chunk_settings(
        chunk_size.unwrap_or(cfg.chunk.size),
        chunk_overlap.unwrap_or(cfg.chunk.overlap),
    );
    let embed_model_name =
        embed_model.unwrap_or_else(|| resolve_model_name(&store, &cfg.models.embed));

    let embedder = Embedder::new(cfg.ollama.base_url.clone(), embed_model_name.clone());
    let result = ingest_path(&path, size, overlap, &embedder).await?;

    if let Some(dim) = result.embedding_dim {
        ensure_metadata(&store, &embed_model_name, dim, size, overlap)?;
    }

    let db = connect_db(&store).await?;
    replace_source_rows(&db, &result.rows, &result.source_paths).await?;

    println!("Index complete");
    println!("  store: {}", store.display());
    println!("  source: {}", path.display());
    println!("  indexed files: {}", result.stats.indexed_files);
    println!("  skipped files: {}", result.stats.skipped_files);
    println!("  chunks written: {}", result.stats.total_chunks);

    if !result.stats.errors.is_empty() {
        println!("  errors:");
        for err in &result.stats.errors {
            println!("    - {}", err);
        }
    }

    Ok(())
}

async fn cmd_query(
    name: Option<&str>,
    question: String,
    top_k: usize,
    show_context: bool,
    gen_model: Option<String>,
    max_tokens: usize,
) -> Result<()> {
    let store = store_dir(name)?;
    ensure_store_layout(&store)?;
    let cfg = load_or_create_config(&store)?;

    let gen_model_name = gen_model.unwrap_or_else(|| resolve_model_name(&store, &cfg.models.chat));
    let embed_model_name = resolve_model_name(&store, &cfg.models.embed);
    let metadata = load_metadata(&store)?;
    metadata.validate_query_model(&embed_model_name)?;

    let db = connect_db(&store).await?;
    let table = db
        .open_table(store::DEFAULT_TABLE_NAME)
        .execute()
        .await
        .context("open table")?;

    let embedder = Embedder::new(cfg.ollama.base_url.clone(), embed_model_name);
    let embedding = embedder.embed(&question).await?;

    let batches: Vec<arrow_array::RecordBatch> = table
        .query()
        .nearest_to(embedding.as_slice())?
        .limit(top_k)
        .execute()
        .await?
        .try_collect::<Vec<_>>()
        .await?;

    let contexts = extract_contexts(&batches)?;
    if contexts.is_empty() {
        println!("No relevant context found in the local store.");
        return Ok(());
    }

    if show_context {
        println!("Retrieved context:");
        for (idx, ctx) in contexts.iter().enumerate() {
            let compact = ctx.replace('\n', " ");
            let preview: String = compact.chars().take(220).collect();
            println!("  [{}] {}", idx + 1, preview);
        }
        println!();
    }

    let generator = Generator::new(cfg.ollama.base_url.clone(), gen_model_name);
    let answer = generator
        .generate_answer(&contexts, &question, max_tokens)
        .await?;

    println!();
    println!("{}", store::strip_thinking(&answer).trim());
    Ok(())
}

async fn cmd_doctor(name: Option<&str>) -> Result<()> {
    let store = store_dir(name)?;
    ensure_store_layout(&store)?;
    let cfg = load_or_create_config(&store)?;
    let base = config::base_dir()?;
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system time before unix epoch")?
        .as_secs();

    println!("Doctor report");
    println!("  time: {}", now);
    println!("  base: {} ({})", base.display(), status(base.exists()));
    println!("  store: {} ({})", store.display(), status(store.exists()));
    println!(
        "  config: {} ({})",
        config::config_path(&store).display(),
        status(config::config_path(&store).exists())
    );
    println!("  ollama url: {}", cfg.ollama.base_url);
    println!("  embed model: {}", cfg.models.embed);
    println!("  chat model: {}", cfg.models.chat);

    let ollama = OllamaClient::new(cfg.ollama.base_url.clone());
    match ollama.list_models().await {
        Ok(models) => {
            println!("  ollama: reachable");
            println!(
                "  embed model installed: {}",
                status(models.iter().any(|model| model == &cfg.models.embed))
            );
            println!(
                "  chat model installed: {}",
                status(models.iter().any(|model| model == &cfg.models.chat))
            );
        }
        Err(err) => {
            println!("  ollama: unreachable ({})", err);
        }
    }

    let metadata_path = store::metadata_path(&store);
    println!(
        "  metadata: {} ({})",
        metadata_path.display(),
        status(metadata_path.exists())
    );
    if metadata_path.exists() {
        let metadata = fs::read_to_string(&metadata_path)?;
        println!("  metadata summary: {}", metadata.replace('\n', " "));
    }

    for sub in ["lancedb", "meta", "cache", "models"] {
        let path = store.join(sub);
        println!("  {}: {} ({})", sub, path.display(), status(path.exists()));
    }

    Ok(())
}
