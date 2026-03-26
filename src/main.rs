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
use lancedb::index::scalar::FullTextSearchQuery;
use lancedb::query::{ExecutableQuery, QueryBase};
use models::{Embedder, Generator, OllamaClient, VisionCaptioner};
use std::fs;
use std::path::Path;
use std::time::Instant;
use std::time::{SystemTime, UNIX_EPOCH};
use store::{
    collect_store_stats, connect_db, ensure_fts_index, ensure_metadata, extract_contexts,
    load_metadata, replace_source_rows,
};

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
        Command::Stat => cmd_stat(name).await?,
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
    let started = Instant::now();
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
    let vision = VisionCaptioner::new(
        cfg.ollama.base_url.clone(),
        resolve_model_name(&store, &cfg.models.vision),
    );
    let result = ingest_path(&path, size, overlap, &embedder, Some(&vision)).await?;

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
    println!("  elapsed: {:.2?}", started.elapsed());

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
    ensure_fts_index(&table, false).await?;

    let embedder = Embedder::new(cfg.ollama.base_url.clone(), embed_model_name);
    let embedding = embedder.embed(&question).await?;

    let batches: Vec<arrow_array::RecordBatch> = table
        .query()
        .full_text_search(FullTextSearchQuery::new(question.clone()))
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

async fn cmd_stat(name: Option<&str>) -> Result<()> {
    let store = store_dir(name)?;
    ensure_store_layout(&store)?;
    let cfg = load_or_create_config(&store)?;
    let metadata = load_metadata(&store).ok();
    let db = connect_db(&store).await?;

    let table = match db.open_table(store::DEFAULT_TABLE_NAME).execute().await {
        Ok(table) => Some(table),
        Err(_) => None,
    };

    let mut batches = Vec::new();
    if let Some(table) = &table {
        batches = table
            .query()
            .execute()
            .await?
            .try_collect::<Vec<_>>()
            .await?;
    }

    let stats = collect_store_stats(&batches, 5)?;
    let total_store_bytes = dir_size_bytes(&store)?;
    let lancedb_bytes = dir_size_bytes(&store.join("lancedb"))?;
    let meta_bytes = dir_size_bytes(&store.join("meta"))?;
    let cache_bytes = dir_size_bytes(&store.join("cache"))?;
    let models_bytes = dir_size_bytes(&store.join("models"))?;

    println!("Store summary");
    println!("  store: {}", store.display());
    println!("  ollama url: {}", cfg.ollama.base_url);
    println!("  rows embedded: {}", stats.total_chunks);
    println!("  source files: {}", stats.unique_sources);
    println!(
        "  content mix: {} text, {} pdf, {} image, {} other",
        stats.content_kinds.text_files,
        stats.content_kinds.pdf_files,
        stats.content_kinds.image_files,
        stats.content_kinds.other_files
    );
    println!("  pdf pages: {}", stats.pdf_pages);
    println!("  embedded chars: {}", fmt_count(stats.total_chars));
    println!(
        "  estimated embedded tokens: ~{}",
        fmt_count(stats.estimated_tokens)
    );

    if stats.total_chunks > 0 {
        println!(
            "  avg chunk: {} chars, ~{} tokens",
            stats.total_chars / stats.total_chunks,
            stats.estimated_tokens / stats.total_chunks
        );
        println!(
            "  chunk range: {}..{} chars",
            stats.min_chunk_chars, stats.max_chunk_chars
        );
    }

    if let Some(metadata) = metadata {
        println!(
            "  embedding: {} (dim {})",
            metadata.embed_model, metadata.embedding_dim
        );
        println!(
            "  chunking: size {}, overlap {}",
            metadata.chunk_size, metadata.chunk_overlap
        );
    } else {
        println!("  embedding: metadata missing");
    }

    println!("  disk usage: {}", fmt_bytes(total_store_bytes));
    println!("    lancedb: {}", fmt_bytes(lancedb_bytes));
    println!("    meta: {}", fmt_bytes(meta_bytes));
    println!("    cache: {}", fmt_bytes(cache_bytes));
    println!("    models: {}", fmt_bytes(models_bytes));

    if !stats.top_sources.is_empty() {
        println!("  top sources by chunk count:");
        for source in &stats.top_sources {
            println!(
                "    - {}  [{} chunks, ~{} tokens]",
                source.source_path,
                source.chunks,
                fmt_count(source.estimated_tokens)
            );
        }
    }

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
    println!("  vision model: {}", cfg.models.vision);

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
            println!(
                "  vision model installed: {}",
                status(models.iter().any(|model| model == &cfg.models.vision))
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

fn dir_size_bytes(path: &Path) -> Result<u64> {
    if !path.exists() {
        return Ok(0);
    }

    if path.is_file() {
        return Ok(path.metadata()?.len());
    }

    let mut total = 0_u64;
    for entry in walkdir::WalkDir::new(path) {
        let entry = entry?;
        if entry.file_type().is_file() {
            total += entry.metadata()?.len();
        }
    }
    Ok(total)
}

fn fmt_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }

    if unit == 0 {
        format!("{} {}", bytes, UNITS[unit])
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}

fn fmt_count(value: usize) -> String {
    let digits = value.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    let len = digits.len();
    for (idx, ch) in digits.chars().enumerate() {
        if idx > 0 && (len - idx).is_multiple_of(3) {
            out.push(',');
        }
        out.push(ch);
    }
    out
}
