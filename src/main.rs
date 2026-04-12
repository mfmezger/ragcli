mod cli;
mod config;
mod fsutil;
mod ingest;
mod models;
mod source_kind;
mod store;

use anyhow::{Context, Result};
use clap::Parser;
use cli::{Cli, Command, ConfigCommand, PdfParserArg};
use config::{
    ensure_store_layout, load_or_create_config, load_or_create_config_with_sources,
    load_or_create_file_config, normalize_chunk_settings, resolve_model_name, save_config, status,
    store_dir,
};
use futures::TryStreamExt;
use ingest::{ingest_path, PdfParser};
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
            pdf_parser,
            exclude,
            include_hidden,
        } => {
            cmd_index(
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
            gen_model,
            max_tokens,
        } => cmd_query(name, question, top_k, show_context, gen_model, max_tokens).await?,
        Command::Config { command } => match command {
            ConfigCommand::Show => cmd_config_show(name).await?,
            ConfigCommand::Set { key, value } => cmd_config_set(name, key, value).await?,
        },
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
    pdf_parser: Option<PdfParserArg>,
    exclude: Vec<String>,
    include_hidden: bool,
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
    let result = ingest_path(
        &path,
        size,
        overlap,
        &embedder,
        Some(&vision),
        match pdf_parser.unwrap_or(PdfParserArg::Native) {
            PdfParserArg::Native => PdfParser::Native,
            PdfParserArg::Liteparse => PdfParser::Liteparse,
        },
        &exclude,
        include_hidden,
    )
    .await?;

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

async fn cmd_config_show(name: Option<&str>) -> Result<()> {
    let store = store_dir(name)?;
    ensure_store_layout(&store)?;
    let (cfg, sources) = load_or_create_config_with_sources(&store)?;

    println!("Store: {}", store.display());
    println!("Config: {}", config::config_path(&store).display());
    println!();
    println!("{}", toml::to_string_pretty(&cfg)?);

    let overrides = sources.overrides();

    if !overrides.is_empty() {
        println!("Active environment overrides:");
        for override_entry in overrides {
            println!("  {}", override_entry);
        }
    }

    Ok(())
}

async fn cmd_config_set(name: Option<&str>, key: String, value: String) -> Result<()> {
    let store = store_dir(name)?;
    ensure_store_layout(&store)?;
    let mut cfg = load_or_create_file_config(&store)?;
    cfg.set_path(&key, &value)?;
    save_config(&store, &cfg)?;

    println!("Updated {}", config::config_path(&store).display());
    println!("  {} = {}", key, value);
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::{Mutex, OnceLock};
    use std::thread;

    fn env_lock() -> &'static Mutex<()> {
        static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        ENV_LOCK.get_or_init(|| Mutex::new(()))
    }

    async fn with_test_env<T, F>(
        config_home: &Path,
        ollama_url: Option<&str>,
        f: impl FnOnce() -> F,
    ) -> T
    where
        F: std::future::Future<Output = T>,
    {
        let _guard = env_lock().lock().unwrap();
        let previous_xdg = env::var_os("XDG_CONFIG_HOME");
        let previous_ollama = env::var_os(config::ENV_OLLAMA_URL);

        unsafe {
            env::set_var("XDG_CONFIG_HOME", config_home);
            match ollama_url {
                Some(url) => env::set_var(config::ENV_OLLAMA_URL, url),
                None => env::remove_var(config::ENV_OLLAMA_URL),
            }
        }

        let result = f().await;

        unsafe {
            match previous_xdg {
                Some(value) => env::set_var("XDG_CONFIG_HOME", value),
                None => env::remove_var("XDG_CONFIG_HOME"),
            }
            match previous_ollama {
                Some(value) => env::set_var(config::ENV_OLLAMA_URL, value),
                None => env::remove_var(config::ENV_OLLAMA_URL),
            }
        }

        result
    }

    fn sequential_json_server(bodies: Vec<&'static str>) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind test server");
        let addr = listener.local_addr().unwrap();

        thread::spawn(move || {
            for body in bodies {
                let (mut stream, _) = listener.accept().expect("accept request");
                let mut buf = [0_u8; 4096];
                let _ = stream.read(&mut buf);
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                stream
                    .write_all(response.as_bytes())
                    .expect("write response");
            }
        });

        format!("http://{}", addr)
    }

    #[test]
    fn test_dir_size_bytes_for_missing_file_file_and_directory() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("a.txt");
        let nested = dir.path().join("nested");
        let nested_file = nested.join("b.txt");
        std::fs::create_dir_all(&nested).unwrap();
        std::fs::write(&file, b"abc").unwrap();
        std::fs::write(&nested_file, b"12345").unwrap();

        assert_eq!(dir_size_bytes(&dir.path().join("missing")).unwrap(), 0);
        assert_eq!(dir_size_bytes(&file).unwrap(), 3);
        assert_eq!(dir_size_bytes(dir.path()).unwrap(), 8);
    }

    #[test]
    fn test_fmt_bytes_and_fmt_count() {
        assert_eq!(fmt_bytes(999), "999 B");
        assert_eq!(fmt_bytes(2048), "2.0 KB");
        assert_eq!(fmt_count(12), "12");
        assert_eq!(fmt_count(1234), "1,234");
        assert_eq!(fmt_count(1234567), "1,234,567");
    }

    #[tokio::test]
    async fn test_cmd_config_set_and_show_succeed_for_named_store() {
        let dir = tempfile::tempdir().unwrap();
        with_test_env(dir.path(), None, || async {
            cmd_config_set(
                Some("test-store"),
                "models.chat".to_string(),
                "chat-z".to_string(),
            )
            .await
            .unwrap();
            cmd_config_show(Some("test-store")).await.unwrap();

            let store = store_dir(Some("test-store")).unwrap();
            let cfg = load_or_create_file_config(&store).unwrap();
            assert_eq!(cfg.models.chat, "chat-z");
        })
        .await;
    }

    #[tokio::test]
    async fn test_cmd_stat_and_doctor_succeed_on_empty_store() {
        let dir = tempfile::tempdir().unwrap();
        with_test_env(dir.path(), None, || async {
            cmd_stat(Some("empty")).await.unwrap();
            cmd_doctor(Some("empty")).await.unwrap();
        })
        .await;
    }

    #[tokio::test]
    async fn test_cmd_doctor_succeeds_with_reachable_mock_ollama() {
        let dir = tempfile::tempdir().unwrap();
        let server = sequential_json_server(vec![
            r#"{"models":[{"name":"nomic-embed-text-v2-moe:latest"},{"name":"qwen3.5:4b"}]}"#,
        ]);

        with_test_env(dir.path(), Some(&server), || async {
            cmd_doctor(Some("reachable")).await.unwrap();
        })
        .await;
    }

    #[tokio::test]
    async fn test_cmd_index_and_query_succeed_with_mock_ollama() {
        let dir = tempfile::tempdir().unwrap();
        let docs = dir.path().join("docs");
        std::fs::create_dir_all(&docs).unwrap();
        let input = docs.join("note.txt");
        std::fs::write(&input, "The project is a local RAG CLI.").unwrap();

        let index_server = sequential_json_server(vec![r#"{"embeddings":[[0.1,0.2]]}"#]);
        with_test_env(dir.path(), Some(&index_server), || async {
            cmd_index(
                Some("e2e"),
                input.clone(),
                Some(200),
                Some(0),
                None,
                None,
                Vec::new(),
                false,
            )
            .await
            .unwrap();
            cmd_stat(Some("e2e")).await.unwrap();
        })
        .await;

        let query_server = sequential_json_server(vec![
            r#"{"embeddings":[[0.1,0.2]]}"#,
            r#"{"message":{"content":"ragcli is a local RAG CLI."}}"#,
        ]);
        with_test_env(dir.path(), Some(&query_server), || async {
            cmd_query(
                Some("e2e"),
                "What is this project?".to_string(),
                5,
                true,
                None,
                64,
            )
            .await
            .unwrap();
        })
        .await;
    }
}
