use crate::config::{ensure_store_layout, load_or_create_config, store_dir};
use crate::store::{collect_store_stats, connect_db, load_metadata};
use anyhow::Result;
use futures::TryStreamExt;
use lancedb::query::ExecutableQuery;
use std::path::Path;

pub async fn run(name: Option<&str>) -> Result<()> {
    let store = store_dir(name)?;
    ensure_store_layout(&store)?;
    let cfg = load_or_create_config(&store)?;
    let metadata = load_metadata(&store).ok();
    let db = connect_db(&store).await?;

    let table = match db
        .open_table(crate::store::DEFAULT_TABLE_NAME)
        .execute()
        .await
    {
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

pub fn dir_size_bytes(path: &Path) -> Result<u64> {
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

pub fn fmt_bytes(bytes: u64) -> String {
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

pub fn fmt_count(value: usize) -> String {
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
    use crate::test_support::with_test_env;

    #[test]
    fn test_helpers_format_counts_and_sizes() {
        assert_eq!(fmt_bytes(999), "999 B");
        assert_eq!(fmt_bytes(2048), "2.0 KB");
        assert_eq!(fmt_count(12), "12");
        assert_eq!(fmt_count(1234), "1,234");
        assert_eq!(fmt_count(1234567), "1,234,567");
    }

    #[test]
    fn test_helpers_measure_directory_sizes() {
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

    #[tokio::test]
    async fn test_run_succeeds_on_empty_store() {
        let dir = tempfile::tempdir().unwrap();
        with_test_env(dir.path(), None, || async {
            run(Some("empty")).await.unwrap();
        })
        .await;
    }
}
