use crate::config::{ensure_store_layout, load_or_create_config, store_dir};
use crate::store::{
    collect_store_stats, connect_db, load_metadata, StoreMetadata, StoreStats, DEFAULT_TABLE_NAME,
};
use anyhow::Result;
use futures::TryStreamExt;
use lancedb::query::ExecutableQuery;
use serde::Serialize;
use std::path::Path;

#[derive(Debug, Serialize)]
pub struct DiskUsageReport {
    pub total_bytes: u64,
    pub lancedb_bytes: u64,
    pub meta_bytes: u64,
    pub cache_bytes: u64,
    pub models_bytes: u64,
}

#[derive(Debug, Serialize)]
pub struct StatReport {
    pub store: String,
    pub ollama_url: String,
    pub stats: StoreStats,
    pub metadata: Option<StoreMetadata>,
    pub disk_usage: DiskUsageReport,
}

pub async fn run(name: Option<&str>, json: bool) -> Result<()> {
    let report = build_report(name).await?;
    if json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        print_human(&report);
    }
    Ok(())
}

async fn build_report(name: Option<&str>) -> Result<StatReport> {
    let store = store_dir(name)?;
    ensure_store_layout(&store)?;
    let cfg = load_or_create_config(&store)?;
    let metadata = load_metadata(&store).ok();
    let db = connect_db(&store).await?;

    let table = match db.open_table(DEFAULT_TABLE_NAME).execute().await {
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
    let disk_usage = DiskUsageReport {
        total_bytes: dir_size_bytes(&store)?,
        lancedb_bytes: dir_size_bytes(&store.join("lancedb"))?,
        meta_bytes: dir_size_bytes(&store.join("meta"))?,
        cache_bytes: dir_size_bytes(&store.join("cache"))?,
        models_bytes: dir_size_bytes(&store.join("models"))?,
    };

    Ok(StatReport {
        store: store.display().to_string(),
        ollama_url: cfg.ollama.base_url,
        stats,
        metadata,
        disk_usage,
    })
}

fn print_human(report: &StatReport) {
    println!("Store summary");
    println!("  store: {}", report.store);
    println!("  ollama url: {}", report.ollama_url);
    println!("  rows embedded: {}", report.stats.total_chunks);
    println!("  source files: {}", report.stats.unique_sources);
    println!(
        "  content mix: {} text, {} pdf, {} image, {} other",
        report.stats.content_kinds.text_files,
        report.stats.content_kinds.pdf_files,
        report.stats.content_kinds.image_files,
        report.stats.content_kinds.other_files
    );
    println!("  pdf pages: {}", report.stats.pdf_pages);
    println!("  embedded chars: {}", fmt_count(report.stats.total_chars));
    println!(
        "  estimated embedded tokens: ~{}",
        fmt_count(report.stats.estimated_tokens)
    );

    if report.stats.total_chunks > 0 {
        println!(
            "  avg chunk: {} chars, ~{} tokens",
            report.stats.total_chars / report.stats.total_chunks,
            report.stats.estimated_tokens / report.stats.total_chunks
        );
        println!(
            "  chunk range: {}..{} chars",
            report.stats.min_chunk_chars, report.stats.max_chunk_chars
        );
    }

    if let Some(metadata) = &report.metadata {
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

    println!("  disk usage: {}", fmt_bytes(report.disk_usage.total_bytes));
    println!(
        "    lancedb: {}",
        fmt_bytes(report.disk_usage.lancedb_bytes)
    );
    println!("    meta: {}", fmt_bytes(report.disk_usage.meta_bytes));
    println!("    cache: {}", fmt_bytes(report.disk_usage.cache_bytes));
    println!("    models: {}", fmt_bytes(report.disk_usage.models_bytes));

    if !report.stats.top_sources.is_empty() {
        println!("  top sources by chunk count:");
        for source in &report.stats.top_sources {
            println!(
                "    - {}  [{} chunks, ~{} tokens]",
                source.source_path,
                source.chunks,
                fmt_count(source.estimated_tokens)
            );
        }
    }
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

    #[tokio::test(flavor = "current_thread")]
    async fn test_build_report_supports_empty_store() {
        let dir = tempfile::tempdir().unwrap();
        with_test_env(dir.path(), None, || async {
            let report = build_report(Some("empty")).await.unwrap();
            assert_eq!(report.stats.total_chunks, 0);
            assert!(serde_json::to_string(&report)
                .unwrap()
                .contains("\"stats\""));
        })
        .await;
    }
}
