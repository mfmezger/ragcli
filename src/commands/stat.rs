use crate::config::{ensure_store_layout, load_or_create_config, store_dir, STORE_SUBDIRECTORIES};
use crate::store::{
    collect_store_stats, connect_db, load_metadata, StoreMetadata, StoreStats, DEFAULT_TABLE_NAME,
};
use anyhow::Result;
use futures::TryStreamExt;
use lancedb::query::ExecutableQuery;
use serde::Serialize;
use std::path::Path;
use walkdir::WalkDir;

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
    pub warnings: Vec<String>,
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
    let (disk_usage, warnings) = collect_disk_usage(&store);

    Ok(StatReport {
        store: store.display().to_string(),
        ollama_url: cfg.ollama.base_url,
        stats,
        metadata,
        disk_usage,
        warnings,
    })
}

fn print_human(report: &StatReport) {
    tracing::info!("Store summary");
    tracing::info!("  store: {}", report.store);
    tracing::info!("  ollama url: {}", report.ollama_url);
    tracing::info!("  rows embedded: {}", report.stats.total_chunks);
    tracing::info!("  source files: {}", report.stats.unique_sources);
    tracing::info!(
        "  content mix: {} text, {} pdf, {} image, {} other",
        report.stats.content_kinds.text_files,
        report.stats.content_kinds.pdf_files,
        report.stats.content_kinds.image_files,
        report.stats.content_kinds.other_files
    );
    tracing::info!("  pdf pages: {}", report.stats.pdf_pages);
    tracing::info!("  embedded chars: {}", fmt_count(report.stats.total_chars));
    tracing::info!(
        "  estimated embedded tokens: ~{}",
        fmt_count(report.stats.estimated_tokens)
    );

    if report.stats.total_chunks > 0 {
        tracing::info!(
            "  avg chunk: {} chars, ~{} tokens",
            report.stats.total_chars / report.stats.total_chunks,
            report.stats.estimated_tokens / report.stats.total_chunks
        );
        tracing::info!(
            "  chunk range: {}..{} chars",
            report.stats.min_chunk_chars, report.stats.max_chunk_chars
        );
    }

    if let Some(metadata) = &report.metadata {
        tracing::info!(
            "  embedding: {} (dim {})",
            metadata.embed_model, metadata.embedding_dim
        );
        tracing::info!(
            "  chunking: size {}, overlap {}",
            metadata.chunk_size, metadata.chunk_overlap
        );
    } else {
        tracing::info!("  embedding: metadata missing");
    }

    tracing::info!("  disk usage: {}", fmt_bytes(report.disk_usage.total_bytes));
    tracing::info!(
        "    lancedb: {}",
        fmt_bytes(report.disk_usage.lancedb_bytes)
    );
    tracing::info!("    meta: {}", fmt_bytes(report.disk_usage.meta_bytes));
    tracing::info!("    cache: {}", fmt_bytes(report.disk_usage.cache_bytes));
    tracing::info!("    models: {}", fmt_bytes(report.disk_usage.models_bytes));
    for warning in &report.warnings {
        tracing::warn!("  warning: {}", warning);
    }

    if !report.stats.top_sources.is_empty() {
        tracing::info!("  top sources by chunk count:");
        for source in &report.stats.top_sources {
            tracing::info!(
                "    - {}  [{} chunks, ~{} tokens]",
                source.source_path,
                fmt_count(source.chunks),
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

fn collect_disk_usage(store: &Path) -> (DiskUsageReport, Vec<String>) {
    let mut report = DiskUsageReport {
        total_bytes: 0,
        lancedb_bytes: 0,
        meta_bytes: 0,
        cache_bytes: 0,
        models_bytes: 0,
    };
    let mut warnings = Vec::new();

    if !store.exists() {
        return (report, warnings);
    }

    for entry in WalkDir::new(store) {
        let entry = match entry {
            Ok(entry) => entry,
            Err(err) => {
                warnings.push(format!("skipped unreadable path: {err}"));
                continue;
            }
        };
        if !entry.file_type().is_file() {
            continue;
        }

        let bytes = match entry.metadata() {
            Ok(metadata) => metadata.len(),
            Err(err) => {
                warnings.push(format!(
                    "skipped unreadable file {}: {}",
                    entry.path().display(),
                    err
                ));
                continue;
            }
        };
        report.total_bytes += bytes;

        let Ok(relative_path) = entry.path().strip_prefix(store) else {
            continue;
        };
        let Some(first_component) = relative_path.components().next() else {
            continue;
        };
        let Some(name) = first_component.as_os_str().to_str() else {
            continue;
        };

        match name {
            "lancedb" => report.lancedb_bytes += bytes,
            "meta" => report.meta_bytes += bytes,
            "cache" => report.cache_bytes += bytes,
            "models" => report.models_bytes += bytes,
            _ => {}
        }
    }

    (report, warnings)
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

    #[test]
    fn test_collect_disk_usage_walks_store_once_and_keeps_root_files() {
        let dir = tempfile::tempdir().unwrap();
        let store = dir.path().join("store");
        std::fs::create_dir_all(&store).unwrap();
        for subdirectory in STORE_SUBDIRECTORIES {
            std::fs::create_dir_all(store.join(subdirectory)).unwrap();
        }
        std::fs::write(store.join("config.toml"), b"abc").unwrap();
        std::fs::write(store.join("lancedb").join("data.bin"), b"12345").unwrap();
        std::fs::write(store.join("meta").join("store.toml"), b"12").unwrap();
        std::fs::write(store.join("cache").join("blob.bin"), b"1234").unwrap();
        std::fs::write(store.join("models").join("weights.gguf"), b"123456").unwrap();
        std::fs::create_dir_all(store.join("other")).unwrap();
        std::fs::write(store.join("other").join("extra.bin"), b"1234567").unwrap();

        let (report, warnings) = collect_disk_usage(&store);

        assert!(warnings.is_empty());
        assert_eq!(report.total_bytes, 27);
        assert_eq!(report.lancedb_bytes, 5);
        assert_eq!(report.meta_bytes, 2);
        assert_eq!(report.cache_bytes, 4);
        assert_eq!(report.models_bytes, 6);
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
