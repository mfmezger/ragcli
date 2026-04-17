use crate::commands::stat::fmt_count;
use crate::config::{ensure_store_layout, store_dir};
use crate::store::{connect_db, delete_source_rows, list_indexed_sources, IndexedSource};
use anyhow::{bail, Result};
use serde::Serialize;
use std::path::Path;

#[derive(Debug, Serialize)]
pub struct PruneReport {
    pub store: String,
    pub dry_run: bool,
    pub stale_sources: Vec<IndexedSource>,
    pub deleted_sources: usize,
    pub deleted_chunks: usize,
}

pub async fn delete(name: Option<&str>, path: String) -> Result<()> {
    let store = store_dir(name)?;
    ensure_store_layout(&store)?;
    let db = connect_db(&store).await?;
    let sources = list_indexed_sources(&db).await?;
    let Some(source) = sources
        .into_iter()
        .find(|source| source.source_path == path)
    else {
        println!("No indexed source matched {}", path);
        return Ok(());
    };

    delete_source_rows(&db, std::slice::from_ref(&source.source_path)).await?;
    println!(
        "Deleted {} [{} chunks, ~{} tokens]",
        source.source_path,
        fmt_count(source.chunks),
        fmt_count(source.estimated_tokens)
    );
    Ok(())
}

pub async fn clear(name: Option<&str>, yes: bool) -> Result<()> {
    if !yes {
        bail!("clear is destructive; rerun with --yes to remove all indexed sources from the selected store");
    }

    let store = store_dir(name)?;
    ensure_store_layout(&store)?;
    let db = connect_db(&store).await?;
    let sources = list_indexed_sources(&db).await?;
    if sources.is_empty() {
        println!("Store already empty: {}", store.display());
        return Ok(());
    }

    let deleted_sources = sources.len();
    let deleted_chunks = sources.iter().map(|source| source.chunks).sum::<usize>();
    let paths = sources
        .into_iter()
        .map(|source| source.source_path)
        .collect::<Vec<_>>();
    delete_source_rows(&db, &paths).await?;

    println!(
        "Cleared store {}: removed {} sources and {} chunks",
        store.display(),
        fmt_count(deleted_sources),
        fmt_count(deleted_chunks)
    );
    Ok(())
}

pub async fn prune(name: Option<&str>, apply: bool, json: bool) -> Result<()> {
    let report = build_prune_report(name, apply).await?;
    if json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        print_prune_human(&report);
    }
    Ok(())
}

async fn build_prune_report(name: Option<&str>, apply: bool) -> Result<PruneReport> {
    let store = store_dir(name)?;
    ensure_store_layout(&store)?;
    let db = connect_db(&store).await?;
    let stale_sources = list_indexed_sources(&db)
        .await?
        .into_iter()
        .filter(|source| !Path::new(&source.source_path).exists())
        .collect::<Vec<_>>();
    let deleted_sources = stale_sources.len();
    let deleted_chunks = stale_sources
        .iter()
        .map(|source| source.chunks)
        .sum::<usize>();

    if apply && !stale_sources.is_empty() {
        let paths = stale_sources
            .iter()
            .map(|source| source.source_path.clone())
            .collect::<Vec<_>>();
        delete_source_rows(&db, &paths).await?;
    }

    Ok(PruneReport {
        store: store.display().to_string(),
        dry_run: !apply,
        stale_sources,
        deleted_sources: if apply { deleted_sources } else { 0 },
        deleted_chunks: if apply { deleted_chunks } else { 0 },
    })
}

fn print_prune_human(report: &PruneReport) {
    if report.dry_run {
        println!("Prune preview");
    } else {
        println!("Prune complete");
    }
    println!("  store: {}", report.store);
    println!("  stale sources: {}", report.stale_sources.len());

    if report.stale_sources.is_empty() {
        println!("  nothing to prune");
        return;
    }

    for source in &report.stale_sources {
        println!(
            "    - {}  [{}, {} chunks, ~{} tokens]",
            source.source_path,
            source.format,
            fmt_count(source.chunks),
            fmt_count(source.estimated_tokens)
        );
    }

    if report.dry_run {
        println!("  rerun with --apply to remove these rows");
    } else {
        println!(
            "  removed {} sources and {} chunks",
            fmt_count(report.deleted_sources),
            fmt_count(report.deleted_chunks)
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source_kind::SourceKind;
    use crate::store::{connect_db, list_indexed_sources, replace_source_rows, ChunkRow};
    use crate::test_support::with_test_env;
    use std::path::Path;

    fn sample_row(source_path: &str, text: &str) -> ChunkRow {
        ChunkRow {
            id: format!("{}-{}", source_path, text),
            source_path: source_path.to_string(),
            chunk_text: text.to_string(),
            chunk_hash: text.to_string(),
            format: SourceKind::from_path(Path::new(source_path))
                .format_label()
                .unwrap_or("text")
                .to_string(),
            page: 0,
            chunk_index: 0,
            metadata: "{}".to_string(),
            embedding: vec![0.1, 0.2],
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn test_clear_requires_explicit_confirmation() {
        let err = clear(Some("danger"), false).await.unwrap_err().to_string();
        assert!(err.contains("--yes"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn test_delete_removes_matching_source() {
        let dir = tempfile::tempdir().unwrap();
        with_test_env(dir.path(), None, || async {
            let store = store_dir(Some("delete-one")).unwrap();
            ensure_store_layout(&store).unwrap();
            let db = connect_db(&store).await.unwrap();
            replace_source_rows(
                &db,
                &[
                    sample_row("docs/a.md", "alpha"),
                    sample_row("docs/b.md", "beta"),
                ],
                &["docs/a.md".to_string(), "docs/b.md".to_string()],
            )
            .await
            .unwrap();

            delete(Some("delete-one"), "docs/a.md".to_string())
                .await
                .unwrap();

            let remaining = list_indexed_sources(&db).await.unwrap();
            assert_eq!(remaining.len(), 1);
            assert_eq!(remaining[0].source_path, "docs/b.md");
        })
        .await;
    }

    #[tokio::test(flavor = "current_thread")]
    async fn test_build_prune_report_marks_missing_sources_without_deleting_by_default() {
        let dir = tempfile::tempdir().unwrap();
        with_test_env(dir.path(), None, || async {
            let live_path = dir.path().join("live.txt");
            std::fs::write(&live_path, "live").unwrap();
            let missing_path = dir.path().join("missing.txt");
            let store = store_dir(Some("prune-preview")).unwrap();
            ensure_store_layout(&store).unwrap();
            let db = connect_db(&store).await.unwrap();
            replace_source_rows(
                &db,
                &[
                    sample_row(&live_path.display().to_string(), "alpha"),
                    sample_row(&missing_path.display().to_string(), "beta"),
                ],
                &[
                    live_path.display().to_string(),
                    missing_path.display().to_string(),
                ],
            )
            .await
            .unwrap();

            let report = build_prune_report(Some("prune-preview"), false)
                .await
                .unwrap();
            assert!(report.dry_run);
            assert_eq!(report.stale_sources.len(), 1);
            assert_eq!(
                report.stale_sources[0].source_path,
                missing_path.display().to_string()
            );
            assert_eq!(report.deleted_sources, 0);

            let remaining = list_indexed_sources(&db).await.unwrap();
            assert_eq!(remaining.len(), 2);
        })
        .await;
    }

    #[tokio::test(flavor = "current_thread")]
    async fn test_build_prune_report_deletes_missing_sources_when_applied() {
        let dir = tempfile::tempdir().unwrap();
        with_test_env(dir.path(), None, || async {
            let live_path = dir.path().join("live.txt");
            std::fs::write(&live_path, "live").unwrap();
            let missing_path = dir.path().join("missing.txt");
            let store = store_dir(Some("prune-apply")).unwrap();
            ensure_store_layout(&store).unwrap();
            let db = connect_db(&store).await.unwrap();
            replace_source_rows(
                &db,
                &[
                    sample_row(&live_path.display().to_string(), "alpha"),
                    sample_row(&missing_path.display().to_string(), "beta"),
                ],
                &[
                    live_path.display().to_string(),
                    missing_path.display().to_string(),
                ],
            )
            .await
            .unwrap();

            let report = build_prune_report(Some("prune-apply"), true).await.unwrap();
            assert!(!report.dry_run);
            assert_eq!(report.deleted_sources, 1);
            assert_eq!(report.deleted_chunks, 1);

            let remaining = list_indexed_sources(&db).await.unwrap();
            assert_eq!(remaining.len(), 1);
            assert_eq!(remaining[0].source_path, live_path.display().to_string());
        })
        .await;
    }
}
