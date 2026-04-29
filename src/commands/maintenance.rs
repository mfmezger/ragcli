use crate::commands::stat::fmt_count;
use crate::config::{ensure_store_layout, store_dir};
use crate::store::{
    clear_store_rows, connect_db, delete_source_rows, list_indexed_source_summaries,
    load_indexed_source_summary, IndexedSourceSummary,
};
use crate::ui::{self, Panel};
use anyhow::{bail, Context, Result};
use serde::Serialize;
use std::path::PathBuf;

#[derive(Debug, Serialize)]
pub struct PruneReport {
    pub store: String,
    pub dry_run: bool,
    pub stale_sources: Vec<IndexedSourceSummary>,
    pub deleted_sources: usize,
    pub deleted_chunks: usize,
}

pub async fn delete(name: Option<&str>, path: String) -> Result<()> {
    let store = store_dir(name)?;
    ensure_store_layout(&store)?;
    let db = connect_db(&store).await?;
    let Some(source) = load_indexed_source_summary(&db, &path).await? else {
        let mut panel = Panel::new("Delete Source");
        panel.kv("source", path, 8);
        panel.kv("status", ui::warn("not indexed"), 8);
        panel.render();
        return Ok(());
    };

    delete_source_rows(&db, std::slice::from_ref(&source.source_path)).await?;
    let mut panel = Panel::new("Delete Source");
    panel.kv("source", &source.source_path, 8);
    panel.kv("format", &source.format, 8);
    panel.kv("chunks", fmt_count(source.chunks), 8);
    panel.kv("status", ui::ok("deleted"), 8);
    panel.render();
    Ok(())
}

pub async fn clear(name: Option<&str>, yes: bool) -> Result<()> {
    if !yes {
        bail!("clear is destructive; rerun with --yes to remove all indexed sources from the selected store");
    }

    let store = store_dir(name)?;
    ensure_store_layout(&store)?;
    let db = connect_db(&store).await?;
    let sources = list_indexed_source_summaries(&db).await?;
    if sources.is_empty() {
        let mut panel = Panel::new("Clear Store");
        panel.kv("store", store.display().to_string(), 8);
        panel.kv("status", ui::warn("already empty"), 8);
        panel.render();
        return Ok(());
    }

    let deleted_sources = sources.len();
    let deleted_chunks = sources.iter().map(|source| source.chunks).sum::<usize>();
    clear_store_rows(&db).await?;

    let mut panel = Panel::new("Clear Store");
    panel.kv("store", store.display().to_string(), 8);
    panel.kv("sources", fmt_count(deleted_sources), 8);
    panel.kv("chunks", fmt_count(deleted_chunks), 8);
    panel.kv("status", ui::ok("cleared"), 8);
    panel.render();
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
    let mut stale_sources = Vec::new();
    for source in list_indexed_source_summaries(&db).await? {
        if source_is_missing(&source)? {
            stale_sources.push(source);
        }
    }
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

fn source_prune_path(source: &IndexedSourceSummary) -> Result<PathBuf> {
    if let Some(path) = &source.source_absolute_path {
        return Ok(PathBuf::from(path));
    }

    let path = PathBuf::from(&source.source_path);
    if path.is_absolute() {
        Ok(path)
    } else {
        bail!(
            "cannot safely prune relative source path {}; reindex it with a newer ragcli before pruning",
            source.source_path
        )
    }
}

fn source_is_missing(source: &IndexedSourceSummary) -> Result<bool> {
    let path = source_prune_path(source)?;
    Ok(!path
        .try_exists()
        .with_context(|| format!("check whether source exists: {}", path.display()))?)
}

fn print_prune_human(report: &PruneReport) {
    ui::command_header(
        if report.dry_run {
            "ragcli prune"
        } else {
            "ragcli prune --apply"
        },
        "",
    );

    let mut summary = Panel::new(if report.dry_run {
        "Prune Preview"
    } else {
        "Prune Complete"
    });
    summary.kv("store", &report.store, 8);
    summary.kv("stale", report.stale_sources.len().to_string(), 8);

    if report.stale_sources.is_empty() {
        summary.kv("status", ui::ok("nothing to prune"), 8);
        summary.render();
        return;
    }

    if report.dry_run {
        summary.kv("status", ui::warn("dry run"), 8);
        summary.hint("Rerun with `--apply` to remove these rows.");
    } else {
        summary.kv("sources", fmt_count(report.deleted_sources), 8);
        summary.kv("chunks", fmt_count(report.deleted_chunks), 8);
        summary.kv("status", ui::ok("removed stale rows"), 8);
    }
    summary.render();

    println!();
    ui::render_table(
        "Stale Sources",
        &["Source", "Format", "Chunks", "Pages"],
        report
            .stale_sources
            .iter()
            .map(|source| {
                vec![
                    source.source_path.clone(),
                    source.format.clone(),
                    fmt_count(source.chunks),
                    if source.page_count > 0 {
                        fmt_count(source.page_count)
                    } else {
                        "-".to_string()
                    },
                ]
            })
            .collect(),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source_kind::SourceKind;
    use crate::store::{
        connect_db, list_indexed_source_summaries, list_indexed_sources, replace_source_rows,
        ChunkRow,
    };
    use crate::test_support::with_test_env;
    use std::path::Path;

    fn sample_row(source_path: &str, text: &str) -> ChunkRow {
        sample_row_with_metadata(source_path, text, "{}")
    }

    fn sample_row_with_metadata(source_path: &str, text: &str, metadata: &str) -> ChunkRow {
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
            metadata: metadata.to_string(),
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
    async fn test_clear_removes_all_rows() {
        let dir = tempfile::tempdir().unwrap();
        with_test_env(dir.path(), None, || async {
            let store = store_dir(Some("clear-store")).unwrap();
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

            clear(Some("clear-store"), true).await.unwrap();

            assert!(list_indexed_source_summaries(&db).await.unwrap().is_empty());
        })
        .await;
    }

    #[tokio::test(flavor = "current_thread")]
    async fn test_build_prune_report_rejects_relative_sources_without_absolute_metadata() {
        let dir = tempfile::tempdir().unwrap();
        with_test_env(dir.path(), None, || async {
            let store = store_dir(Some("prune-relative")).unwrap();
            ensure_store_layout(&store).unwrap();
            let db = connect_db(&store).await.unwrap();
            replace_source_rows(
                &db,
                &[sample_row("docs/a.md", "alpha")],
                &["docs/a.md".to_string()],
            )
            .await
            .unwrap();

            let err = build_prune_report(Some("prune-relative"), false)
                .await
                .unwrap_err()
                .to_string();
            assert!(err.contains("cannot safely prune relative source path docs/a.md"));
        })
        .await;
    }

    #[tokio::test(flavor = "current_thread")]
    async fn test_build_prune_report_uses_absolute_metadata_for_relative_sources() {
        let dir = tempfile::tempdir().unwrap();
        with_test_env(dir.path(), None, || async {
            let live_path = dir.path().join("docs").join("a.md");
            std::fs::create_dir_all(live_path.parent().unwrap()).unwrap();
            std::fs::write(&live_path, "live").unwrap();
            let store = store_dir(Some("prune-absolute")).unwrap();
            ensure_store_layout(&store).unwrap();
            let db = connect_db(&store).await.unwrap();
            let metadata = format!(r#"{{"source_absolute_path":"{}"}}"#, live_path.display());
            replace_source_rows(
                &db,
                &[sample_row_with_metadata("docs/a.md", "alpha", &metadata)],
                &["docs/a.md".to_string()],
            )
            .await
            .unwrap();

            let report = build_prune_report(Some("prune-absolute"), false)
                .await
                .unwrap();
            assert!(report.stale_sources.is_empty());
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
