use crate::commands::stat::fmt_count;
use crate::config::{ensure_store_layout, store_dir};
use crate::store::{connect_db, list_indexed_sources, IndexedSource};
use anyhow::Result;
use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct SourcesReport {
    pub store: String,
    pub total_sources: usize,
    pub sources: Vec<IndexedSource>,
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

async fn build_report(name: Option<&str>) -> Result<SourcesReport> {
    let store = store_dir(name)?;
    ensure_store_layout(&store)?;
    let db = connect_db(&store).await?;
    let sources = list_indexed_sources(&db).await?;

    Ok(SourcesReport {
        store: store.display().to_string(),
        total_sources: sources.len(),
        sources,
    })
}

fn print_human(report: &SourcesReport) {
    println!("Indexed sources");
    println!("  store: {}", report.store);
    println!("  sources: {}", report.total_sources);

    if report.sources.is_empty() {
        println!("  no indexed sources");
        return;
    }

    for source in &report.sources {
        let mut details = vec![
            source.format.clone(),
            format!("{} chunks", fmt_count(source.chunks)),
            format!("~{} tokens", fmt_count(source.estimated_tokens)),
            format!("{} chars", fmt_count(source.chars)),
        ];
        if source.page_count > 0 {
            details.push(format!("{} pages", fmt_count(source.page_count)));
        }

        println!("    - {}  [{}]", source.source_path, details.join(", "));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source_kind::SourceKind;
    use crate::store::{connect_db, replace_source_rows, ChunkRow};
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
    async fn test_build_report_lists_indexed_sources() {
        let dir = tempfile::tempdir().unwrap();
        with_test_env(dir.path(), None, || async {
            let store = store_dir(Some("inspect")).unwrap();
            ensure_store_layout(&store).unwrap();
            let db = connect_db(&store).await.unwrap();
            replace_source_rows(
                &db,
                &[
                    sample_row("docs/a.md", "alpha"),
                    sample_row("docs/a.md", "beta"),
                    sample_row("notes/todo.txt", "gamma"),
                ],
                &["docs/a.md".to_string(), "notes/todo.txt".to_string()],
            )
            .await
            .unwrap();

            let report = build_report(Some("inspect")).await.unwrap();
            assert_eq!(report.total_sources, 2);
            assert_eq!(report.sources[0].source_path, "docs/a.md");
            assert_eq!(report.sources[0].chunks, 2);
            assert_eq!(report.sources[0].format, "markdown");
            assert_eq!(report.sources[1].source_path, "notes/todo.txt");
        })
        .await;
    }
}
