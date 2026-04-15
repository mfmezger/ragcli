use crate::cli::PdfParserArg;
use crate::config::{
    ensure_store_layout, load_or_create_config, normalize_chunk_settings, resolve_model_name,
    store_dir,
};
use crate::ingest::{ingest_path, PdfParser};
use crate::models::{Embedder, VisionCaptioner};
use crate::store::{connect_db, ensure_metadata, load_source_fingerprints, replace_source_rows};
use anyhow::Result;
use std::path::PathBuf;
use std::time::Instant;

pub async fn run(
    name: Option<&str>,
    path: PathBuf,
    chunk_size: Option<usize>,
    chunk_overlap: Option<usize>,
    embed_model: Option<String>,
    pdf_parser: Option<PdfParserArg>,
    exclude: Vec<String>,
    include_hidden: bool,
    force: bool,
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
    let db = connect_db(&store).await?;
    let existing_fingerprints = if force {
        Default::default()
    } else {
        load_source_fingerprints(&db).await?
    };
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
        &existing_fingerprints,
        force,
    )
    .await?;

    if let Some(dim) = result.embedding_dim {
        ensure_metadata(&store, &embed_model_name, dim, size, overlap)?;
    }

    if !result.source_paths.is_empty() {
        replace_source_rows(&db, &result.rows, &result.source_paths).await?;
    }

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commands::stat;
    use crate::store::{connect_db, extract_contexts, DEFAULT_TABLE_NAME};
    use crate::test_support::{sequential_json_server, with_test_env};
    use futures::TryStreamExt;
    use lancedb::query::ExecutableQuery;

    #[tokio::test(flavor = "current_thread")]
    async fn test_run_indexes_file_with_mock_ollama() {
        let dir = tempfile::tempdir().unwrap();
        let docs = dir.path().join("docs");
        std::fs::create_dir_all(&docs).unwrap();
        let input = docs.join("note.txt");
        std::fs::write(&input, "The project is a local RAG CLI.").unwrap();

        let server = sequential_json_server(vec![r#"{"embeddings":[[0.1,0.2]]}"#]);
        with_test_env(dir.path(), Some(&server), || async {
            run(
                Some("e2e"),
                input.clone(),
                Some(200),
                Some(0),
                None,
                None,
                Vec::new(),
                false,
                false,
            )
            .await
            .unwrap();
            stat::run(Some("e2e"), false).await.unwrap();
        })
        .await;
    }

    #[tokio::test(flavor = "current_thread")]
    async fn test_run_skips_unchanged_file_without_reembedding() {
        let dir = tempfile::tempdir().unwrap();
        let docs = dir.path().join("docs");
        std::fs::create_dir_all(&docs).unwrap();
        let input = docs.join("note.txt");
        std::fs::write(&input, "The project is a local RAG CLI.").unwrap();

        let first_server = sequential_json_server(vec![r#"{"embeddings":[[0.1,0.2]]}"#]);
        with_test_env(dir.path(), Some(&first_server), || async {
            run(
                Some("e2e"),
                input.clone(),
                Some(200),
                Some(0),
                None,
                None,
                Vec::new(),
                false,
                false,
            )
            .await
            .unwrap();
        })
        .await;

        with_test_env(dir.path(), Some("http://127.0.0.1:9"), || async {
            run(
                Some("e2e"),
                input.clone(),
                Some(200),
                Some(0),
                None,
                None,
                Vec::new(),
                false,
                false,
            )
            .await
            .unwrap();
        })
        .await;
    }

    #[tokio::test(flavor = "current_thread")]
    async fn test_run_replaces_rows_when_file_changes() {
        let dir = tempfile::tempdir().unwrap();
        let docs = dir.path().join("docs");
        std::fs::create_dir_all(&docs).unwrap();
        let input = docs.join("note.txt");
        std::fs::write(&input, "old content").unwrap();

        let first_server = sequential_json_server(vec![r#"{"embeddings":[[0.1,0.2]]}"#]);
        with_test_env(dir.path(), Some(&first_server), || async {
            run(
                Some("e2e"),
                input.clone(),
                Some(200),
                Some(0),
                None,
                None,
                Vec::new(),
                false,
                false,
            )
            .await
            .unwrap();
        })
        .await;

        std::thread::sleep(std::time::Duration::from_millis(2));
        std::fs::write(&input, "new content").unwrap();

        let second_server = sequential_json_server(vec![r#"{"embeddings":[[0.3,0.4]]}"#]);
        with_test_env(dir.path(), Some(&second_server), || async {
            run(
                Some("e2e"),
                input.clone(),
                Some(200),
                Some(0),
                None,
                None,
                Vec::new(),
                false,
                false,
            )
            .await
            .unwrap();
        })
        .await;

        with_test_env(dir.path(), None, || async {
            let store = store_dir(Some("e2e")).unwrap();
            let db = connect_db(&store).await.unwrap();
            let table = db.open_table(DEFAULT_TABLE_NAME).execute().await.unwrap();
            let batches = table
                .query()
                .execute()
                .await
                .unwrap()
                .try_collect::<Vec<_>>()
                .await
                .unwrap();
            let contexts = extract_contexts(&batches).unwrap();

            assert_eq!(contexts.len(), 1);
            assert!(contexts[0].contains("new content"));
            assert!(!contexts[0].contains("old content"));
        })
        .await;
    }
}
