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
use tracing::{field, Instrument};

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
    let command_span = tracing::info_span!(
        "index_command",
        store_name = name.unwrap_or("default"),
        path = %path.display(),
        requested_chunk_size = field::debug(chunk_size),
        requested_chunk_overlap = field::debug(chunk_overlap),
        requested_embed_model = field::debug(embed_model.as_deref()),
        pdf_parser = field::debug(pdf_parser),
        exclude_count = exclude.len(),
        include_hidden,
        force,
        indexed_files = field::Empty,
        total_chunks = field::Empty,
        skipped_files = field::Empty,
        error_count = field::Empty,
        elapsed_ms = field::Empty,
    );

    let command_span_inner = command_span.clone();
    async move {
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
        let vision_model_name = resolve_model_name(&store, &cfg.models.vision);
        command_span_inner.record("requested_chunk_size", size);
        command_span_inner.record("requested_chunk_overlap", overlap);
        command_span_inner.record("requested_embed_model", field::debug(&embed_model_name));

        let embedder = Embedder::new(cfg.ollama.base_url.clone(), embed_model_name.clone());
        let vision = VisionCaptioner::new(cfg.ollama.base_url.clone(), vision_model_name);
        let db = connect_db(&store).await?;
        let existing_fingerprints = if force {
            Default::default()
        } else {
            load_source_fingerprints(&db).await?
        };

        let parser = match pdf_parser.unwrap_or(PdfParserArg::Native) {
            PdfParserArg::Native => PdfParser::Native,
            PdfParserArg::Liteparse => PdfParser::Liteparse,
        };
        let ingest_span = tracing::info_span!(
            "ingest_path",
            path = %path.display(),
            chunk_size = size,
            chunk_overlap = overlap,
            parser = ?parser,
            include_hidden,
            force,
        );
        let result = ingest_path(
            &path,
            size,
            overlap,
            &embedder,
            Some(&vision),
            parser,
            &exclude,
            include_hidden,
            &existing_fingerprints,
            force,
        )
        .instrument(ingest_span)
        .await?;

        if let Some(dim) = result.embedding_dim {
            ensure_metadata(&store, &embed_model_name, dim, size, overlap)?;
        }

        if !result.source_paths.is_empty() {
            replace_source_rows(&db, &result.rows, &result.source_paths).await?;
        }

        let elapsed_ms = started.elapsed().as_millis();
        command_span_inner.record("indexed_files", result.stats.indexed_files);
        command_span_inner.record("total_chunks", result.stats.total_chunks);
        command_span_inner.record("skipped_files", result.stats.skipped_files);
        command_span_inner.record("error_count", result.stats.errors.len());
        command_span_inner.record("elapsed_ms", elapsed_ms as u64);

        println!(
            "Index complete: {} files, {} chunks, {} skipped, {}ms",
            result.stats.indexed_files,
            result.stats.total_chunks,
            result.stats.skipped_files,
            elapsed_ms
        );

        if !result.stats.errors.is_empty() {
            for err in &result.stats.errors {
                eprintln!("index error: {err}");
            }
        }

        Ok(())
    }
    .instrument(command_span)
    .await
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
