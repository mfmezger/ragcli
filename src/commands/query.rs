use crate::config::{ensure_store_layout, load_or_create_config, resolve_model_name, store_dir};
use crate::models::{Embedder, Generator};
use crate::store::{self, connect_db, ensure_fts_index, extract_contexts, load_metadata};
use anyhow::{Context, Result};
use futures::TryStreamExt;
use lancedb::index::scalar::FullTextSearchQuery;
use lancedb::query::{ExecutableQuery, QueryBase};

pub async fn run(
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commands::index;
    use crate::test_support::{sequential_json_server, with_test_env};

    #[tokio::test(flavor = "current_thread")]
    async fn test_run_answers_from_indexed_store_with_mock_ollama() {
        let dir = tempfile::tempdir().unwrap();
        let docs = dir.path().join("docs");
        std::fs::create_dir_all(&docs).unwrap();
        let input = docs.join("note.txt");
        std::fs::write(&input, "The project is a local RAG CLI.").unwrap();

        let index_server = sequential_json_server(vec![r#"{"embeddings":[[0.1,0.2]]}"#]);
        with_test_env(dir.path(), Some(&index_server), || async {
            index::run(
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
        })
        .await;

        let query_server = sequential_json_server(vec![
            r#"{"embeddings":[[0.1,0.2]]}"#,
            r#"{"message":{"content":"ragcli is a local RAG CLI."}}"#,
        ]);
        with_test_env(dir.path(), Some(&query_server), || async {
            run(
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
