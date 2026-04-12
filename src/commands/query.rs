use crate::cli::QueryModeArg;
use crate::config::{
    ensure_store_layout, load_or_create_config, resolve_model_name, store_dir, Config,
};
use crate::models::{Embedder, Generator};
use crate::store::{self, build_retrieval_filter, connect_db, ensure_fts_index, load_metadata};
use anyhow::{Context, Result};
use arrow_array::{Float32Array, Float64Array, Int32Array, RecordBatch, StringArray};
use futures::TryStreamExt;
use lancedb::index::scalar::FullTextSearchQuery;
use lancedb::query::{ExecutableQuery, QueryBase};
use std::path::PathBuf;

#[derive(Debug)]
pub struct QueryCommand {
    pub question: String,
    pub mode: QueryModeArg,
    pub top_k: usize,
    pub fetch_k: usize,
    pub max_iterations: usize,
    pub rewrite: bool,
    pub rerank: bool,
    pub show_context: bool,
    pub show_plan: bool,
    pub show_scores: bool,
    pub show_citations: bool,
    pub show_trace: bool,
    pub source: Option<String>,
    pub path_prefix: Option<String>,
    pub page: Option<i32>,
    pub format: Option<String>,
    pub gen_model: Option<String>,
    pub max_tokens: usize,
}

struct QueryRuntime {
    store: PathBuf,
    cfg: Config,
    gen_model_name: String,
    embed_model_name: String,
}

#[derive(Clone, Debug, PartialEq)]
struct QueryHit {
    context: String,
    source_path: String,
    page: Option<i32>,
    format: Option<String>,
    score: Option<f64>,
}

#[derive(Debug)]
struct SimpleQueryResult {
    requested_mode: QueryModeArg,
    execution_label: &'static str,
    hits: Vec<QueryHit>,
    trace: Vec<String>,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum QueryExecutionPath {
    Simple,
    AgenticStub,
}

pub async fn run(name: Option<&str>, command: QueryCommand) -> Result<()> {
    let runtime = prepare_runtime(name, &command)?;

    let result = match execution_path(command.mode) {
        QueryExecutionPath::Simple => run_simple_query(&runtime, &command).await?,
        QueryExecutionPath::AgenticStub => run_agentic_query_command(&runtime, &command).await?,
    };

    if result.hits.is_empty() {
        println!("No relevant context found in the local store.");
        return Ok(());
    }

    print_query_plan(&command, &result);
    print_query_trace(&command, &result);
    print_scores(&command, &result);
    print_citations(&command, &result);
    print_contexts(&command, &result);

    let answer = generate_answer(&runtime, &command, &result).await?;
    println!();
    println!("{}", store::strip_thinking(&answer).trim());
    Ok(())
}

fn prepare_runtime(name: Option<&str>, command: &QueryCommand) -> Result<QueryRuntime> {
    let store = store_dir(name)?;
    ensure_store_layout(&store)?;
    let cfg = load_or_create_config(&store)?;

    let gen_model_name = command
        .gen_model
        .clone()
        .unwrap_or_else(|| resolve_model_name(&store, &cfg.models.chat));
    let embed_model_name = resolve_model_name(&store, &cfg.models.embed);
    let metadata = load_metadata(&store)?;
    metadata.validate_query_model(&embed_model_name)?;

    Ok(QueryRuntime {
        store,
        cfg,
        gen_model_name,
        embed_model_name,
    })
}

async fn run_simple_query(
    runtime: &QueryRuntime,
    command: &QueryCommand,
) -> Result<SimpleQueryResult> {
    let hits = retrieve_hits(runtime, &command.question, command).await?;

    Ok(SimpleQueryResult {
        requested_mode: command.mode,
        execution_label: mode_label(command.mode),
        hits,
        trace: vec![format!(
            "mode {} uses the current hybrid retrieval pipeline",
            mode_label(command.mode)
        )],
    })
}

async fn run_agentic_query_command(
    runtime: &QueryRuntime,
    command: &QueryCommand,
) -> Result<SimpleQueryResult> {
    let hits = retrieve_hits(runtime, &command.question, command).await?;

    Ok(SimpleQueryResult {
        requested_mode: command.mode,
        execution_label: "hybrid",
        hits,
        trace: vec![
            format!(
                "requested {} mode routed through the agentic query stub",
                mode_label(command.mode)
            ),
            "current implementation falls back to the hybrid retrieval pipeline".to_string(),
            format!(
                "max_iterations reserved for future agent loops: {}",
                command.max_iterations
            ),
        ],
    })
}

async fn retrieve_hits(
    runtime: &QueryRuntime,
    question: &str,
    command: &QueryCommand,
) -> Result<Vec<QueryHit>> {
    let db = connect_db(&runtime.store).await?;
    let table = db
        .open_table(store::DEFAULT_TABLE_NAME)
        .execute()
        .await
        .context("open table")?;
    ensure_fts_index(&table, false).await?;

    let embedder = Embedder::new(
        runtime.cfg.ollama.base_url.clone(),
        runtime.embed_model_name.clone(),
    );
    let embedding = embedder.embed(question).await?;

    let mut query = table
        .query()
        .full_text_search(FullTextSearchQuery::new(question.to_string()))
        .nearest_to(embedding.as_slice())?
        .limit(retrieval_limit(command));

    if let Some(filter) = build_retrieval_filter(
        command.source.as_deref(),
        command.path_prefix.as_deref(),
        command.page,
        command.format.as_deref(),
    ) {
        query = query.only_if(filter);
    }

    let batches: Vec<RecordBatch> = query.execute().await?.try_collect::<Vec<_>>().await?;

    let mut hits = extract_hits(&batches)?;
    hits.truncate(command.top_k);
    Ok(hits)
}

fn extract_hits(batches: &[RecordBatch]) -> Result<Vec<QueryHit>> {
    let mut hits = Vec::new();

    for batch in batches {
        let text_col = batch
            .column_by_name("chunk_text")
            .context("chunk_text column missing")?
            .as_any()
            .downcast_ref::<StringArray>()
            .context("chunk_text column type")?;
        let source_col = batch
            .column_by_name("source_path")
            .context("source_path column missing")?
            .as_any()
            .downcast_ref::<StringArray>()
            .context("source_path column type")?;
        let page_col = batch
            .column_by_name("page")
            .and_then(|column| column.as_any().downcast_ref::<Int32Array>());
        let format_col = batch
            .column_by_name("format")
            .and_then(|column| column.as_any().downcast_ref::<StringArray>());

        for row in 0..batch.num_rows() {
            let source_path = source_col.value(row).to_string();
            let context = format!("Source: {}\n{}", source_path, text_col.value(row));
            hits.push(QueryHit {
                context,
                source_path,
                page: page_col
                    .map(|column| column.value(row))
                    .filter(|page| *page > 0),
                format: format_col.map(|column| column.value(row).to_string()),
                score: score_at(batch, row),
            });
        }
    }

    Ok(hits)
}

fn score_at(batch: &RecordBatch, row: usize) -> Option<f64> {
    for name in ["_score", "score", "_distance", "distance"] {
        let column = batch.column_by_name(name)?;
        if let Some(values) = column.as_any().downcast_ref::<Float32Array>() {
            return Some(values.value(row) as f64);
        }
        if let Some(values) = column.as_any().downcast_ref::<Float64Array>() {
            return Some(values.value(row));
        }
    }
    None
}

fn print_query_plan(command: &QueryCommand, result: &SimpleQueryResult) {
    if !command.show_plan {
        return;
    }

    println!("Query plan:");
    println!("  requested_mode: {}", mode_label(result.requested_mode));
    println!("  execution_path: {}", result.execution_label);
    println!("  top_k: {}", command.top_k);
    println!("  fetch_k: {}", command.fetch_k);
    println!("  max_iterations: {}", command.max_iterations);
    println!("  rewrite: {}", command.rewrite);
    println!("  rerank: {}", command.rerank);
    println!();
}

fn print_query_trace(command: &QueryCommand, result: &SimpleQueryResult) {
    if !command.show_trace {
        return;
    }

    println!("Query trace:");
    for entry in &result.trace {
        println!("  - {entry}");
    }
    println!("  - retrieved_hits: {}", result.hits.len());
    println!();
}

fn print_scores(command: &QueryCommand, result: &SimpleQueryResult) {
    if !command.show_scores {
        return;
    }

    println!("Scores:");
    for (idx, hit) in result.hits.iter().enumerate() {
        match hit.score {
            Some(score) => println!("  [{}] {:.6} {}", idx + 1, score, hit.source_path),
            None => println!("  [{}] unavailable {}", idx + 1, hit.source_path),
        }
    }
    println!();
}

fn print_citations(command: &QueryCommand, result: &SimpleQueryResult) {
    if !command.show_citations {
        return;
    }

    println!("Citations:");
    for (idx, hit) in result.hits.iter().enumerate() {
        match (&hit.format, hit.page) {
            (Some(format), Some(page)) => {
                println!(
                    "  [{}] {} (format: {}, page: {})",
                    idx + 1,
                    hit.source_path,
                    format,
                    page
                )
            }
            (Some(format), None) => {
                println!("  [{}] {} (format: {})", idx + 1, hit.source_path, format)
            }
            (None, Some(page)) => println!("  [{}] {} (page: {})", idx + 1, hit.source_path, page),
            (None, None) => println!("  [{}] {}", idx + 1, hit.source_path),
        }
    }
    println!();
}

fn print_contexts(command: &QueryCommand, result: &SimpleQueryResult) {
    if !command.show_context {
        return;
    }

    println!("Retrieved context:");
    for (idx, hit) in result.hits.iter().enumerate() {
        let compact = hit.context.replace('\n', " ");
        let preview: String = compact.chars().take(220).collect();
        println!("  [{}] {}", idx + 1, preview);
    }
    println!();
}

async fn generate_answer(
    runtime: &QueryRuntime,
    command: &QueryCommand,
    result: &SimpleQueryResult,
) -> Result<String> {
    let generator = Generator::new(
        runtime.cfg.ollama.base_url.clone(),
        runtime.gen_model_name.clone(),
    );
    let contexts = result
        .hits
        .iter()
        .map(|hit| hit.context.clone())
        .collect::<Vec<_>>();
    generator
        .generate_answer(&contexts, &command.question, command.max_tokens)
        .await
}

fn execution_path(mode: QueryModeArg) -> QueryExecutionPath {
    match mode {
        QueryModeArg::Naive | QueryModeArg::Hybrid => QueryExecutionPath::Simple,
        QueryModeArg::Agentic | QueryModeArg::Local | QueryModeArg::Global | QueryModeArg::Mix => {
            QueryExecutionPath::AgenticStub
        }
    }
}

fn retrieval_limit(command: &QueryCommand) -> usize {
    command.fetch_k.max(command.top_k).max(1)
}

fn mode_label(mode: QueryModeArg) -> &'static str {
    match mode {
        QueryModeArg::Naive => "naive",
        QueryModeArg::Hybrid => "hybrid",
        QueryModeArg::Agentic => "agentic",
        QueryModeArg::Local => "local",
        QueryModeArg::Global => "global",
        QueryModeArg::Mix => "mix",
    }
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
                QueryCommand {
                    question: "What is this project?".to_string(),
                    mode: QueryModeArg::Hybrid,
                    top_k: 5,
                    fetch_k: 20,
                    max_iterations: 2,
                    rewrite: false,
                    rerank: false,
                    show_context: true,
                    show_plan: false,
                    show_scores: false,
                    show_citations: false,
                    show_trace: false,
                    source: None,
                    path_prefix: None,
                    page: None,
                    format: None,
                    gen_model: None,
                    max_tokens: 64,
                },
            )
            .await
            .unwrap();
        })
        .await;
    }

    #[test]
    fn test_retrieval_limit_uses_fetch_k_when_larger_than_top_k() {
        let command = QueryCommand {
            question: "What is ragcli?".to_string(),
            mode: QueryModeArg::Hybrid,
            top_k: 5,
            fetch_k: 20,
            max_iterations: 2,
            rewrite: false,
            rerank: false,
            show_context: false,
            show_plan: false,
            show_scores: false,
            show_citations: false,
            show_trace: false,
            source: None,
            path_prefix: None,
            page: None,
            format: None,
            gen_model: None,
            max_tokens: 256,
        };

        assert_eq!(retrieval_limit(&command), 20);
    }

    #[test]
    fn test_execution_path_routes_agentic_modes_to_stub() {
        assert_eq!(
            execution_path(QueryModeArg::Hybrid),
            QueryExecutionPath::Simple
        );
        assert_eq!(
            execution_path(QueryModeArg::Naive),
            QueryExecutionPath::Simple
        );
        assert_eq!(
            execution_path(QueryModeArg::Agentic),
            QueryExecutionPath::AgenticStub
        );
        assert_eq!(
            execution_path(QueryModeArg::Local),
            QueryExecutionPath::AgenticStub
        );
        assert_eq!(
            execution_path(QueryModeArg::Global),
            QueryExecutionPath::AgenticStub
        );
        assert_eq!(
            execution_path(QueryModeArg::Mix),
            QueryExecutionPath::AgenticStub
        );
    }
}
