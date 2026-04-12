use crate::citation::{labeled_contexts, render_citations};
use crate::cli::QueryModeArg;
use crate::config::{
    ensure_store_layout, load_or_create_config, resolve_model_name, store_dir, Config,
};
use crate::models::{Embedder, Generator};
use crate::retrieval::{
    apply_rerank_order, merge_candidates, prune_candidates, RetrievalCandidate,
};
use crate::rewrite::{rewrite_query_for_retrieval, trim_json_fences, QueryRewriteSet};
use crate::store::{self, build_retrieval_filter, connect_db, ensure_fts_index, load_metadata};
use anyhow::{Context, Result};
use arrow_array::{Float32Array, Float64Array, Int32Array, RecordBatch, StringArray};
use futures::TryStreamExt;
use lancedb::index::scalar::FullTextSearchQuery;
use lancedb::query::{ExecutableQuery, QueryBase};
use std::path::PathBuf;

const RERANK_MAX_TEXT_CHARS: usize = 600;
const RERANK_RESPONSE_TOKENS: usize = 384;

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

#[derive(Debug)]
struct SimpleQueryResult {
    requested_mode: QueryModeArg,
    execution_label: &'static str,
    rewrite_set: QueryRewriteSet,
    hits: Vec<RetrievalCandidate>,
    trace: Vec<String>,
}

#[derive(Debug, serde::Deserialize)]
struct RerankPayload {
    ranked_ids: Vec<RerankItem>,
}

#[derive(Debug, serde::Deserialize)]
struct RerankItem {
    id: String,
    score: f32,
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
    let mut trace = vec![format!(
        "mode {} uses the current hybrid retrieval pipeline",
        mode_label(command.mode)
    )];
    let rewrite_set = build_rewrite_set(runtime, command, &mut trace).await;
    let hits =
        retrieve_candidates(runtime, command, &rewrite_set.query_variants(), &mut trace).await?;

    Ok(SimpleQueryResult {
        requested_mode: command.mode,
        execution_label: mode_label(command.mode),
        rewrite_set,
        hits,
        trace,
    })
}

async fn run_agentic_query_command(
    runtime: &QueryRuntime,
    command: &QueryCommand,
) -> Result<SimpleQueryResult> {
    let mut trace = vec![
        format!(
            "requested {} mode routed through the agentic query stub",
            mode_label(command.mode)
        ),
        "current implementation falls back to the hybrid retrieval pipeline".to_string(),
        format!(
            "max_iterations reserved for future agent loops: {}",
            command.max_iterations
        ),
    ];
    let rewrite_set = build_rewrite_set(runtime, command, &mut trace).await;
    let hits =
        retrieve_candidates(runtime, command, &rewrite_set.query_variants(), &mut trace).await?;

    Ok(SimpleQueryResult {
        requested_mode: command.mode,
        execution_label: "hybrid",
        rewrite_set,
        hits,
        trace,
    })
}

async fn build_rewrite_set(
    runtime: &QueryRuntime,
    command: &QueryCommand,
    trace: &mut Vec<String>,
) -> QueryRewriteSet {
    if !command.rewrite {
        trace.push("rewrite disabled; using original query only".to_string());
        return QueryRewriteSet::fallback(&command.question);
    }

    let generator = Generator::new(
        runtime.cfg.ollama.base_url.clone(),
        runtime.gen_model_name.clone(),
    );
    match rewrite_query_for_retrieval(&generator, &command.question).await {
        Ok(rewrite_set) => {
            let variants = rewrite_set.query_variants();
            trace.push(format!(
                "rewrite enabled; {} retrieval query variant(s) prepared",
                variants.len()
            ));
            for variant in variants.iter().skip(1) {
                trace.push(format!("rewrite variant: {variant}"));
            }
            rewrite_set
        }
        Err(err) => {
            trace.push(format!(
                "rewrite failed; falling back to original query ({})",
                err.root_cause()
            ));
            QueryRewriteSet::fallback(&command.question)
        }
    }
}

async fn retrieve_candidates(
    runtime: &QueryRuntime,
    command: &QueryCommand,
    queries: &[String],
    trace: &mut Vec<String>,
) -> Result<Vec<RetrievalCandidate>> {
    let mut groups = Vec::new();
    for query in queries {
        groups.push(retrieve_candidates_for_query(runtime, query, command).await?);
    }

    let merged = merge_candidates(groups);
    trace.push(format!(
        "merged {} candidate(s) across {} retrieval query variant(s)",
        merged.len(),
        queries.len()
    ));

    let reranked = rerank_candidates(runtime, command, merged, trace).await;
    let pruned = prune_candidates(reranked, command.top_k);
    trace.push(format!("kept {} candidate(s) after pruning", pruned.len()));
    Ok(pruned)
}

async fn retrieve_candidates_for_query(
    runtime: &QueryRuntime,
    question: &str,
    command: &QueryCommand,
) -> Result<Vec<RetrievalCandidate>> {
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
    extract_candidates(&batches)
}

fn extract_candidates(batches: &[RecordBatch]) -> Result<Vec<RetrievalCandidate>> {
    let mut hits = Vec::new();

    for batch in batches {
        let id_col = batch
            .column_by_name("id")
            .and_then(|column| column.as_any().downcast_ref::<StringArray>());
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
        let metadata_col = batch
            .column_by_name("metadata")
            .and_then(|column| column.as_any().downcast_ref::<StringArray>());
        let page_col = batch
            .column_by_name("page")
            .and_then(|column| column.as_any().downcast_ref::<Int32Array>());
        let chunk_index_col = batch
            .column_by_name("chunk_index")
            .and_then(|column| column.as_any().downcast_ref::<Int32Array>());

        for row in 0..batch.num_rows() {
            let vector_score = vector_score_at(batch, row);
            let fused_score = relevance_score_at(batch, row).or(vector_score);
            hits.push(RetrievalCandidate {
                id: id_col
                    .map(|column| column.value(row).to_string())
                    .unwrap_or_default(),
                source_path: source_col.value(row).to_string(),
                chunk_text: text_col.value(row).to_string(),
                metadata: metadata_col
                    .map(|column| column.value(row).to_string())
                    .unwrap_or_default(),
                page: page_col.map(|column| column.value(row)).unwrap_or_default(),
                chunk_index: chunk_index_col
                    .map(|column| column.value(row))
                    .unwrap_or_default(),
                vector_score,
                keyword_score: relevance_score_at(batch, row),
                fused_score,
                rerank_score: None,
            });
        }
    }

    Ok(hits)
}

fn relevance_score_at(batch: &RecordBatch, row: usize) -> Option<f32> {
    for name in ["_score", "score"] {
        let column = batch.column_by_name(name)?;
        if let Some(values) = column.as_any().downcast_ref::<Float32Array>() {
            return Some(values.value(row));
        }
        if let Some(values) = column.as_any().downcast_ref::<Float64Array>() {
            return Some(values.value(row) as f32);
        }
    }
    None
}

fn vector_score_at(batch: &RecordBatch, row: usize) -> Option<f32> {
    if let Some(score) = relevance_score_at(batch, row) {
        return Some(score);
    }

    for name in ["_distance", "distance"] {
        let column = batch.column_by_name(name)?;
        let distance = if let Some(values) = column.as_any().downcast_ref::<Float32Array>() {
            values.value(row)
        } else if let Some(values) = column.as_any().downcast_ref::<Float64Array>() {
            values.value(row) as f32
        } else {
            continue;
        };
        return Some(distance_to_similarity(distance));
    }
    None
}

fn distance_to_similarity(distance: f32) -> f32 {
    1.0 / (1.0 + distance.max(0.0))
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
    println!(
        "  query_variants: {}",
        result.rewrite_set.query_variants().join(" | ")
    );
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
        println!(
            "  [{}] best={:.6} fused={} rerank={} {}",
            idx + 1,
            hit.best_score().unwrap_or_default(),
            format_score(hit.fused_score),
            format_score(hit.rerank_score),
            hit.source_path
        );
    }
    println!();
}

fn print_citations(command: &QueryCommand, result: &SimpleQueryResult) {
    if !command.show_citations {
        return;
    }

    println!("Citations:");
    for citation in render_citations(&result.hits) {
        println!("  {citation}");
    }
    println!();
}

fn print_contexts(command: &QueryCommand, result: &SimpleQueryResult) {
    if !command.show_context {
        return;
    }

    println!("Retrieved context:");
    for (idx, hit) in result.hits.iter().enumerate() {
        let compact = retrieval_context(hit).replace('\n', " ");
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
    let contexts = labeled_contexts(&result.hits);
    generator
        .generate_answer(&contexts, &command.question, command.max_tokens)
        .await
}

fn retrieval_context(hit: &RetrievalCandidate) -> String {
    format!("Source: {}\n{}", hit.source_path, hit.chunk_text)
}

async fn rerank_candidates(
    runtime: &QueryRuntime,
    command: &QueryCommand,
    candidates: Vec<RetrievalCandidate>,
    trace: &mut Vec<String>,
) -> Vec<RetrievalCandidate> {
    if !command.rerank || candidates.len() <= 1 {
        trace.push("rerank disabled; using fused retrieval order".to_string());
        return candidates;
    }

    let generator = Generator::new(
        runtime.cfg.ollama.base_url.clone(),
        runtime.gen_model_name.clone(),
    );
    match rerank_with_model(
        &generator,
        &command.question,
        &candidates,
        command.fetch_k.min(candidates.len()),
    )
    .await
    {
        Ok(reranked) => {
            trace.push(format!(
                "rerank enabled; reordered {} candidate(s)",
                reranked.len()
            ));
            reranked
        }
        Err(err) => {
            trace.push(format!(
                "rerank failed; falling back to fused retrieval order ({})",
                err.root_cause()
            ));
            candidates
        }
    }
}

async fn rerank_with_model(
    generator: &Generator,
    question: &str,
    candidates: &[RetrievalCandidate],
    top_n: usize,
) -> Result<Vec<RetrievalCandidate>> {
    let prompt = build_rerank_prompt(question, candidates);
    let response = generator
        .generate_json(
            "You rerank retrieval candidates for a local RAG CLI. Respond with strict JSON only and no markdown fences.",
            &prompt,
            RERANK_RESPONSE_TOKENS,
        )
        .await
        .context("generate rerank JSON")?;
    let ranked_ids = parse_rerank_payload(&response)?;
    Ok(apply_rerank_order(candidates, &ranked_ids, top_n))
}

fn build_rerank_prompt(question: &str, candidates: &[RetrievalCandidate]) -> String {
    let items = candidates
        .iter()
        .map(|candidate| {
            format!(
                "id={}\nsource={}\ntext={}",
                candidate.dedupe_key(),
                candidate.source_path,
                truncate_for_rerank(&candidate.chunk_text)
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n");

    format!(
        concat!(
            "Return a JSON object with key ranked_ids. ",
            "ranked_ids must be an array of objects with keys id and score. ",
            "Include only the most relevant candidates in descending order.\n\n",
            "Question: {}\n\nCandidates:\n{}"
        ),
        question, items
    )
}

fn parse_rerank_payload(raw: &str) -> Result<Vec<(String, f32)>> {
    let trimmed = trim_json_fences(raw);
    let payload: RerankPayload = serde_json::from_str(trimmed).context("parse rerank JSON")?;
    Ok(payload
        .ranked_ids
        .into_iter()
        .map(|item| (item.id.trim().to_string(), item.score))
        .filter(|(id, _)| !id.is_empty())
        .collect())
}

fn truncate_for_rerank(text: &str) -> String {
    let mut truncated = text.chars().take(RERANK_MAX_TEXT_CHARS).collect::<String>();
    if text.chars().count() > RERANK_MAX_TEXT_CHARS {
        truncated.push_str("...");
    }
    truncated
}

fn format_score(score: Option<f32>) -> String {
    score
        .map(|value| format!("{value:.6}"))
        .unwrap_or_else(|| "unavailable".to_string())
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

    #[tokio::test(flavor = "current_thread")]
    async fn test_rewrite_falls_back_to_original_query_when_json_is_invalid() {
        let dir = tempfile::tempdir().unwrap();
        let docs = dir.path().join("docs");
        std::fs::create_dir_all(&docs).unwrap();
        let input = docs.join("note.txt");
        std::fs::write(
            &input,
            "Configuration resolution happens during query execution.",
        )
        .unwrap();

        let index_server = sequential_json_server(vec![r#"{"embeddings":[[0.1,0.2]]}"#]);
        with_test_env(dir.path(), Some(&index_server), || async {
            index::run(
                Some("rewrite-fallback"),
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
            r#"{"message":{"content":"not json"}}"#,
            r#"{"embeddings":[[0.1,0.2]]}"#,
            r#"{"message":{"content":"Configuration resolution happens during query execution."}}"#,
        ]);
        with_test_env(dir.path(), Some(&query_server), || async {
            run(
                Some("rewrite-fallback"),
                QueryCommand {
                    question: "How does config resolution work?".to_string(),
                    mode: QueryModeArg::Hybrid,
                    top_k: 5,
                    fetch_k: 20,
                    max_iterations: 2,
                    rewrite: true,
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
                    max_tokens: 64,
                },
            )
            .await
            .unwrap();
        })
        .await;
    }

    #[test]
    fn test_parse_rerank_payload_accepts_json_fences() {
        let ranked = parse_rerank_payload(
            "```json\n{\"ranked_ids\":[{\"id\":\"a\",\"score\":0.9},{\"id\":\"b\",\"score\":0.3}]}\n```",
        )
        .unwrap();

        assert_eq!(ranked, vec![("a".to_string(), 0.9), ("b".to_string(), 0.3)]);
    }

    #[test]
    fn test_distance_to_similarity_makes_smaller_distances_score_higher() {
        assert!(distance_to_similarity(0.2) > distance_to_similarity(0.8));
    }

    #[test]
    fn test_truncate_for_rerank_limits_candidate_text() {
        let text = "a".repeat(RERANK_MAX_TEXT_CHARS + 25);
        let truncated = truncate_for_rerank(&text);
        assert_eq!(truncated.len(), RERANK_MAX_TEXT_CHARS + 3);
        assert!(truncated.ends_with("..."));
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
