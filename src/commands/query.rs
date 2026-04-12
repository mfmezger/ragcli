use crate::agent::{
    assess_evidence, build_query_plan, fallback_evidence_assessment, fallback_query_plan,
    fallback_support_check, verify_answer_support, AgentIteration, EvidenceAssessment,
    EvidenceVerdict, QueryPlan, RetrievalStrategy, SupportCheck,
};
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
use std::collections::BTreeSet;
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
    plan: Option<QueryPlan>,
    iterations: Vec<AgentIteration>,
    support_check: Option<SupportCheck>,
    answer: Option<String>,
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
    Agentic,
    AgenticStub,
}

pub async fn run(name: Option<&str>, command: QueryCommand) -> Result<()> {
    let runtime = prepare_runtime(name, &command)?;

    let result = match execution_path(command.mode) {
        QueryExecutionPath::Simple => run_simple_query(&runtime, &command).await?,
        QueryExecutionPath::Agentic => run_agentic_query_command(&runtime, &command).await?,
        QueryExecutionPath::AgenticStub => {
            run_agentic_stub_query_command(&runtime, &command).await?
        }
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

    let answer = match &result.answer {
        Some(answer) => answer.clone(),
        None => generate_answer(&runtime, &command, &result).await?,
    };
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
        plan: None,
        iterations: Vec::new(),
        support_check: None,
        answer: None,
        hits,
        trace,
    })
}

async fn run_agentic_query_command(
    runtime: &QueryRuntime,
    command: &QueryCommand,
) -> Result<SimpleQueryResult> {
    let generator = Generator::new(
        runtime.cfg.ollama.base_url.clone(),
        runtime.gen_model_name.clone(),
    );
    let mut trace = vec![format!(
        "mode {} uses the agentic retrieval loop",
        mode_label(command.mode)
    )];
    let plan = match build_query_plan(&generator, &command.question).await {
        Ok(plan) => {
            trace.push("planner produced a structured query plan".to_string());
            plan
        }
        Err(err) => {
            trace.push(format!(
                "planner failed; falling back to heuristic plan ({})",
                err.root_cause()
            ));
            fallback_query_plan(&command.question)
        }
    };

    let rewrite_set = build_rewrite_set(runtime, command, &mut trace).await;
    let mut iterations = Vec::new();
    let mut previous_keys = Vec::new();
    let mut active_queries = initial_agent_queries(&command.question, &plan, &rewrite_set);
    let mut hits = Vec::new();
    let mut accumulated_hits = Vec::new();

    for iteration in 1..=command.max_iterations.max(1) {
        trace.push(format!(
            "iteration {iteration}: querying {} variant(s)",
            active_queries.len()
        ));
        let iteration_hits =
            retrieve_candidates(runtime, command, &active_queries, &mut trace).await?;
        hits = iteration_hits.clone();
        accumulated_hits = merge_candidates([accumulated_hits, iteration_hits]);
        let kept_hits = prune_candidates(accumulated_hits.clone(), command.top_k);
        let assessment = match assess_evidence(&generator, &command.question, &kept_hits).await {
            Ok(assessment) => assessment,
            Err(err) => {
                trace.push(format!(
                    "evidence assessment failed; using heuristic fallback ({})",
                    err.root_cause()
                ));
                fallback_evidence_assessment(&kept_hits)
            }
        };
        let notes = assessment_notes(&assessment);
        iterations.push(AgentIteration {
            iteration,
            query_variants: active_queries.clone(),
            retrieved_count: hits.len(),
            kept_count: kept_hits.len(),
            sufficiency: assessment.verdict.clone(),
            notes: notes.clone(),
        });
        for note in notes {
            trace.push(format!("iteration {iteration}: {note}"));
        }

        if matches!(assessment.verdict, EvidenceVerdict::Sufficient) {
            trace.push(format!("iteration {iteration}: evidence marked sufficient"));
            break;
        }
        if iteration >= command.max_iterations.max(1) {
            trace.push("iteration budget exhausted".to_string());
            break;
        }

        let current_keys = kept_hits
            .iter()
            .map(|hit| hit.dedupe_key())
            .collect::<Vec<_>>();
        if !previous_keys.is_empty() && current_keys == previous_keys {
            trace.push("retrieval produced no material evidence change; halting".to_string());
            break;
        }
        previous_keys = current_keys;

        let next_queries =
            refine_agent_queries(&command.question, &plan, &rewrite_set, &assessment);
        if next_queries == active_queries {
            trace.push("no better follow-up queries available; halting".to_string());
            break;
        }
        active_queries = next_queries;
    }

    let final_hits = prune_candidates(accumulated_hits, command.top_k);
    let answer = generate_answer_from_hits(runtime, command, &final_hits).await?;
    let support_check =
        match verify_answer_support(&generator, &command.question, &answer, &final_hits).await {
            Ok(check) => check,
            Err(err) => {
                trace.push(format!(
                    "support verification failed; using heuristic fallback ({})",
                    err.root_cause()
                ));
                fallback_support_check(&answer, &hits)
            }
        };
    if support_check.supported {
        trace.push("answer support verification passed".to_string());
    } else {
        trace.push(format!(
            "answer support verification found {} unsupported claim(s)",
            support_check.unsupported_claims.len()
        ));
    }

    Ok(SimpleQueryResult {
        requested_mode: command.mode,
        execution_label: "agentic",
        rewrite_set,
        plan: Some(plan),
        iterations,
        support_check: Some(support_check),
        answer: Some(answer),
        hits: final_hits,
        trace,
    })
}

async fn run_agentic_stub_query_command(
    runtime: &QueryRuntime,
    command: &QueryCommand,
) -> Result<SimpleQueryResult> {
    let mut trace = vec![
        format!(
            "requested {} mode routed through the graph-mode stub",
            mode_label(command.mode)
        ),
        "current implementation falls back to the hybrid retrieval pipeline".to_string(),
    ];
    let rewrite_set = build_rewrite_set(runtime, command, &mut trace).await;
    let hits =
        retrieve_candidates(runtime, command, &rewrite_set.query_variants(), &mut trace).await?;

    Ok(SimpleQueryResult {
        requested_mode: command.mode,
        execution_label: "hybrid",
        rewrite_set,
        plan: None,
        iterations: Vec::new(),
        support_check: None,
        answer: None,
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
    if let Some(plan) = &result.plan {
        println!("  question_type: {}", question_type_label(plan));
        println!("  strategy: {}", strategy_label(plan));
        println!("  reasoning: {}", plan.reasoning);
        if !plan.subqueries.is_empty() {
            println!("  subqueries: {}", plan.subqueries.join(" | "));
        }
    }
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
    for iteration in &result.iterations {
        println!(
            "  - iteration {} summary: verdict={}, queries={}, kept={}",
            iteration.iteration,
            evidence_verdict_label(&iteration.sufficiency),
            iteration.query_variants.join(" | "),
            iteration.kept_count
        );
    }
    if let Some(check) = &result.support_check {
        println!(
            "  - support_check: {}",
            if check.supported {
                "supported"
            } else {
                "unsupported"
            }
        );
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
    generate_answer_from_hits(runtime, command, &result.hits).await
}

async fn generate_answer_from_hits(
    runtime: &QueryRuntime,
    command: &QueryCommand,
    hits: &[RetrievalCandidate],
) -> Result<String> {
    let generator = Generator::new(
        runtime.cfg.ollama.base_url.clone(),
        runtime.gen_model_name.clone(),
    );
    let contexts = labeled_contexts(hits);
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
    let ranked_positions = parse_rerank_payload(&response)?;
    let ranked_ids = resolve_rerank_positions(candidates, &ranked_positions);
    Ok(apply_rerank_order(candidates, &ranked_ids, top_n))
}

fn build_rerank_prompt(question: &str, candidates: &[RetrievalCandidate]) -> String {
    let items = candidates
        .iter()
        .enumerate()
        .map(|(idx, candidate)| {
            format!(
                "id={}\nsource={}\ntext={}",
                idx + 1,
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
            "Use the numeric candidate ids exactly as provided. ",
            "Include only the most relevant candidates in descending order.\n\n",
            "Question: {}\n\nCandidates:\n{}"
        ),
        question, items
    )
}

fn parse_rerank_payload(raw: &str) -> Result<Vec<(usize, f32)>> {
    let trimmed = trim_json_fences(raw);
    let payload: RerankPayload = serde_json::from_str(trimmed).context("parse rerank JSON")?;
    Ok(payload
        .ranked_ids
        .into_iter()
        .filter_map(|item| {
            let id = item.id.trim().parse::<usize>().ok()?;
            Some((id, item.score))
        })
        .collect())
}

fn resolve_rerank_positions(
    candidates: &[RetrievalCandidate],
    ranked_positions: &[(usize, f32)],
) -> Vec<(String, f32)> {
    ranked_positions
        .iter()
        .filter_map(|(position, score)| {
            let candidate = candidates.get(position.saturating_sub(1))?;
            Some((candidate.dedupe_key(), *score))
        })
        .collect()
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

fn initial_agent_queries(
    question: &str,
    plan: &QueryPlan,
    rewrite_set: &QueryRewriteSet,
) -> Vec<String> {
    match plan.strategy {
        RetrievalStrategy::Decompose if !plan.subqueries.is_empty() => plan.subqueries.clone(),
        _ => rewrite_set.query_variants(),
    }
    .into_iter()
    .chain(std::iter::once(question.to_string()))
    .collect::<BTreeSet<_>>()
    .into_iter()
    .collect()
}

fn refine_agent_queries(
    question: &str,
    plan: &QueryPlan,
    rewrite_set: &QueryRewriteSet,
    assessment: &EvidenceAssessment,
) -> Vec<String> {
    if !assessment.missing_aspects.is_empty() {
        return assessment
            .missing_aspects
            .iter()
            .cloned()
            .chain(std::iter::once(question.to_string()))
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
    }

    if matches!(plan.strategy, RetrievalStrategy::Decompose) && !plan.subqueries.is_empty() {
        return plan
            .subqueries
            .iter()
            .cloned()
            .chain(rewrite_set.query_variants())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
    }

    rewrite_set.query_variants()
}

fn assessment_notes(assessment: &EvidenceAssessment) -> Vec<String> {
    let mut notes = vec![format!(
        "evidence verdict={}",
        evidence_verdict_label(&assessment.verdict)
    )];
    if !assessment.missing_aspects.is_empty() {
        notes.push(format!(
            "missing_aspects={}",
            assessment.missing_aspects.join(" | ")
        ));
    }
    notes
}

fn evidence_verdict_label(verdict: &EvidenceVerdict) -> &'static str {
    match verdict {
        EvidenceVerdict::Sufficient => "sufficient",
        EvidenceVerdict::Partial => "partial",
        EvidenceVerdict::Insufficient => "insufficient",
    }
}

fn question_type_label(plan: &QueryPlan) -> &'static str {
    match plan.question_type {
        crate::agent::QuestionType::Lookup => "Lookup",
        crate::agent::QuestionType::Compare => "Compare",
        crate::agent::QuestionType::MultiHop => "MultiHop",
        crate::agent::QuestionType::Summary => "Summary",
        crate::agent::QuestionType::Exploratory => "Exploratory",
    }
}

fn strategy_label(plan: &QueryPlan) -> &'static str {
    match plan.strategy {
        RetrievalStrategy::Direct => "Direct",
        RetrievalStrategy::Rewrite => "Rewrite",
        RetrievalStrategy::Decompose => "Decompose",
        RetrievalStrategy::BroadThenRerank => "BroadThenRerank",
    }
}

fn execution_path(mode: QueryModeArg) -> QueryExecutionPath {
    match mode {
        QueryModeArg::Naive | QueryModeArg::Hybrid => QueryExecutionPath::Simple,
        QueryModeArg::Agentic => QueryExecutionPath::Agentic,
        QueryModeArg::Local | QueryModeArg::Global | QueryModeArg::Mix => {
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

    #[tokio::test(flavor = "current_thread")]
    async fn test_agentic_mode_retries_when_evidence_is_partial() {
        let dir = tempfile::tempdir().unwrap();
        let docs = dir.path().join("docs");
        std::fs::create_dir_all(&docs).unwrap();
        let input = docs.join("note.txt");
        std::fs::write(
            &input,
            "Config resolution interacts with metadata validation during query execution.",
        )
        .unwrap();

        let index_server = sequential_json_server(vec![r#"{"embeddings":[[0.1,0.2]]}"#]);
        with_test_env(dir.path(), Some(&index_server), || async {
            index::run(
                Some("agentic-retry"),
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
            r#"{"message":{"content":"{\"question_type\":\"MultiHop\",\"strategy\":\"Decompose\",\"reasoning\":\"needs two facts\",\"subqueries\":[\"config resolution\",\"metadata validation\"]}"}}"#,
            r#"{"message":{"content":"{\"semantic_variant\":\"How does config resolution interact with metadata validation?\",\"keyword_variant\":\"config resolution metadata validation\",\"subqueries\":[\"config resolution\",\"metadata validation\"]}"}}"#,
            r#"{"embeddings":[[0.1,0.2]]}"#,
            r#"{"embeddings":[[0.1,0.2]]}"#,
            r#"{"embeddings":[[0.1,0.2]]}"#,
            r#"{"message":{"content":"{\"verdict\":\"partial\",\"missing_aspects\":[\"interaction between config resolution and metadata validation\"]}"}}"#,
            r#"{"embeddings":[[0.1,0.2]]}"#,
            r#"{"embeddings":[[0.1,0.2]]}"#,
            r#"{"message":{"content":"{\"verdict\":\"sufficient\",\"missing_aspects\":[]}"}}"#,
            r#"{"message":{"content":"Config resolution interacts with metadata validation during query execution. [1]"}}"#,
            r#"{"message":{"content":"{\"supported\":true,\"unsupported_claims\":[],\"notes\":[\"evidence covers the answer\"]}"}}"#,
        ]);
        with_test_env(dir.path(), Some(&query_server), || async {
            run(
                Some("agentic-retry"),
                QueryCommand {
                    question: "How do config resolution and metadata validation interact?"
                        .to_string(),
                    mode: QueryModeArg::Agentic,
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
            "```json\n{\"ranked_ids\":[{\"id\":\"1\",\"score\":0.9},{\"id\":\"2\",\"score\":0.3}]}\n```",
        )
        .unwrap();

        assert_eq!(ranked, vec![(1, 0.9), (2, 0.3)]);
    }

    #[test]
    fn test_resolve_rerank_positions_maps_numeric_ids_to_candidates() {
        let candidates = vec![
            RetrievalCandidate {
                id: "alpha".to_string(),
                source_path: "src/a.rs".to_string(),
                chunk_text: "a".to_string(),
                metadata: String::new(),
                page: 0,
                chunk_index: 0,
                vector_score: Some(0.1),
                keyword_score: None,
                fused_score: Some(0.1),
                rerank_score: None,
            },
            RetrievalCandidate {
                id: "beta".to_string(),
                source_path: "src/b.rs".to_string(),
                chunk_text: "b".to_string(),
                metadata: String::new(),
                page: 0,
                chunk_index: 1,
                vector_score: Some(0.2),
                keyword_score: None,
                fused_score: Some(0.2),
                rerank_score: None,
            },
        ];

        let resolved = resolve_rerank_positions(&candidates, &[(2, 0.8), (1, 0.4)]);
        assert_eq!(
            resolved,
            vec![("beta".to_string(), 0.8), ("alpha".to_string(), 0.4)]
        );
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
            QueryExecutionPath::Agentic
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
