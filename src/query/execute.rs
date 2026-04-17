use crate::agent::{
    assess_evidence, build_query_plan, fallback_evidence_assessment, fallback_query_plan,
    fallback_support_check, verify_answer_support, AgentIteration, EvidenceAssessment,
    EvidenceVerdict, QueryPlan, RetrievalStrategy,
};
use crate::citation::labeled_contexts;
use crate::cli::QueryModeArg;
use crate::commands::query::QueryCommand;
use crate::graph::placeholder_plan;
use crate::models::Generator;
use crate::retrieval::{merge_candidates, prune_candidates, RetrievalCandidate};
use crate::rewrite::{rewrite_query_for_retrieval, QueryRewriteSet};
use crate::store;
use anyhow::Result;
use std::collections::BTreeSet;

use super::render::{
    evidence_verdict_label, print_citations, print_contexts, print_query_plan,
    print_query_trace, print_scores,
};
use super::retrieve::retrieve_candidates;
use super::runtime::{mode_label, prepare_runtime};
use super::types::{QueryExecutionPath, QueryResult, QueryRuntime};

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

async fn run_simple_query(runtime: &QueryRuntime, command: &QueryCommand) -> Result<QueryResult> {
    let mut trace = vec![format!(
        "mode {} uses the current hybrid retrieval pipeline",
        mode_label(command.mode)
    )];
    let rewrite_set = build_rewrite_set(runtime, command, &mut trace).await;
    let hits =
        retrieve_candidates(runtime, command, &rewrite_set.query_variants(), &mut trace).await?;

    Ok(QueryResult {
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
) -> Result<QueryResult> {
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
    let mut accumulated_hits = Vec::new();

    for iteration in 1..=command.max_iterations.max(1) {
        trace.push(format!(
            "iteration {iteration}: querying {} variant(s)",
            active_queries.len()
        ));
        let iteration_hits =
            retrieve_candidates(runtime, command, &active_queries, &mut trace).await?;
        let retrieved_count = iteration_hits.len();
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
            retrieved_count,
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
                fallback_support_check(&answer, &final_hits)
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

    Ok(QueryResult {
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
) -> Result<QueryResult> {
    let mut trace = vec![format!(
        "requested {} mode routed through the graph-mode placeholder path",
        mode_label(command.mode)
    )];
    let rewrite_set = build_rewrite_set(runtime, command, &mut trace).await;
    let stub_plan = placeholder_plan(command.mode, &rewrite_set);
    trace.extend(stub_plan.notes.iter().cloned());
    let hits = retrieve_candidates(runtime, command, &stub_plan.query_variants, &mut trace).await?;

    Ok(QueryResult {
        requested_mode: command.mode,
        execution_label: stub_plan.execution_label,
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

async fn generate_answer(
    runtime: &QueryRuntime,
    command: &QueryCommand,
    result: &QueryResult,
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

fn execution_path(mode: QueryModeArg) -> QueryExecutionPath {
    match mode {
        QueryModeArg::Naive | QueryModeArg::Hybrid => QueryExecutionPath::Simple,
        QueryModeArg::Agentic => QueryExecutionPath::Agentic,
        QueryModeArg::Local | QueryModeArg::Global | QueryModeArg::Mix => {
            QueryExecutionPath::AgenticStub
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commands::index;
    use crate::test_support::{sequential_json_server, with_test_env};

    use super::super::runtime::retrieval_limit;

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

    #[tokio::test(flavor = "current_thread")]
    async fn test_local_global_and_mix_modes_use_distinct_placeholder_paths() {
        let dir = tempfile::tempdir().unwrap();
        let docs = dir.path().join("docs");
        std::fs::create_dir_all(&docs).unwrap();
        let input = docs.join("note.txt");
        std::fs::write(&input, "Config checks use metadata validation.").unwrap();

        let index_server = sequential_json_server(vec![r#"{"embeddings":[[0.1,0.2]]}"#]);
        with_test_env(dir.path(), Some(&index_server), || async {
            index::run(
                Some("graph-modes"),
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

        let query_server = sequential_json_server(vec![
            r#"{"embeddings":[[0.1,0.2]]}"#,
            r#"{"message":{"content":"local answer [1]"}}"#,
            r#"{"embeddings":[[0.1,0.2]]}"#,
            r#"{"message":{"content":"global answer [1]"}}"#,
            r#"{"embeddings":[[0.1,0.2]]}"#,
            r#"{"message":{"content":"mix answer [1]"}}"#,
        ]);
        with_test_env(dir.path(), Some(&query_server), || async {
            for mode in [QueryModeArg::Local, QueryModeArg::Global, QueryModeArg::Mix] {
                run(
                    Some("graph-modes"),
                    QueryCommand {
                        question: "How do config checks work?".to_string(),
                        mode,
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
                        max_tokens: 64,
                    },
                )
                .await
                .unwrap();
            }
        })
        .await;
    }
}
