use crate::agent::{EvidenceVerdict, QueryPlan, RetrievalStrategy};
use crate::citation::render_citations;
use crate::commands::query::QueryCommand;
use crate::retrieval::RetrievalCandidate;

use super::runtime::mode_label;
use super::types::QueryResult;

pub(crate) fn print_query_plan(command: &QueryCommand, result: &QueryResult) {
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
}

pub(crate) fn print_query_trace(command: &QueryCommand, result: &QueryResult) {
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
}

pub(crate) fn print_scores(command: &QueryCommand, result: &QueryResult) {
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
}

pub(crate) fn print_citations(command: &QueryCommand, result: &QueryResult) {
    if !command.show_citations {
        return;
    }

    println!("Citations:");
    for citation in render_citations(&result.hits) {
        println!("  {citation}");
    }
}

pub(crate) fn print_contexts(command: &QueryCommand, result: &QueryResult) {
    if !command.show_context {
        return;
    }

    println!("Retrieved context:");
    for (idx, hit) in result.hits.iter().enumerate() {
        let compact = retrieval_context(hit).replace('\n', " ");
        let preview: String = compact.chars().take(220).collect();
        println!("  [{}] {}", idx + 1, preview);
    }
}

fn retrieval_context(hit: &RetrievalCandidate) -> String {
    format!("Source: {}\n{}", hit.source_path, hit.chunk_text)
}

fn format_score(score: Option<f32>) -> String {
    score
        .map(|value| format!("{value:.6}"))
        .unwrap_or_else(|| "unavailable".to_string())
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
