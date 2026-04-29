use crate::agent::{EvidenceVerdict, QueryPlan, RetrievalStrategy};
use crate::citation::render_citations;
use crate::commands::query::QueryCommand;
use crate::retrieval::RetrievalCandidate;
use crate::ui::{self, Panel};

use super::runtime::mode_label;
use super::types::QueryResult;

pub(crate) fn print_query_plan(command: &QueryCommand, result: &QueryResult) {
    if !command.show_plan {
        return;
    }

    let mut panel = Panel::new("Query Plan:");
    panel.kv("requested", mode_label(result.requested_mode), 14);
    panel.kv("execution", result.execution_label, 14);
    panel.kv("top k", command.top_k.to_string(), 14);
    panel.kv("fetch k", command.fetch_k.to_string(), 14);
    panel.kv("iterations", command.max_iterations.to_string(), 14);
    panel.kv("rewrite", enabled(command.rewrite), 14);
    panel.kv("rerank", enabled(command.rerank), 14);
    panel.prose(
        "variants",
        &result.rewrite_set.query_variants().join(" | "),
        14,
    );
    if let Some(plan) = &result.plan {
        panel.kv("question", question_type_label(plan), 14);
        panel.kv("strategy", strategy_label(plan), 14);
        panel.prose("reasoning", &plan.reasoning, 14);
        if !plan.subqueries.is_empty() {
            panel.prose("subqueries", &plan.subqueries.join(" | "), 14);
        }
    }
    panel.render();
}

pub(crate) fn print_query_trace(command: &QueryCommand, result: &QueryResult) {
    if !command.show_trace {
        return;
    }

    let mut panel = Panel::new("Query Trace:");
    for entry in &result.trace {
        panel.prose("step", entry, 9);
    }
    for iteration in &result.iterations {
        panel.prose(
            "iteration",
            &format!(
                "{} summary: verdict={}, queries={}, kept={}",
                iteration.iteration,
                evidence_verdict_label(&iteration.sufficiency),
                iteration.query_variants.join(" | "),
                iteration.kept_count
            ),
            9,
        );
    }
    if let Some(check) = &result.support_check {
        panel.kv(
            "support",
            if check.supported {
                ui::ok("supported")
            } else {
                ui::err("unsupported")
            },
            9,
        );
    }
    panel.kv("hits", result.hits.len().to_string(), 9);
    panel.render();
}

pub(crate) fn print_scores(command: &QueryCommand, result: &QueryResult) {
    if !command.show_scores {
        return;
    }

    ui::render_table(
        "Scores:",
        &["#", "Best", "Fused", "Rerank", "Source"],
        result
            .hits
            .iter()
            .enumerate()
            .map(|(idx, hit)| {
                vec![
                    (idx + 1).to_string(),
                    format!("{:.6}", hit.best_score().unwrap_or_default()),
                    format_score(hit.fused_score),
                    format_score(hit.rerank_score),
                    hit.source_path.clone(),
                ]
            })
            .collect(),
    );
}

pub(crate) fn print_citations(command: &QueryCommand, result: &QueryResult) {
    if !command.show_citations {
        return;
    }

    let mut panel = Panel::new("Citations:");
    for citation in render_citations(&result.hits) {
        panel.prose("", &citation, 0);
    }
    panel.render();
}

pub(crate) fn print_contexts(command: &QueryCommand, result: &QueryResult) {
    if !command.show_context {
        return;
    }

    let mut panel = Panel::new("Retrieved context:");
    for (idx, hit) in result.hits.iter().enumerate() {
        let compact = retrieval_context(hit).replace('\n', " ");
        let preview: String = compact.chars().take(220).collect();
        panel.prose(&format!("[{}]", idx + 1), &preview, 5);
    }
    panel.render();
}

fn retrieval_context(hit: &RetrievalCandidate) -> String {
    format!("Source: {}\n{}", hit.source_path, hit.chunk_text)
}

fn enabled(value: bool) -> String {
    if value {
        ui::ok("enabled")
    } else {
        "disabled".to_string()
    }
}

fn format_score(score: Option<f32>) -> String {
    score
        .map(|value| format!("{value:.6}"))
        .unwrap_or_else(|| "unavailable".to_string())
}

pub(crate) fn evidence_verdict_label(verdict: &EvidenceVerdict) -> &'static str {
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
