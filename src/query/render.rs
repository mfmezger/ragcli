use crate::agent::{EvidenceVerdict, QueryPlan, RetrievalStrategy};
use crate::citation::render_citations;
use crate::commands::query::QueryCommand;
use crate::retrieval::RetrievalCandidate;
use crate::ui::{self, Panel};

const CONTEXT_PREVIEW_CHARS: usize = 700;
const CONTEXT_WRAP_WIDTH: usize = 96;
const CONTEXT_LABEL_WIDTH: usize = 9;

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
        if idx > 0 {
            panel.push("");
        }
        panel.prose(
            &format!("[{}]", idx + 1),
            &retrieval_context_heading(hit),
            CONTEXT_LABEL_WIDTH,
        );
        push_context_preview(&mut panel, &hit.chunk_text);
    }
    panel.render();
}

fn retrieval_context_heading(hit: &RetrievalCandidate) -> String {
    let mut parts = vec![format!("source={}", hit.source_path)];
    if hit.page > 0 {
        parts.push(format!("page={}", hit.page));
    }
    parts.push(format!("chunk={}", hit.chunk_index));
    if let Some(score) = hit.best_score() {
        parts.push(format!("score={score:.6}"));
    }
    parts.join(" · ")
}

fn push_context_preview(panel: &mut Panel, text: &str) {
    let (preview, truncated) = context_preview(text);
    let mut first_line = true;

    for line in preview.lines() {
        let label = if first_line { "snippet" } else { "" };
        for row in ui::wrapped_dim_rows(
            label,
            line.trim_end(),
            CONTEXT_LABEL_WIDTH,
            CONTEXT_WRAP_WIDTH,
        ) {
            panel.push(row);
        }
        first_line = false;
    }

    if first_line {
        for row in ui::wrapped_dim_rows("snippet", "", CONTEXT_LABEL_WIDTH, CONTEXT_WRAP_WIDTH) {
            panel.push(row);
        }
    }

    if truncated {
        panel.prose(
            "",
            "… truncated; increase top-k or inspect the source for full text",
            CONTEXT_LABEL_WIDTH,
        );
    }
}

fn context_preview(text: &str) -> (String, bool) {
    let trimmed = text.trim();
    let mut preview: String = trimmed.chars().take(CONTEXT_PREVIEW_CHARS).collect();
    let truncated = trimmed.chars().count() > CONTEXT_PREVIEW_CHARS;
    if truncated {
        preview.push('…');
    }
    (preview, truncated)
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

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(source_path: &str, page: i32, chunk_index: i32) -> RetrievalCandidate {
        RetrievalCandidate {
            id: String::new(),
            source_path: source_path.to_string(),
            chunk_text: "alpha\nbeta".to_string(),
            metadata: String::new(),
            page,
            chunk_index,
            vector_score: Some(0.42),
            keyword_score: None,
            fused_score: Some(0.1234567),
            rerank_score: None,
        }
    }

    #[test]
    fn test_retrieval_context_heading_includes_location_and_score() {
        let heading = retrieval_context_heading(&candidate("docs/guide.md", 3, 7));

        assert_eq!(
            heading,
            "source=docs/guide.md · page=3 · chunk=7 · score=0.123457"
        );
    }

    #[test]
    fn test_retrieval_context_heading_omits_page_when_absent() {
        let heading = retrieval_context_heading(&candidate("src/main.rs", 0, 2));

        assert_eq!(heading, "source=src/main.rs · chunk=2 · score=0.123457");
    }

    #[test]
    fn test_context_preview_trims_and_marks_truncation() {
        let input = format!("  {}  ", "x".repeat(CONTEXT_PREVIEW_CHARS + 1));
        let (preview, truncated) = context_preview(&input);

        assert!(truncated);
        assert_eq!(preview.chars().count(), CONTEXT_PREVIEW_CHARS + 1);
        assert!(preview.ends_with('…'));
    }

    #[test]
    fn test_context_preview_preserves_short_multiline_text() {
        let (preview, truncated) = context_preview(" alpha\nbeta ");

        assert!(!truncated);
        assert_eq!(preview, "alpha\nbeta");
    }
}
