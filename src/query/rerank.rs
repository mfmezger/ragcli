use crate::commands::query::QueryCommand;
use crate::models::Generator;
use crate::retrieval::{apply_rerank_order, RetrievalCandidate};
use crate::rewrite::trim_json_fences;
use anyhow::{Context, Result};
use tracing::{field, Instrument};

use super::types::QueryRuntime;

const RERANK_MAX_TEXT_CHARS: usize = 600;
const RERANK_RESPONSE_TOKENS: usize = 384;

#[derive(Debug, serde::Deserialize)]
struct RerankPayload {
    ranked_ids: Vec<RerankItem>,
}

#[derive(Debug, serde::Deserialize)]
struct RerankItem {
    id: String,
    score: f32,
}

pub(crate) async fn rerank_candidates(
    runtime: &QueryRuntime,
    command: &QueryCommand,
    candidates: Vec<RetrievalCandidate>,
    trace: &mut Vec<String>,
) -> Vec<RetrievalCandidate> {
    let span = tracing::info_span!(
        "rerank_candidates",
        enabled = command.rerank,
        candidate_count = candidates.len(),
        fetch_k = command.fetch_k,
        returned_candidates = field::Empty,
    );

    let span_inner = span.clone();
    async move {
        if !command.rerank || candidates.len() <= 1 {
            trace.push("rerank disabled; using fused retrieval order".to_string());
            span_inner.record("returned_candidates", candidates.len());
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
                span_inner.record("returned_candidates", reranked.len());
                trace.push(format!(
                    "rerank enabled; reordered {} candidate(s)",
                    reranked.len()
                ));
                reranked
            }
            Err(err) => {
                span_inner.record("returned_candidates", candidates.len());
                trace.push(format!(
                    "rerank failed; falling back to fused retrieval order ({})",
                    err.root_cause()
                ));
                candidates
            }
        }
    }
    .instrument(span)
    .await
}

async fn rerank_with_model(
    generator: &Generator,
    question: &str,
    candidates: &[RetrievalCandidate],
    top_n: usize,
) -> Result<Vec<RetrievalCandidate>> {
    let span = tracing::info_span!(
        "rerank_with_model",
        question_chars = question.chars().count(),
        candidate_count = candidates.len(),
        top_n,
    );

    async move {
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
    .instrument(span)
    .await
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

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_truncate_for_rerank_limits_candidate_text() {
        let text = "a".repeat(RERANK_MAX_TEXT_CHARS + 25);
        let truncated = truncate_for_rerank(&text);
        assert_eq!(truncated.len(), RERANK_MAX_TEXT_CHARS + 3);
        assert!(truncated.ends_with("..."));
    }
}
