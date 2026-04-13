//! RAG query planning, evidence assessment, and answer verification via LLM.
//!
//! The agent module drives the iterative retrieval loop. It classifies a user question,
//! builds a retrieval plan, gathers evidence, and verifies that generated answers are
//! actually supported by the retrieved chunks.
//!
//! # Query lifecycle
//!
//! 1. [`build_query_plan`] — classifies the question type and chooses a retrieval strategy
//! 2. Retrieval against the store (handled by the caller / `app.rs`)
//! 3. [`assess_evidence`] — LLM judges whether retrieved candidates are sufficient
//! 4. Optionally iterate with rewritten subqueries
//! 5. [`verify_answer_support`] — LLM checks that the final answer is grounded in evidence

use crate::jsonutil::parse_json;
use crate::models::Generator;
use crate::retrieval::RetrievalCandidate;
use anyhow::{Context, Result};
use serde::Deserialize;

const SUMMARY_TEXT_MAX_CHARS: usize = 600;

/// Configuration for the agentic retrieval loop.
#[allow(dead_code)]
#[derive(Clone, Debug, PartialEq)]
pub struct AgentQueryConfig {
    /// Maximum candidates to return from the store per iteration.
    pub top_k: usize,
    /// Total candidates to fetch before reranking / pruning.
    pub fetch_k: usize,
    /// Maximum retrieval iterations before giving up.
    pub max_iterations: usize,
    /// Whether to rewrite the query each iteration.
    pub enable_rewrite: bool,
    /// Whether to run the LLM reranker after retrieval.
    pub enable_rerank: bool,
}

/// Class of a user question, influencing which retrieval strategy is selected.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum QuestionType {
    /// A single factual lookup — "What is X?"
    Lookup,
    /// Comparison between two or more things — "How does X differ from Y?"
    Compare,
    /// Multi-step reasoning requiring multiple facts — "Why does X happen?"
    MultiHop,
    /// Summarization or synthesis — "Summarize the debate around X."
    Summary,
    /// Open-ended exploration — "What related topics exist?"
    Exploratory,
}

/// Retrieval strategy selected by the query planner.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RetrievalStrategy {
    /// Use the original question directly against the store.
    Direct,
    /// Rewrite the question into semantic / keyword variants before retrieving.
    Rewrite,
    /// Decompose the question into subqueries, retrieve each, then combine.
    Decompose,
    /// Retrieve broadly then rerank with an LLM.
    BroadThenRerank,
}

/// Retrieval plan produced by the LLM query planner.
#[derive(Clone, Debug, PartialEq)]
pub struct QueryPlan {
    /// Classified question type.
    pub question_type: QuestionType,
    /// Selected retrieval strategy.
    pub strategy: RetrievalStrategy,
    /// Human-readable reasoning for why this strategy was chosen.
    pub reasoning: String,
    /// Decomposed subqueries when strategy is `Decompose`.
    pub subqueries: Vec<String>,
}

/// Result of an LLM evidence assessment.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum EvidenceVerdict {
    /// Retrieved candidates fully cover what is needed to answer.
    Sufficient,
    /// Candidates are partially relevant; more iterations or reformulation may help.
    Partial,
    /// Candidates do not contain enough relevant information.
    Insufficient,
}

/// One iteration of the agentic retrieval loop.
#[derive(Clone, Debug, PartialEq)]
pub struct AgentIteration {
    /// Zero-based iteration index.
    pub iteration: usize,
    /// Query variants used in this iteration (original + rewritten).
    pub query_variants: Vec<String>,
    /// Total candidates retrieved before pruning.
    pub retrieved_count: usize,
    /// Candidates retained after deduplication and pruning.
    pub kept_count: usize,
    /// LLM judgment of evidence sufficiency.
    pub sufficiency: EvidenceVerdict,
    /// Human-readable notes from this iteration.
    pub notes: Vec<String>,
}

/// Final result of a full agentic retrieval session.
#[allow(dead_code)]
#[derive(Clone, Debug, PartialEq)]
pub struct AgentResult {
    /// Generated answer string.
    pub answer: String,
    /// Retrieval candidates used as context for generation.
    pub contexts: Vec<RetrievalCandidate>,
    /// Query plan selected by the planner.
    pub plan: QueryPlan,
    /// All iterations run before reaching a terminal verdict.
    pub iterations: Vec<AgentIteration>,
    /// Support check confirming answer is grounded in evidence.
    pub support_check: SupportCheck,
}

/// Result of verifying that a generated answer is supported by retrieved evidence.
#[derive(Clone, Debug, PartialEq)]
pub struct SupportCheck {
    /// Whether all substantive claims in the answer have supporting evidence.
    pub supported: bool,
    /// List of specific claims in the answer that lack evidence.
    pub unsupported_claims: Vec<String>,
    /// Additional notes from the verifier.
    pub notes: Vec<String>,
}

/// Evidence assessment returned by the LLM.
#[derive(Clone, Debug, PartialEq)]
pub struct EvidenceAssessment {
    /// Overall verdict on evidence sufficiency.
    pub verdict: EvidenceVerdict,
    /// Specific aspects of the question that are not covered by retrieved candidates.
    pub missing_aspects: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct QueryPlanPayload {
    question_type: String,
    strategy: String,
    reasoning: Option<String>,
    #[serde(default)]
    subqueries: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct EvidencePayload {
    verdict: String,
    #[serde(default)]
    missing_aspects: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct SupportPayload {
    supported: bool,
    #[serde(default)]
    unsupported_claims: Vec<String>,
    #[serde(default)]
    notes: Vec<String>,
}

/// Builds a structured query plan from a user question using an LLM.
///
/// The LLM is instructed to classify the question type and select an appropriate
/// retrieval strategy, returning strict JSON. Returns an error if the LLM misformats
/// its response.
pub async fn build_query_plan(generator: &Generator, question: &str) -> Result<QueryPlan> {
    let response = generator
        .generate_json(
            "You build retrieval plans for a local RAG CLI. Respond with strict JSON only and no markdown fences.",
            &format!(
                concat!(
                    "Return a JSON object with keys question_type, strategy, reasoning, and subqueries. ",
                    "question_type must be one of Lookup, Compare, MultiHop, Summary, Exploratory. ",
                    "strategy must be one of Direct, Rewrite, Decompose, BroadThenRerank. ",
                    "subqueries must be an array and may be empty. ",
                    "Question: {}"
                ),
                question,
            ),
            256,
        )
        .await
        .context("generate query plan JSON")?;

    parse_query_plan(&response)
}

/// Asks an LLM to judge whether the retrieved candidates are sufficient to answer
/// the given question.
///
/// Returns an [`EvidenceAssessment`] with a verdict and, if insufficient,
/// a list of missing aspects. This is used to decide whether to continue iterating
/// or to produce an answer from what was gathered.
pub async fn assess_evidence(
    generator: &Generator,
    question: &str,
    candidates: &[RetrievalCandidate],
) -> Result<EvidenceAssessment> {
    let response = generator
        .generate_json(
            "You assess whether retrieved evidence is sufficient to answer a local RAG query. Respond with strict JSON only and no markdown fences.",
            &format!(
                concat!(
                    "Return a JSON object with keys verdict and missing_aspects. ",
                    "verdict must be one of sufficient, partial, or insufficient. ",
                    "missing_aspects must be an array and may be empty.\n\n",
                    "Question: {}\n\nEvidence:\n{}"
                ),
                question,
                summarize_candidates(candidates),
            ),
            256,
        )
        .await
        .context("generate evidence assessment JSON")?;

    parse_evidence_assessment(&response)
}

/// Asks an LLM to verify that every substantive claim in an answer is grounded
/// in the provided evidence candidates.
///
/// Returns a [`SupportCheck`] indicating whether the answer is fully supported,
/// and listing any claims that lack backing. Used as a final check before returning
/// an answer to the user.
pub async fn verify_answer_support(
    generator: &Generator,
    question: &str,
    answer: &str,
    evidence: &[RetrievalCandidate],
) -> Result<SupportCheck> {
    let response = generator
        .generate_json(
            "You verify whether an answer is supported by retrieved evidence for a local RAG CLI. Respond with strict JSON only and no markdown fences.",
            &format!(
                concat!(
                    "Return a JSON object with keys supported, unsupported_claims, and notes. ",
                    "unsupported_claims and notes must be arrays and may be empty.\n\n",
                    "Question: {}\n\nAnswer: {}\n\nEvidence:\n{}"
                ),
                question,
                answer,
                summarize_candidates(evidence),
            ),
            256,
        )
        .await
        .context("generate support check JSON")?;

    parse_support_check(&response)
}

/// Heuristic fallback plan used when the LLM fails to produce a parseable plan.
///
/// Uses simple keyword detection to decide between `Direct` and `Decompose`:
/// presence of "and" / "interact" / "connect" triggers decomposition.
pub fn fallback_query_plan(question: &str) -> QueryPlan {
    let lower = question.to_ascii_lowercase();
    let strategy =
        if lower.contains(" and ") || lower.contains("interact") || lower.contains("connect") {
            //FIXME: not language agnostic. should be handeled by agent
            RetrievalStrategy::Decompose
        } else {
            RetrievalStrategy::Direct
        };
    let question_type = if matches!(strategy, RetrievalStrategy::Decompose) {
        QuestionType::MultiHop
    } else {
        QuestionType::Lookup
    };
    QueryPlan {
        question_type,
        strategy,
        reasoning: "Fallback heuristic plan based on the query wording.".to_string(),
        subqueries: Vec::new(),
    }
}

/// Heuristic fallback for evidence assessment when the LLM is unavailable.
///
/// Returns `Sufficient` for 3+ candidates, `Partial` for 1–2, and `Insufficient` for none.
pub fn fallback_evidence_assessment(candidates: &[RetrievalCandidate]) -> EvidenceAssessment {
    let verdict = if candidates.len() >= 3 {
        EvidenceVerdict::Sufficient
    } else if candidates.is_empty() {
        EvidenceVerdict::Insufficient
    } else {
        EvidenceVerdict::Partial
    };

    EvidenceAssessment {
        verdict,
        missing_aspects: Vec::new(),
    }
}

/// Heuristic fallback support check when the LLM is unavailable.
///
/// Returns `supported = true` if the answer is non-empty and there is at least one
/// evidence candidate; otherwise `supported = false`.
pub fn fallback_support_check(answer: &str, evidence: &[RetrievalCandidate]) -> SupportCheck {
    let supported = !answer.trim().is_empty() && !evidence.is_empty();
    SupportCheck {
        supported,
        unsupported_claims: if supported {
            Vec::new()
        } else {
            vec!["Unable to confirm answer support from retrieved evidence.".to_string()]
        },
        notes: vec!["Fallback support heuristic used.".to_string()],
    }
}

/// Parses a JSON query plan from the LLM. Used by [`build_query_plan`].
pub fn parse_query_plan(raw: &str) -> Result<QueryPlan> {
    let payload: QueryPlanPayload = parse_json(raw, "parse query plan JSON")?;
    Ok(QueryPlan {
        question_type: parse_question_type(&payload.question_type)?,
        strategy: parse_retrieval_strategy(&payload.strategy)?,
        reasoning: payload.reasoning.unwrap_or_default().trim().to_string(),
        subqueries: payload
            .subqueries
            .into_iter()
            .map(|item| item.trim().to_string())
            .filter(|item| !item.is_empty())
            .collect(),
    })
}

/// Parses an evidence assessment JSON from the LLM. Used by [`assess_evidence`].
pub fn parse_evidence_assessment(raw: &str) -> Result<EvidenceAssessment> {
    let payload: EvidencePayload = parse_json(raw, "parse evidence assessment JSON")?;
    Ok(EvidenceAssessment {
        verdict: parse_evidence_verdict(&payload.verdict)?,
        missing_aspects: payload
            .missing_aspects
            .into_iter()
            .map(|item| item.trim().to_string())
            .filter(|item| !item.is_empty())
            .collect(),
    })
}

/// Parses a support check JSON from the LLM. Used by [`verify_answer_support`].
pub fn parse_support_check(raw: &str) -> Result<SupportCheck> {
    let payload: SupportPayload = parse_json(raw, "parse support check JSON")?;
    Ok(SupportCheck {
        supported: payload.supported,
        unsupported_claims: payload
            .unsupported_claims
            .into_iter()
            .map(|item| item.trim().to_string())
            .filter(|item| !item.is_empty())
            .collect(),
        notes: payload
            .notes
            .into_iter()
            .map(|item| item.trim().to_string())
            .filter(|item| !item.is_empty())
            .collect(),
    })
}

/// Builds a compact multi-line summary of retrieval candidates for inclusion in LLM prompts.
///
/// Each line is `source=<path> page=<N> text=<truncated>` so the LLM can reference
/// specific chunks without receiving the full text of every candidate.
fn summarize_candidates(candidates: &[RetrievalCandidate]) -> String {
    candidates
        .iter()
        .map(|candidate| {
            format!(
                "source={} page={} text={}",
                candidate.source_path,
                candidate.page,
                summarize_candidate_text(&candidate.chunk_text)
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Truncates chunk text to [`SUMMARY_TEXT_MAX_CHARS`] characters for prompt size control.
fn summarize_candidate_text(text: &str) -> String {
    let normalized = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut truncated = normalized
        .chars()
        .take(SUMMARY_TEXT_MAX_CHARS)
        .collect::<String>();
    if normalized.chars().count() > SUMMARY_TEXT_MAX_CHARS {
        truncated.push_str("...");
    }
    truncated
}

/// Parses a question type string (case-insensitive) into a [`QuestionType`].
fn parse_question_type(value: &str) -> Result<QuestionType> {
    match value.trim().to_ascii_lowercase().as_str() {
        "lookup" => Ok(QuestionType::Lookup),
        "compare" => Ok(QuestionType::Compare),
        "multihop" => Ok(QuestionType::MultiHop),
        "summary" => Ok(QuestionType::Summary),
        "exploratory" => Ok(QuestionType::Exploratory),
        other => anyhow::bail!("unknown question type: {other}"),
    }
}

/// Parses a retrieval strategy string (case-insensitive) into a [`RetrievalStrategy`].
fn parse_retrieval_strategy(value: &str) -> Result<RetrievalStrategy> {
    match value.trim().to_ascii_lowercase().as_str() {
        "direct" => Ok(RetrievalStrategy::Direct),
        "rewrite" => Ok(RetrievalStrategy::Rewrite),
        "decompose" => Ok(RetrievalStrategy::Decompose),
        "broadthenrerank" => Ok(RetrievalStrategy::BroadThenRerank),
        other => anyhow::bail!("unknown retrieval strategy: {other}"),
    }
}

/// Parses an evidence verdict string (case-insensitive) into an [`EvidenceVerdict`].
fn parse_evidence_verdict(value: &str) -> Result<EvidenceVerdict> {
    match value.trim().to_ascii_lowercase().as_str() {
        "sufficient" => Ok(EvidenceVerdict::Sufficient),
        "partial" => Ok(EvidenceVerdict::Partial),
        "insufficient" => Ok(EvidenceVerdict::Insufficient),
        other => anyhow::bail!("unknown evidence verdict: {other}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(source: &str, chunk_text: &str) -> RetrievalCandidate {
        RetrievalCandidate {
            id: String::new(),
            source_path: source.to_string(),
            chunk_text: chunk_text.to_string(),
            metadata: String::new(),
            page: 0,
            chunk_index: 0,
            vector_score: None,
            keyword_score: None,
            fused_score: None,
            rerank_score: None,
        }
    }

    #[test]
    fn test_parse_query_plan_accepts_json_fences() {
        let plan = parse_query_plan(
            "```json\n{\"question_type\":\"MultiHop\",\"strategy\":\"Decompose\",\"reasoning\":\"needs two facts\",\"subqueries\":[\"config resolution\",\"metadata validation\"]}\n```",
        )
        .unwrap();

        assert_eq!(plan.question_type, QuestionType::MultiHop);
        assert_eq!(plan.strategy, RetrievalStrategy::Decompose);
        assert_eq!(plan.subqueries.len(), 2);
    }

    #[test]
    fn test_fallback_evidence_assessment_marks_partial_for_small_result_sets() {
        let assessment = fallback_evidence_assessment(&[candidate("src/config.rs", "config")]);
        assert_eq!(assessment.verdict, EvidenceVerdict::Partial);
    }

    #[test]
    fn test_parse_support_check_errors_on_invalid_json() {
        let err = parse_support_check("not json").unwrap_err().to_string();
        assert!(err.contains("parse support check JSON"));
    }

    #[test]
    fn test_parse_query_plan_accepts_case_insensitive_values() {
        let plan = parse_query_plan(
            "{\"question_type\":\"lookup\",\"strategy\":\"decompose\",\"reasoning\":\"ok\",\"subqueries\":[]}",
        )
        .unwrap();
        assert_eq!(plan.question_type, QuestionType::Lookup);
        assert_eq!(plan.strategy, RetrievalStrategy::Decompose);
    }

    #[test]
    fn test_summarize_candidates_includes_all_candidates() {
        let summary = summarize_candidates(&[
            candidate("a", "one"),
            candidate("b", "two"),
            candidate("c", "three"),
            candidate("d", "four"),
            candidate("e", "five"),
            candidate("f", "six"),
            candidate("g", "seven"),
        ]);

        assert!(summary.contains("source=g page=0 text=seven"));
    }

    #[test]
    fn test_summarize_candidates_normalizes_newlines_and_spaces() {
        let summary = summarize_candidates(&[candidate("src/a.rs", "one\n\n two\tthree")]);
        assert!(summary.contains("source=src/a.rs page=0 text=one two three"));
    }

    #[test]
    fn test_summarize_candidate_text_truncates_long_chunks() {
        let text = format!("{} tail", "a".repeat(SUMMARY_TEXT_MAX_CHARS + 20));
        let summarized = summarize_candidate_text(&text);
        assert_eq!(summarized.len(), SUMMARY_TEXT_MAX_CHARS + 3);
        assert!(summarized.ends_with("..."));
    }

    #[test]
    fn test_parse_evidence_assessment_includes_raw_snippet_in_error() {
        let err = parse_evidence_assessment("not json at all")
            .unwrap_err()
            .to_string();
        assert!(err.contains("parse evidence assessment JSON"));
        assert!(err.contains("not json"));
    }
}
