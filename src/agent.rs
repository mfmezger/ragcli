use crate::models::Generator;
use crate::retrieval::RetrievalCandidate;
use crate::rewrite::trim_json_fences;
use anyhow::{Context, Result};
use serde::Deserialize;

const SUMMARY_TEXT_MAX_CHARS: usize = 600;

#[allow(dead_code)]
#[derive(Clone, Debug, PartialEq)]
pub struct AgentQueryConfig {
    pub top_k: usize,
    pub fetch_k: usize,
    pub max_iterations: usize,
    pub enable_rewrite: bool,
    pub enable_rerank: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum QuestionType {
    Lookup,
    Compare,
    MultiHop,
    Summary,
    Exploratory,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RetrievalStrategy {
    Direct,
    Rewrite,
    Decompose,
    BroadThenRerank,
}

#[derive(Clone, Debug, PartialEq)]
pub struct QueryPlan {
    pub question_type: QuestionType,
    pub strategy: RetrievalStrategy,
    pub reasoning: String,
    pub subqueries: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum EvidenceVerdict {
    Sufficient,
    Partial,
    Insufficient,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AgentIteration {
    pub iteration: usize,
    pub query_variants: Vec<String>,
    pub retrieved_count: usize,
    pub kept_count: usize,
    pub sufficiency: EvidenceVerdict,
    pub notes: Vec<String>,
}

#[allow(dead_code)]
#[derive(Clone, Debug, PartialEq)]
pub struct AgentResult {
    pub answer: String,
    pub contexts: Vec<RetrievalCandidate>,
    pub plan: QueryPlan,
    pub iterations: Vec<AgentIteration>,
    pub support_check: SupportCheck,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SupportCheck {
    pub supported: bool,
    pub unsupported_claims: Vec<String>,
    pub notes: Vec<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EvidenceAssessment {
    pub verdict: EvidenceVerdict,
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

pub fn fallback_query_plan(question: &str) -> QueryPlan {
    let lower = question.to_ascii_lowercase();
    let strategy =
        if lower.contains(" and ") || lower.contains("interact") || lower.contains("connect") {
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

pub fn parse_query_plan(raw: &str) -> Result<QueryPlan> {
    let payload: QueryPlanPayload =
        serde_json::from_str(trim_json_fences(raw)).context("parse query plan JSON")?;
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

pub fn parse_evidence_assessment(raw: &str) -> Result<EvidenceAssessment> {
    let payload: EvidencePayload =
        serde_json::from_str(trim_json_fences(raw)).context("parse evidence assessment JSON")?;
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

pub fn parse_support_check(raw: &str) -> Result<SupportCheck> {
    let payload: SupportPayload =
        serde_json::from_str(trim_json_fences(raw)).context("parse support check JSON")?;
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

fn parse_retrieval_strategy(value: &str) -> Result<RetrievalStrategy> {
    match value.trim().to_ascii_lowercase().as_str() {
        "direct" => Ok(RetrievalStrategy::Direct),
        "rewrite" => Ok(RetrievalStrategy::Rewrite),
        "decompose" => Ok(RetrievalStrategy::Decompose),
        "broadthenrerank" => Ok(RetrievalStrategy::BroadThenRerank),
        other => anyhow::bail!("unknown retrieval strategy: {other}"),
    }
}

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
}
