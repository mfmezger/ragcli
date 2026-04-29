use crate::agent::{AgentIteration, QueryPlan, SupportCheck};
use crate::cli::QueryModeArg;
use crate::config::Config;
use crate::retrieval::RetrievalCandidate;
use crate::rewrite::QueryRewriteSet;
use serde::Serialize;
use std::path::PathBuf;

pub(crate) struct QueryRuntime {
    pub(crate) store: PathBuf,
    pub(crate) cfg: Config,
    pub(crate) gen_model_name: String,
    pub(crate) embed_model_name: String,
}

#[derive(Debug)]
pub(crate) struct QueryResult {
    pub(crate) requested_mode: QueryModeArg,
    pub(crate) execution_label: &'static str,
    pub(crate) rewrite_set: QueryRewriteSet,
    pub(crate) plan: Option<QueryPlan>,
    pub(crate) iterations: Vec<AgentIteration>,
    pub(crate) support_check: Option<SupportCheck>,
    pub(crate) answer: Option<String>,
    pub(crate) hits: Vec<RetrievalCandidate>,
    pub(crate) trace: Vec<String>,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) enum QueryExecutionPath {
    Simple,
    Agentic,
    AgenticStub,
}

#[derive(Debug, Serialize)]
pub(crate) struct QueryJsonReport {
    pub question: String,
    pub answer: Option<String>,
    pub mode: String,
    pub hits: Vec<HitJson>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub plan: Option<PlanJson>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub iterations: Vec<IterationJson>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub support_check: Option<SupportCheckJson>,
}

#[derive(Debug, Serialize)]
pub(crate) struct HitJson {
    pub source: String,
    pub page: i32,
    pub chunk_index: i32,
    pub score: Option<f32>,
    pub text: String,
}

#[derive(Debug, Serialize)]
pub(crate) struct PlanJson {
    pub question_type: String,
    pub strategy: String,
    pub reasoning: String,
    pub subqueries: Vec<String>,
}

#[derive(Debug, Serialize)]
pub(crate) struct IterationJson {
    pub iteration: usize,
    pub verdict: String,
    pub retrieved: usize,
    pub kept: usize,
}

#[derive(Debug, Serialize)]
pub(crate) struct SupportCheckJson {
    pub supported: bool,
    pub unsupported_claims: Vec<String>,
}

impl QueryJsonReport {
    pub(crate) fn from_result(
        question: &str,
        mode: &str,
        result: &QueryResult,
    ) -> Self {
        Self {
            question: question.to_string(),
            answer: result.answer.clone(),
            mode: mode.to_string(),
            hits: result
                .hits
                .iter()
                .map(|hit| HitJson {
                    source: hit.source_path.clone(),
                    page: hit.page,
                    chunk_index: hit.chunk_index,
                    score: hit.best_score(),
                    text: hit.chunk_text.clone(),
                })
                .collect(),
            plan: result.plan.as_ref().map(|plan| PlanJson {
                question_type: question_type_str(&plan.question_type),
                strategy: strategy_str(&plan.strategy),
                reasoning: plan.reasoning.clone(),
                subqueries: plan.subqueries.clone(),
            }),
            iterations: result
                .iterations
                .iter()
                .map(|iter| IterationJson {
                    iteration: iter.iteration,
                    verdict: evidence_verdict_str(&iter.sufficiency),
                    retrieved: iter.retrieved_count,
                    kept: iter.kept_count,
                })
                .collect(),
            support_check: result.support_check.as_ref().map(|check| SupportCheckJson {
                supported: check.supported,
                unsupported_claims: check.unsupported_claims.clone(),
            }),
        }
    }
}

fn question_type_str(qt: &crate::agent::QuestionType) -> String {
    match qt {
        crate::agent::QuestionType::Lookup => "Lookup".to_string(),
        crate::agent::QuestionType::Compare => "Compare".to_string(),
        crate::agent::QuestionType::MultiHop => "MultiHop".to_string(),
        crate::agent::QuestionType::Summary => "Summary".to_string(),
        crate::agent::QuestionType::Exploratory => "Exploratory".to_string(),
    }
}

fn strategy_str(s: &crate::agent::RetrievalStrategy) -> String {
    match s {
        crate::agent::RetrievalStrategy::Direct => "Direct".to_string(),
        crate::agent::RetrievalStrategy::Rewrite => "Rewrite".to_string(),
        crate::agent::RetrievalStrategy::Decompose => "Decompose".to_string(),
        crate::agent::RetrievalStrategy::BroadThenRerank => "BroadThenRerank".to_string(),
    }
}

fn evidence_verdict_str(v: &crate::agent::EvidenceVerdict) -> String {
    match v {
        crate::agent::EvidenceVerdict::Sufficient => "sufficient".to_string(),
        crate::agent::EvidenceVerdict::Partial => "partial".to_string(),
        crate::agent::EvidenceVerdict::Insufficient => "insufficient".to_string(),
    }
}
