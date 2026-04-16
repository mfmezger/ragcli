use crate::agent::{AgentIteration, QueryPlan, SupportCheck};
use crate::cli::QueryModeArg;
use crate::config::Config;
use crate::retrieval::RetrievalCandidate;
use crate::rewrite::QueryRewriteSet;
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
