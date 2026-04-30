use crate::cli::QueryModeArg;
use anyhow::Result;

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

pub async fn run(name: Option<&str>, command: QueryCommand) -> Result<()> {
    crate::query::run(name, command).await
}
