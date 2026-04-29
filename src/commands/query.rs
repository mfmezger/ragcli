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

#[allow(clippy::too_many_arguments)]
impl QueryCommand {
    pub fn new(
        question: String,
        mode: QueryModeArg,
        top_k: usize,
        fetch_k: usize,
        max_iterations: usize,
        rewrite: bool,
        rerank: bool,
        show_context: bool,
        show_plan: bool,
        show_scores: bool,
        show_citations: bool,
        show_trace: bool,
        source: Option<String>,
        path_prefix: Option<String>,
        page: Option<i32>,
        format: Option<String>,
        gen_model: Option<String>,
        max_tokens: usize,
    ) -> Self {
        Self {
            question,
            mode,
            top_k,
            fetch_k,
            max_iterations,
            rewrite,
            rerank,
            show_context,
            show_plan,
            show_scores,
            show_citations,
            show_trace,
            source,
            path_prefix,
            page,
            format,
            gen_model,
            max_tokens,
        }
    }
}

pub async fn run(name: Option<&str>, command: QueryCommand) -> Result<()> {
    crate::query::run(name, command).await
}
