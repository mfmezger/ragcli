use crate::cli::QueryModeArg;
use crate::commands::query::QueryCommand;
use crate::config::{ensure_store_layout, load_or_create_config, resolve_model_name, store_dir};
use crate::store::load_metadata;
use anyhow::Result;

use super::types::QueryRuntime;

pub(crate) fn prepare_runtime(name: Option<&str>, command: &QueryCommand) -> Result<QueryRuntime> {
    let store = store_dir(name)?;
    ensure_store_layout(&store)?;
    let cfg = load_or_create_config(&store)?;

    let gen_model_name = command
        .gen_model
        .clone()
        .unwrap_or_else(|| resolve_model_name(&store, &cfg.models.chat));
    let embed_model_name = resolve_model_name(&store, &cfg.models.embed);
    let metadata = load_metadata(&store)?;
    metadata.validate_query_model(&embed_model_name)?;

    Ok(QueryRuntime {
        store,
        cfg,
        gen_model_name,
        embed_model_name,
    })
}

pub(crate) fn retrieval_limit(command: &QueryCommand) -> usize {
    command.fetch_k.max(command.top_k).max(1)
}

pub(crate) fn mode_label(mode: QueryModeArg) -> &'static str {
    match mode {
        QueryModeArg::Naive => "naive",
        QueryModeArg::Hybrid => "hybrid",
        QueryModeArg::Agentic => "agentic",
        QueryModeArg::Local => "local",
        QueryModeArg::Global => "global",
        QueryModeArg::Mix => "mix",
    }
}
