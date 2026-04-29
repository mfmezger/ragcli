use crate::commands::query::QueryCommand;
use crate::models::Embedder;
use crate::retrieval::{merge_candidates, prune_candidates, RetrievalCandidate};
use crate::store::{self, build_retrieval_filter, connect_db, ensure_fts_index};
use anyhow::{Context, Result};
use arrow_array::{Float32Array, Float64Array, Int32Array, RecordBatch, StringArray};
use futures::TryStreamExt;
use lancedb::index::scalar::FullTextSearchQuery;
use lancedb::query::{QueryBase, QueryExecutionOptions};
use tracing::{field, Instrument};

use super::rerank::rerank_candidates;
use super::runtime::retrieval_limit;
use super::types::QueryRuntime;

pub(crate) async fn retrieve_candidates(
    runtime: &QueryRuntime,
    command: &QueryCommand,
    queries: &[String],
    trace: &mut Vec<String>,
) -> Result<Vec<RetrievalCandidate>> {
    let span = tracing::info_span!(
        "retrieve_candidates",
        query_variants = queries.len(),
        top_k = command.top_k,
        fetch_k = command.fetch_k,
        has_source_filter = command.source.is_some(),
        has_path_prefix_filter = command.path_prefix.is_some(),
        page = field::debug(command.page),
        has_format_filter = command.format.is_some(),
        merged_candidates = field::Empty,
        pruned_candidates = field::Empty,
    );

    let span_inner = span.clone();
    async move {
        let mut groups = Vec::new();
        for query in queries {
            groups.push(retrieve_candidates_for_query(runtime, query, command).await?);
        }

        let merged = merge_candidates(groups);
        span_inner.record("merged_candidates", merged.len());
        trace.push(format!(
            "merged {} candidate(s) across {} retrieval query variant(s)",
            merged.len(),
            queries.len()
        ));

        let reranked = rerank_candidates(runtime, command, merged, trace).await;
        let pruned = prune_candidates(reranked, command.top_k);
        span_inner.record("pruned_candidates", pruned.len());
        trace.push(format!("kept {} candidate(s) after pruning", pruned.len()));
        Ok(pruned)
    }
    .instrument(span)
    .await
}

async fn retrieve_candidates_for_query(
    runtime: &QueryRuntime,
    question: &str,
    command: &QueryCommand,
) -> Result<Vec<RetrievalCandidate>> {
    let span = tracing::info_span!(
        "retrieve_query_variant",
        query_chars = question.chars().count(),
        retrieval_limit = retrieval_limit(command),
        has_source_filter = command.source.is_some(),
        has_path_prefix_filter = command.path_prefix.is_some(),
        page = field::debug(command.page),
        has_format_filter = command.format.is_some(),
        batch_count = field::Empty,
        hit_count = field::Empty,
    );

    let span_inner = span.clone();
    async move {
        let db = connect_db(&runtime.store).await?;
        let table = db
            .open_table(store::DEFAULT_TABLE_NAME)
            .execute()
            .await
            .context("open table")?;
        ensure_fts_index(&table, false).await?;

        let embedder = Embedder::new(
            runtime.cfg.ollama.base_url.clone(),
            runtime.embed_model_name.clone(),
        );
        let embedding = embedder.embed(question).await?;

        let mut query = table
            .query()
            .full_text_search(FullTextSearchQuery::new(question.to_string()))
            .nearest_to(embedding.as_slice())?
            .limit(retrieval_limit(command));

        if let Some(filter) = build_retrieval_filter(
            command.source.as_deref(),
            command.path_prefix.as_deref(),
            command.page,
            command.format.as_deref(),
        ) {
            query = query.only_if(filter);
        }

        let batches: Vec<RecordBatch> = query
            .execute_hybrid(QueryExecutionOptions::default())
            .await?
            .try_collect::<Vec<_>>()
            .await?;
        span_inner.record("batch_count", batches.len());
        let hits = extract_candidates(&batches)?;
        span_inner.record("hit_count", hits.len());
        Ok(hits)
    }
    .instrument(span)
    .await
}

fn extract_candidates(batches: &[RecordBatch]) -> Result<Vec<RetrievalCandidate>> {
    let mut hits = Vec::new();

    for batch in batches {
        let id_col = batch
            .column_by_name("id")
            .and_then(|column| column.as_any().downcast_ref::<StringArray>());
        let text_col = batch
            .column_by_name("chunk_text")
            .context("chunk_text column missing")?
            .as_any()
            .downcast_ref::<StringArray>()
            .context("chunk_text column type")?;
        let source_col = batch
            .column_by_name("source_path")
            .context("source_path column missing")?
            .as_any()
            .downcast_ref::<StringArray>()
            .context("source_path column type")?;
        let metadata_col = batch
            .column_by_name("metadata")
            .and_then(|column| column.as_any().downcast_ref::<StringArray>());
        let page_col = batch
            .column_by_name("page")
            .and_then(|column| column.as_any().downcast_ref::<Int32Array>());
        let chunk_index_col = batch
            .column_by_name("chunk_index")
            .and_then(|column| column.as_any().downcast_ref::<Int32Array>());

        for row in 0..batch.num_rows() {
            let fused_score = hybrid_relevance_score_at(batch, row);
            let vector_score = vector_score_at(batch, row, fused_score.is_some());
            let keyword_score = if fused_score.is_some() {
                None
            } else {
                raw_score_at(batch, row)
            };
            hits.push(RetrievalCandidate {
                id: id_col
                    .map(|column| column.value(row).to_string())
                    .unwrap_or_default(),
                source_path: source_col.value(row).to_string(),
                chunk_text: text_col.value(row).to_string(),
                metadata: metadata_col
                    .map(|column| column.value(row).to_string())
                    .unwrap_or_default(),
                page: page_col.map(|column| column.value(row)).unwrap_or_default(),
                chunk_index: chunk_index_col
                    .map(|column| column.value(row))
                    .unwrap_or_default(),
                vector_score,
                keyword_score,
                fused_score: fused_score.or(vector_score).or(keyword_score),
                rerank_score: None,
            });
        }
    }

    Ok(hits)
}

fn hybrid_relevance_score_at(batch: &RecordBatch, row: usize) -> Option<f32> {
    numeric_column_value(batch, row, "_relevance_score")
}

fn raw_score_at(batch: &RecordBatch, row: usize) -> Option<f32> {
    ["_score", "score"]
        .into_iter()
        .find_map(|name| numeric_column_value(batch, row, name))
}

fn vector_score_at(batch: &RecordBatch, row: usize, has_hybrid_relevance: bool) -> Option<f32> {
    for name in ["_distance", "distance"] {
        if let Some(distance) = numeric_column_value(batch, row, name) {
            return Some(distance_to_similarity(distance));
        }
    }

    if has_hybrid_relevance {
        return None;
    }

    raw_score_at(batch, row)
}

fn numeric_column_value(batch: &RecordBatch, row: usize, name: &str) -> Option<f32> {
    let column = batch.column_by_name(name)?;
    if let Some(values) = column.as_any().downcast_ref::<Float32Array>() {
        return Some(values.value(row));
    }
    if let Some(values) = column.as_any().downcast_ref::<Float64Array>() {
        return Some(values.value(row) as f32);
    }
    None
}

fn distance_to_similarity(distance: f32) -> f32 {
    1.0 / (1.0 + distance.max(0.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Float32Array, Int32Array, StringArray};
    use arrow_schema::{DataType, Field, Schema};
    use std::sync::Arc;

    #[test]
    fn test_distance_to_similarity_makes_smaller_distances_score_higher() {
        assert!(distance_to_similarity(0.2) > distance_to_similarity(0.8));
    }

    #[test]
    fn test_extract_candidates_prefers_hybrid_relevance_over_raw_score() {
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("source_path", DataType::Utf8, false),
                Field::new("chunk_text", DataType::Utf8, false),
                Field::new("page", DataType::Int32, false),
                Field::new("chunk_index", DataType::Int32, false),
                Field::new("_score", DataType::Float32, true),
                Field::new("_relevance_score", DataType::Float32, false),
            ])),
            vec![
                Arc::new(StringArray::from_iter_values(["docs/totoro.md"])),
                Arc::new(StringArray::from_iter_values([
                    "Chibi Totoro and Chu Totoro",
                ])),
                Arc::new(Int32Array::from_iter_values([3])),
                Arc::new(Int32Array::from_iter_values([0])),
                Arc::new(Float32Array::from_iter_values([0.000001])),
                Arc::new(Float32Array::from_iter_values([0.032795697])),
            ],
        )
        .unwrap();

        let candidates = extract_candidates(&[batch]).unwrap();

        assert_eq!(candidates[0].fused_score, Some(0.032795697));
        assert_eq!(candidates[0].keyword_score, None);
        assert_eq!(candidates[0].vector_score, None);
    }
}
