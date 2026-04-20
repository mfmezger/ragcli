//! LanceDB storage helpers and store metadata utilities.

use crate::fsutil::write_atomic;
use crate::source_kind::{ContentCategory, SourceKind};
use anyhow::{bail, Context, Result};
use arrow_array::types::Float32Type;
use arrow_array::{FixedSizeListArray, Int32Array, RecordBatch, RecordBatchIterator, StringArray};
use arrow_schema::{DataType, Field, Schema};
use futures::TryStreamExt;
use lancedb::database::CreateTableMode;
use lancedb::index::scalar::FtsIndexBuilder;
use lancedb::index::Index;
use lancedb::index::IndexType;
use lancedb::query::{ExecutableQuery, QueryBase, Select};
use lancedb::{connect, Connection, Error as LanceDbError, Table};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Default table name used for stored chunks.
pub const DEFAULT_TABLE_NAME: &str = "chunks";
/// Default column used for full-text search indexing.
pub const DEFAULT_FTS_COLUMN: &str = "chunk_text";
const STORE_SCHEMA_VERSION: u32 = 2;

/// In-memory representation of a chunk before it is written to LanceDB.
#[derive(Debug)]
pub struct ChunkRow {
    /// Stable row identifier.
    pub id: String,
    /// Original source file path.
    pub source_path: String,
    /// Chunk text stored for retrieval.
    pub chunk_text: String,
    /// Stable content hash for the chunk.
    pub chunk_hash: String,
    /// Indexed content format such as `text`, `markdown`, `pdf`, or `image`.
    pub format: String,
    /// Page number for paginated sources, or `0` when not applicable.
    pub page: i32,
    /// Zero-based chunk index within the source unit.
    pub chunk_index: i32,
    /// JSON-encoded metadata associated with the chunk.
    pub metadata: String,
    /// Embedding vector for semantic search.
    pub embedding: Vec<f32>,
}

/// Metadata recorded for a persisted store.
#[derive(Debug, Serialize, Deserialize)]
pub struct StoreMetadata {
    /// Schema version of the on-disk metadata format.
    pub schema_version: u32,
    /// Embedding model used to build the store.
    pub embed_model: String,
    /// Embedding dimensionality stored in LanceDB.
    pub embedding_dim: usize,
    /// Chunk size used during indexing.
    pub chunk_size: usize,
    /// Chunk overlap used during indexing.
    pub chunk_overlap: usize,
}

/// Per-source change detection data loaded from stored chunk metadata.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct SourceFingerprint {
    /// Stable content fingerprint for the source file.
    pub fingerprint: String,
    /// Source file size in bytes at indexing time.
    pub size_bytes: u64,
    /// Source file modification time in Unix milliseconds at indexing time.
    pub modified_unix_ms: u64,
}

/// Counts of source files by content kind.
#[derive(Debug, Default, Serialize)]
pub struct ContentKindCounts {
    /// Number of text or Markdown files.
    pub text_files: usize,
    /// Number of PDF files.
    pub pdf_files: usize,
    /// Number of image files.
    pub image_files: usize,
    /// Number of files with other or unknown formats.
    pub other_files: usize,
}

/// Per-source chunk statistics.
#[derive(Debug, Default, Serialize)]
pub struct SourceChunkStat {
    /// Source file path.
    pub source_path: String,
    /// Number of chunks stored for this source.
    pub chunks: usize,
    /// Total characters stored for this source.
    pub chars: usize,
    /// Approximate token count for this source.
    pub estimated_tokens: usize,
}

/// Per-source metadata used for store inspection commands.
#[derive(Debug, Default, Serialize, Clone, PartialEq, Eq)]
pub struct IndexedSource {
    /// Source file path.
    pub source_path: String,
    /// Indexed format label such as `text`, `markdown`, `pdf`, or `image`.
    pub format: String,
    /// Number of chunks stored for this source.
    pub chunks: usize,
    /// Total characters stored for this source.
    pub chars: usize,
    /// Approximate token count for this source.
    pub estimated_tokens: usize,
    /// Number of unique pages represented for paginated sources.
    pub page_count: usize,
}

/// Lightweight per-source metadata used for maintenance commands.
#[derive(Debug, Default, Serialize, Clone, PartialEq, Eq)]
pub struct IndexedSourceSummary {
    /// Source file path.
    pub source_path: String,
    /// Canonical absolute source file path when recorded at index time.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_absolute_path: Option<String>,
    /// Indexed format label such as `text`, `markdown`, `pdf`, or `image`.
    pub format: String,
    /// Number of chunks stored for this source.
    pub chunks: usize,
    /// Number of unique pages represented for paginated sources.
    pub page_count: usize,
}

/// Aggregate statistics for a store.
#[derive(Debug, Default, Serialize)]
pub struct StoreStats {
    /// Total number of stored chunks.
    pub total_chunks: usize,
    /// Number of unique source files.
    pub unique_sources: usize,
    /// Number of unique PDF pages represented in the store.
    pub pdf_pages: usize,
    /// Total characters stored across all chunks.
    pub total_chars: usize,
    /// Approximate total token count across all chunks.
    pub estimated_tokens: usize,
    /// Smallest chunk size in characters.
    pub min_chunk_chars: usize,
    /// Largest chunk size in characters.
    pub max_chunk_chars: usize,
    /// Counts of sources by content kind.
    pub content_kinds: ContentKindCounts,
    /// Largest sources by chunk count.
    pub top_sources: Vec<SourceChunkStat>,
}

impl StoreMetadata {
    /// Validates that a query is using the same embedding model as the store.
    pub fn validate_query_model(&self, embed_model: &str) -> Result<()> {
        if self.embed_model != embed_model {
            bail!(
                "embedding model mismatch: store was built with {}, current config resolves to {}",
                self.embed_model,
                embed_model
            );
        }
        Ok(())
    }
}

/// Connects to the store's LanceDB database.
pub async fn connect_db(store: &Path) -> Result<Connection> {
    let db_uri = store.join("lancedb").to_string_lossy().to_string();
    connect(&db_uri)
        .execute()
        .await
        .context("connect to LanceDB")
}

/// Returns the path to the store metadata file.
pub fn metadata_path(store: &Path) -> PathBuf {
    store.join("meta").join("store.toml")
}

/// Loads store metadata from disk.
pub fn load_metadata(store: &Path) -> Result<StoreMetadata> {
    let path = metadata_path(store);
    let raw = fs::read_to_string(&path)
        .with_context(|| format!("read store metadata: {}", path.display()))?;
    let metadata: StoreMetadata = toml::from_str(&raw)
        .with_context(|| format!("parse store metadata: {}", path.display()))?;
    Ok(metadata)
}

/// Ensures store metadata exists and matches the current indexing settings.
pub fn ensure_metadata(
    store: &Path,
    embed_model: &str,
    embedding_dim: usize,
    chunk_size: usize,
    chunk_overlap: usize,
) -> Result<()> {
    let path = metadata_path(store);
    let next = StoreMetadata {
        schema_version: STORE_SCHEMA_VERSION,
        embed_model: embed_model.to_string(),
        embedding_dim,
        chunk_size,
        chunk_overlap,
    };

    if path.exists() {
        let current = load_metadata(store)?;
        if current.schema_version != next.schema_version
            || current.embed_model != next.embed_model
            || current.embedding_dim != next.embedding_dim
            || current.chunk_size != next.chunk_size
            || current.chunk_overlap != next.chunk_overlap
        {
            bail!(
                "store metadata mismatch; use a new --name or remove the old store before re-indexing"
            );
        }
        return Ok(());
    }

    let raw = toml::to_string_pretty(&next)?;
    write_atomic(&path, &raw)?;
    Ok(())
}

async fn open_table_if_exists(db: &Connection) -> Result<Option<Table>> {
    match db.open_table(DEFAULT_TABLE_NAME).execute().await {
        Ok(table) => Ok(Some(table)),
        Err(LanceDbError::TableNotFound { .. }) => Ok(None),
        Err(err) => Err(err.into()),
    }
}

/// Loads stored per-source fingerprint data from chunk metadata.
pub async fn load_source_fingerprints(
    db: &Connection,
) -> Result<BTreeMap<String, SourceFingerprint>> {
    let Some(table) = open_table_if_exists(db).await? else {
        return Ok(BTreeMap::new());
    };
    let mut stream = table
        .query()
        .select(Select::columns(&["source_path", "metadata"]))
        .execute()
        .await?;

    let mut fingerprints = BTreeMap::new();
    while let Some(batch) = stream.try_next().await? {
        let source_col = batch
            .column_by_name("source_path")
            .context("source_path column missing")?
            .as_any()
            .downcast_ref::<StringArray>()
            .context("source_path column type")?;
        let metadata_col = batch
            .column_by_name("metadata")
            .context("metadata column missing")?
            .as_any()
            .downcast_ref::<StringArray>()
            .context("metadata column type")?;

        for row_idx in 0..batch.num_rows() {
            let source = source_col.value(row_idx);
            if fingerprints.contains_key(source) {
                continue;
            }

            let Ok(metadata): Result<serde_json::Value, _> =
                serde_json::from_str(metadata_col.value(row_idx))
            else {
                continue;
            };
            let Some(fingerprint) = metadata
                .get("source_fingerprint")
                .and_then(serde_json::Value::as_str)
            else {
                continue;
            };
            let size_bytes = metadata
                .get("source_size_bytes")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or_default();
            let modified_unix_ms = metadata
                .get("source_modified_unix_ms")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or_default();
            fingerprints
                .entry(source.to_string())
                .or_insert_with(|| SourceFingerprint {
                    fingerprint: fingerprint.to_string(),
                    size_bytes,
                    modified_unix_ms,
                });
        }
    }

    Ok(fingerprints)
}

pub async fn replace_source_rows(
    db: &Connection,
    rows: &[ChunkRow],
    source_paths: &[String],
) -> Result<()> {
    let table = open_or_create_table(db, rows).await?;

    if let Some(filter) = build_source_delete_filter(source_paths) {
        table.delete(&filter).await?;
    }

    if !rows.is_empty() {
        let schema = build_schema(rows[0].embedding.len());
        let batch = build_record_batch(schema.clone(), rows)?;
        let batches = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema);
        table.add(Box::new(batches)).execute().await?;
    }
    ensure_fts_index(&table, true).await?;
    Ok(())
}

/// Lists indexed sources together with aggregate per-source metadata.
pub async fn list_indexed_sources(db: &Connection) -> Result<Vec<IndexedSource>> {
    let Some(table) = open_table_if_exists(db).await? else {
        return Ok(Vec::new());
    };

    let mut stream = table
        .query()
        .select(Select::columns(&[
            "source_path",
            "chunk_text",
            "page",
            "format",
        ]))
        .execute()
        .await?;
    let mut sources = BTreeMap::new();
    let mut pages_by_source = BTreeMap::new();
    while let Some(batch) = stream.try_next().await? {
        accumulate_indexed_sources(&mut sources, &mut pages_by_source, &batch)?;
    }
    Ok(finalize_indexed_sources(sources, pages_by_source))
}

/// Lists lightweight indexed source metadata without scanning chunk text.
pub async fn list_indexed_source_summaries(db: &Connection) -> Result<Vec<IndexedSourceSummary>> {
    let Some(table) = open_table_if_exists(db).await? else {
        return Ok(Vec::new());
    };

    let mut stream = table
        .query()
        .select(Select::columns(&[
            "source_path",
            "page",
            "format",
            "metadata",
        ]))
        .execute()
        .await?;
    let mut sources = BTreeMap::new();
    let mut pages_by_source = BTreeMap::new();
    while let Some(batch) = stream.try_next().await? {
        accumulate_indexed_source_summaries(&mut sources, &mut pages_by_source, &batch)?;
    }
    Ok(finalize_indexed_source_summaries(sources, pages_by_source))
}

/// Loads lightweight metadata for a single indexed source path.
pub async fn load_indexed_source_summary(
    db: &Connection,
    source_path: &str,
) -> Result<Option<IndexedSourceSummary>> {
    let Some(table) = open_table_if_exists(db).await? else {
        return Ok(None);
    };

    let mut stream = table
        .query()
        .only_if(format!("source_path = {}", sql_string(source_path)))
        .select(Select::columns(&[
            "source_path",
            "page",
            "format",
            "metadata",
        ]))
        .execute()
        .await?;
    let mut sources = BTreeMap::new();
    let mut pages_by_source = BTreeMap::new();
    while let Some(batch) = stream.try_next().await? {
        accumulate_indexed_source_summaries(&mut sources, &mut pages_by_source, &batch)?;
    }
    Ok(finalize_indexed_source_summaries(sources, pages_by_source)
        .into_iter()
        .next())
}

/// Deletes all stored rows for the given source paths.
pub async fn delete_source_rows(db: &Connection, source_paths: &[String]) -> Result<()> {
    let Some(filter) = build_source_delete_filter(source_paths) else {
        return Ok(());
    };
    let Some(table) = open_table_if_exists(db).await? else {
        return Ok(());
    };

    table.delete(&filter).await?;
    ensure_fts_index(&table, true).await?;

    Ok(())
}

/// Deletes every stored row in the selected table.
pub async fn clear_store_rows(db: &Connection) -> Result<()> {
    let Some(table) = open_table_if_exists(db).await? else {
        return Ok(());
    };

    table.delete("source_path IS NOT NULL").await?;
    ensure_fts_index(&table, true).await?;

    Ok(())
}

/// Ensures the full-text search index exists for the default text column.
pub async fn ensure_fts_index(table: &Table, replace: bool) -> Result<()> {
    let has_fts = table.list_indices().await?.into_iter().any(|index| {
        index.index_type == IndexType::FTS
            && index.columns.len() == 1
            && index.columns[0] == DEFAULT_FTS_COLUMN
    });

    if has_fts && !replace {
        return Ok(());
    }

    let mut builder = table.create_index(
        &[DEFAULT_FTS_COLUMN],
        Index::FTS(FtsIndexBuilder::default()),
    );
    builder = builder.replace(replace);
    builder.execute().await.context("create FTS index")?;
    Ok(())
}

async fn open_or_create_table(db: &Connection, rows: &[ChunkRow]) -> Result<Table> {
    match db.open_table(DEFAULT_TABLE_NAME).execute().await {
        Ok(table) => Ok(table),
        Err(_) => {
            let dim = rows
                .first()
                .map(|row| row.embedding.len())
                .context("cannot create table without rows")?;
            let schema = build_schema(dim);
            let batch = build_record_batch(schema.clone(), rows)?;
            let batches = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema);
            db.create_table(DEFAULT_TABLE_NAME, Box::new(batches))
                .mode(CreateTableMode::Overwrite)
                .execute()
                .await
                .context("create table")
        }
    }
}

fn build_schema(dim: usize) -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("source_path", DataType::Utf8, false),
        Field::new("chunk_text", DataType::Utf8, false),
        Field::new("chunk_hash", DataType::Utf8, false),
        Field::new("format", DataType::Utf8, false),
        Field::new("page", DataType::Int32, false),
        Field::new("chunk_index", DataType::Int32, false),
        Field::new("metadata", DataType::Utf8, false),
        Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                dim as i32,
            ),
            true,
        ),
    ]))
}

fn build_record_batch(schema: Arc<Schema>, rows: &[ChunkRow]) -> Result<RecordBatch> {
    let ids = StringArray::from_iter_values(rows.iter().map(|r| r.id.as_str()));
    let source_paths = StringArray::from_iter_values(rows.iter().map(|r| r.source_path.as_str()));
    let chunk_texts = StringArray::from_iter_values(rows.iter().map(|r| r.chunk_text.as_str()));
    let chunk_hashes = StringArray::from_iter_values(rows.iter().map(|r| r.chunk_hash.as_str()));
    let formats = StringArray::from_iter_values(rows.iter().map(|r| r.format.as_str()));
    let pages = Int32Array::from_iter_values(rows.iter().map(|r| r.page));
    let chunk_indices = Int32Array::from_iter_values(rows.iter().map(|r| r.chunk_index));
    let metadata = StringArray::from_iter_values(rows.iter().map(|r| r.metadata.as_str()));
    let vectors = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
        rows.iter()
            .map(|r| Some(r.embedding.iter().map(|v| Some(*v)).collect::<Vec<_>>())),
        rows[0].embedding.len() as i32,
    );

    Ok(RecordBatch::try_new(
        schema,
        vec![
            Arc::new(ids),
            Arc::new(source_paths),
            Arc::new(chunk_texts),
            Arc::new(chunk_hashes),
            Arc::new(formats),
            Arc::new(pages),
            Arc::new(chunk_indices),
            Arc::new(metadata),
            Arc::new(vectors),
        ],
    )?)
}

/// Builds a SQL filter expression for retrieval constraints.
pub fn build_retrieval_filter(
    source: Option<&str>,
    path_prefix: Option<&str>,
    page: Option<i32>,
    format: Option<&str>,
) -> Option<String> {
    let mut clauses = Vec::new();

    if let Some(source) = source {
        clauses.push(format!("source_path = {}", sql_string(source)));
    }

    if let Some(path_prefix) = path_prefix {
        clauses.push(format!(
            "source_path LIKE {} ESCAPE '\\'",
            sql_like_prefix(path_prefix)
        ));
    }

    if let Some(page) = page {
        clauses.push(format!("page = {page}"));
    }

    if let Some(format) = format {
        clauses.push(format!("format = {}", sql_string(format)));
    }

    if clauses.is_empty() {
        None
    } else {
        Some(clauses.join(" AND "))
    }
}

/// Extracts retrieval contexts from query result batches.
#[cfg(test)]
pub fn extract_contexts(batches: &[RecordBatch]) -> Result<Vec<String>> {
    let mut out = Vec::new();
    for batch in batches {
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
        for i in 0..batch.num_rows() {
            out.push(format!(
                "Source: {}\n{}",
                source_col.value(i),
                text_col.value(i)
            ));
        }
    }
    Ok(out)
}

/// Collects lightweight per-source metadata from stored chunk batches.
#[cfg(test)]
pub fn collect_indexed_source_summaries(
    batches: &[RecordBatch],
) -> Result<Vec<IndexedSourceSummary>> {
    let mut sources = BTreeMap::new();
    let mut pages_by_source = BTreeMap::new();
    for batch in batches {
        accumulate_indexed_source_summaries(&mut sources, &mut pages_by_source, batch)?;
    }
    Ok(finalize_indexed_source_summaries(sources, pages_by_source))
}

fn accumulate_indexed_source_summaries(
    sources: &mut BTreeMap<String, IndexedSourceSummary>,
    pages_by_source: &mut BTreeMap<String, BTreeSet<i32>>,
    batch: &RecordBatch,
) -> Result<()> {
    let source_col = batch
        .column_by_name("source_path")
        .context("source_path column missing")?
        .as_any()
        .downcast_ref::<StringArray>()
        .context("source_path column type")?;
    let page_col = batch
        .column_by_name("page")
        .context("page column missing")?
        .as_any()
        .downcast_ref::<Int32Array>()
        .context("page column type")?;
    let format_col = batch
        .column_by_name("format")
        .context("format column missing")?
        .as_any()
        .downcast_ref::<StringArray>()
        .context("format column type")?;
    let metadata_col = batch
        .column_by_name("metadata")
        .context("metadata column missing")?
        .as_any()
        .downcast_ref::<StringArray>()
        .context("metadata column type")?;

    for i in 0..batch.num_rows() {
        let source = source_col.value(i);
        let entry = sources
            .entry(source.to_string())
            .or_insert_with(|| IndexedSourceSummary {
                source_path: source.to_string(),
                format: format_col.value(i).to_string(),
                ..Default::default()
            });
        if entry.source_absolute_path.is_none() {
            entry.source_absolute_path = source_absolute_path_from_metadata(metadata_col.value(i));
        }
        entry.chunks += 1;

        let page = page_col.value(i);
        if page > 0 {
            pages_by_source
                .entry(source.to_string())
                .or_default()
                .insert(page);
        }
    }

    Ok(())
}

fn finalize_indexed_source_summaries(
    mut sources: BTreeMap<String, IndexedSourceSummary>,
    pages_by_source: BTreeMap<String, BTreeSet<i32>>,
) -> Vec<IndexedSourceSummary> {
    for (source, pages) in pages_by_source {
        if let Some(entry) = sources.get_mut(&source) {
            entry.page_count = pages.len();
        }
    }
    sources.into_values().collect()
}

/// Collects per-source metadata from stored chunk batches.
#[cfg(test)]
pub fn collect_indexed_sources(batches: &[RecordBatch]) -> Result<Vec<IndexedSource>> {
    let mut sources = BTreeMap::new();
    let mut pages_by_source = BTreeMap::new();
    for batch in batches {
        accumulate_indexed_sources(&mut sources, &mut pages_by_source, batch)?;
    }
    Ok(finalize_indexed_sources(sources, pages_by_source))
}

fn accumulate_indexed_sources(
    sources: &mut BTreeMap<String, IndexedSource>,
    pages_by_source: &mut BTreeMap<String, BTreeSet<i32>>,
    batch: &RecordBatch,
) -> Result<()> {
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
    let page_col = batch
        .column_by_name("page")
        .context("page column missing")?
        .as_any()
        .downcast_ref::<Int32Array>()
        .context("page column type")?;
    let format_col = batch
        .column_by_name("format")
        .context("format column missing")?
        .as_any()
        .downcast_ref::<StringArray>()
        .context("format column type")?;

    for i in 0..batch.num_rows() {
        let source = source_col.value(i);
        let text = text_col.value(i);
        let char_count = text.chars().count();
        let entry = sources
            .entry(source.to_string())
            .or_insert_with(|| IndexedSource {
                source_path: source.to_string(),
                format: format_col.value(i).to_string(),
                ..Default::default()
            });
        entry.chunks += 1;
        entry.chars += char_count;
        entry.estimated_tokens += estimate_token_count_from_chars(char_count);

        let page = page_col.value(i);
        if page > 0 {
            pages_by_source
                .entry(source.to_string())
                .or_default()
                .insert(page);
        }
    }

    Ok(())
}

fn finalize_indexed_sources(
    mut sources: BTreeMap<String, IndexedSource>,
    pages_by_source: BTreeMap<String, BTreeSet<i32>>,
) -> Vec<IndexedSource> {
    for (source, pages) in pages_by_source {
        if let Some(entry) = sources.get_mut(&source) {
            entry.page_count = pages.len();
        }
    }
    sources.into_values().collect()
}

/// Collects summary statistics from stored chunk batches.
pub fn collect_store_stats(batches: &[RecordBatch], top_n: usize) -> Result<StoreStats> {
    let mut stats = StoreStats::default();
    let mut source_stats: BTreeMap<String, SourceChunkStat> = BTreeMap::new();
    let mut pdf_pages = BTreeSet::new();

    for batch in batches {
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
        let page_col = batch
            .column_by_name("page")
            .context("page column missing")?
            .as_any()
            .downcast_ref::<Int32Array>()
            .context("page column type")?;
        let mut current_source = None;
        let mut current_kind = SourceKind::Unsupported;

        for i in 0..batch.num_rows() {
            let source = source_col.value(i);
            let text = text_col.value(i);
            let chars = text.chars().count();
            let estimated_tokens = estimate_token_count(text);

            stats.total_chunks += 1;
            stats.total_chars += chars;
            stats.estimated_tokens += estimated_tokens;
            if stats.total_chunks == 1 {
                stats.min_chunk_chars = chars;
                stats.max_chunk_chars = chars;
            } else {
                stats.min_chunk_chars = stats.min_chunk_chars.min(chars);
                stats.max_chunk_chars = stats.max_chunk_chars.max(chars);
            }

            let entry = source_stats
                .entry(source.to_string())
                .or_insert_with(|| SourceChunkStat {
                    source_path: source.to_string(),
                    ..Default::default()
                });
            entry.chunks += 1;
            entry.chars += chars;
            entry.estimated_tokens += estimated_tokens;

            if current_source != Some(source) {
                current_source = Some(source);
                current_kind = SourceKind::from_path(Path::new(source));
            }

            if current_kind == SourceKind::Pdf && page_col.value(i) > 0 {
                pdf_pages.insert((source.to_string(), page_col.value(i)));
            }
        }
    }

    stats.unique_sources = source_stats.len();
    stats.pdf_pages = pdf_pages.len();

    for source in source_stats.keys() {
        match SourceKind::from_path(Path::new(source)).content_category() {
            ContentCategory::Pdf => stats.content_kinds.pdf_files += 1,
            ContentCategory::Image => stats.content_kinds.image_files += 1,
            ContentCategory::Text => stats.content_kinds.text_files += 1,
            ContentCategory::Other => stats.content_kinds.other_files += 1,
        }
    }

    let mut top_sources = source_stats.into_values().collect::<Vec<_>>();
    top_sources.sort_by(|a, b| {
        b.chunks
            .cmp(&a.chunks)
            .then_with(|| b.estimated_tokens.cmp(&a.estimated_tokens))
            .then_with(|| a.source_path.cmp(&b.source_path))
    });
    top_sources.truncate(top_n);
    stats.top_sources = top_sources;

    Ok(stats)
}

/// Removes a single `<think>...</think>` block from a model response when present.
pub fn strip_thinking(text: &str) -> String {
    if let Some(start) = text.find("<think>") {
        let end = text.find("</think>");
        let end_idx = end.map(|idx| idx + "</think>".len()).unwrap_or(start);
        let mut out = String::new();
        out.push_str(&text[..start]);
        out.push_str(&text[end_idx..]);
        return out;
    }
    text.to_string()
}

fn build_source_delete_filter(source_paths: &[String]) -> Option<String> {
    let source_paths = source_paths.iter().cloned().collect::<BTreeSet<_>>();
    match source_paths.len() {
        0 => None,
        1 => source_paths
            .iter()
            .next()
            .map(|source_path| format!("source_path = {}", sql_string(source_path))),
        _ => Some(format!(
            "source_path IN ({})",
            source_paths
                .iter()
                .map(|source_path| sql_string(source_path))
                .collect::<Vec<_>>()
                .join(", ")
        )),
    }
}

fn sql_string(value: &str) -> String {
    format!("'{}'", value.replace('\'', "''"))
}

fn sql_like_prefix(value: &str) -> String {
    let escaped = value
        .replace('\\', "\\\\")
        .replace('%', "\\%")
        .replace('_', "\\_")
        .replace('\'', "''");
    format!("'{escaped}%'")
}

fn source_absolute_path_from_metadata(raw: &str) -> Option<String> {
    serde_json::from_str::<serde_json::Value>(raw)
        .ok()
        .and_then(|metadata| {
            metadata
                .get("source_absolute_path")
                .and_then(serde_json::Value::as_str)
                .map(str::to_string)
        })
}

fn estimate_token_count(text: &str) -> usize {
    estimate_token_count_from_chars(text.chars().count())
}

fn estimate_token_count_from_chars(chars: usize) -> usize {
    chars.div_ceil(4)
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::TryStreamExt;
    use lancedb::index::scalar::FullTextSearchQuery;
    use lancedb::query::ExecutableQuery;
    use lancedb::query::QueryBase;

    fn sample_row(source_path: &str, text: &str, value: f32) -> ChunkRow {
        ChunkRow {
            id: format!("{}-{}", source_path, text),
            source_path: source_path.to_string(),
            chunk_text: text.to_string(),
            chunk_hash: text.to_string(),
            format: SourceKind::from_path(Path::new(source_path))
                .format_label()
                .unwrap_or("text")
                .to_string(),
            page: 0,
            chunk_index: 0,
            metadata: "{}".to_string(),
            embedding: vec![value, value],
        }
    }

    #[tokio::test]
    async fn test_load_source_fingerprints_reads_metadata_values() {
        let dir = tempfile::tempdir().unwrap();
        let db = connect_db(dir.path()).await.unwrap();

        replace_source_rows(
            &db,
            &[
                ChunkRow {
                    metadata: r#"{"source_fingerprint":"fp-a"}"#.to_string(),
                    ..sample_row("a.txt", "alpha", 1.0)
                },
                ChunkRow {
                    metadata: r#"{"source_fingerprint":"fp-b"}"#.to_string(),
                    ..sample_row("b.txt", "beta", 2.0)
                },
            ],
            &["a.txt".to_string(), "b.txt".to_string()],
        )
        .await
        .unwrap();

        let fingerprints = load_source_fingerprints(&db).await.unwrap();
        assert_eq!(
            fingerprints.get("a.txt"),
            Some(&SourceFingerprint {
                fingerprint: "fp-a".to_string(),
                size_bytes: 0,
                modified_unix_ms: 0,
            })
        );
        assert_eq!(
            fingerprints.get("b.txt"),
            Some(&SourceFingerprint {
                fingerprint: "fp-b".to_string(),
                size_bytes: 0,
                modified_unix_ms: 0,
            })
        );
    }

    #[tokio::test]
    async fn test_replace_source_rows_removes_duplicates() {
        let dir = tempfile::tempdir().unwrap();
        let db = connect_db(dir.path()).await.unwrap();

        replace_source_rows(
            &db,
            &[sample_row("a.txt", "old", 1.0)],
            &["a.txt".to_string()],
        )
        .await
        .unwrap();
        replace_source_rows(
            &db,
            &[
                sample_row("a.txt", "new", 2.0),
                sample_row("b.txt", "other", 3.0),
            ],
            &["a.txt".to_string(), "b.txt".to_string()],
        )
        .await
        .unwrap();

        let table = db.open_table(DEFAULT_TABLE_NAME).execute().await.unwrap();
        let batches: Vec<RecordBatch> = table
            .query()
            .execute()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let contexts = extract_contexts(&batches).unwrap();

        assert_eq!(contexts.len(), 2);
        assert!(contexts.iter().any(|ctx| ctx.contains("new")));
        assert!(!contexts.iter().any(|ctx| ctx.contains("old")));
    }

    #[test]
    fn test_build_source_delete_filter_uses_in_clause_for_multiple_sources() {
        let filter = build_source_delete_filter(&[
            "b.txt".to_string(),
            "a.txt".to_string(),
            "a.txt".to_string(),
        ])
        .unwrap();
        assert_eq!(filter, "source_path IN ('a.txt', 'b.txt')");
    }

    #[tokio::test]
    async fn test_delete_source_rows_removes_only_requested_source() {
        let dir = tempfile::tempdir().unwrap();
        let db = connect_db(dir.path()).await.unwrap();

        replace_source_rows(
            &db,
            &[
                sample_row("a.txt", "alpha", 1.0),
                sample_row("b.txt", "beta", 2.0),
            ],
            &["a.txt".to_string(), "b.txt".to_string()],
        )
        .await
        .unwrap();

        delete_source_rows(&db, &["a.txt".to_string()])
            .await
            .unwrap();

        let sources = list_indexed_sources(&db).await.unwrap();
        assert_eq!(sources.len(), 1);
        assert_eq!(sources[0].source_path, "b.txt");
    }

    #[tokio::test]
    async fn test_clear_store_rows_removes_every_source() {
        let dir = tempfile::tempdir().unwrap();
        let db = connect_db(dir.path()).await.unwrap();

        replace_source_rows(
            &db,
            &[
                sample_row("a.txt", "alpha", 1.0),
                sample_row("b.txt", "beta", 2.0),
            ],
            &["a.txt".to_string(), "b.txt".to_string()],
        )
        .await
        .unwrap();

        clear_store_rows(&db).await.unwrap();

        assert!(list_indexed_source_summaries(&db).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_hybrid_search_keeps_keyword_match() {
        let dir = tempfile::tempdir().unwrap();
        let db = connect_db(dir.path()).await.unwrap();

        replace_source_rows(
            &db,
            &[
                sample_row("a.txt", "semantic neighbor", -10.0),
                sample_row("b.txt", "rarekeyword exact match", 100.0),
            ],
            &["a.txt".to_string(), "b.txt".to_string()],
        )
        .await
        .unwrap();

        let table = db.open_table(DEFAULT_TABLE_NAME).execute().await.unwrap();
        let batches: Vec<RecordBatch> = table
            .query()
            .full_text_search(FullTextSearchQuery::new("rarekeyword".to_string()))
            .nearest_to(&[-10.0, -10.0])
            .unwrap()
            .limit(2)
            .execute()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let contexts = extract_contexts(&batches).unwrap();

        assert_eq!(contexts.len(), 2);
        assert!(contexts.iter().any(|ctx| ctx.contains("semantic neighbor")));
        assert!(contexts
            .iter()
            .any(|ctx| ctx.contains("rarekeyword exact match")));
    }

    #[tokio::test]
    async fn test_hybrid_search_can_filter_by_source_and_page() {
        let dir = tempfile::tempdir().unwrap();
        let db = connect_db(dir.path()).await.unwrap();

        replace_source_rows(
            &db,
            &[
                ChunkRow {
                    page: 1,
                    ..sample_row("docs/guide.pdf", "rarekeyword page one", -10.0)
                },
                ChunkRow {
                    page: 2,
                    ..sample_row("docs/guide.pdf", "rarekeyword page two", 100.0)
                },
                ChunkRow {
                    page: 2,
                    ..sample_row("notes/todo.txt", "rarekeyword wrong source", 100.0)
                },
            ],
            &["docs/guide.pdf".to_string(), "notes/todo.txt".to_string()],
        )
        .await
        .unwrap();

        let filter = build_retrieval_filter(Some("docs/guide.pdf"), None, Some(2), None).unwrap();
        let table = db.open_table(DEFAULT_TABLE_NAME).execute().await.unwrap();
        let batches: Vec<RecordBatch> = table
            .query()
            .full_text_search(FullTextSearchQuery::new("rarekeyword".to_string()))
            .nearest_to(&[-10.0, -10.0])
            .unwrap()
            .only_if(filter)
            .limit(5)
            .execute()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let contexts = extract_contexts(&batches).unwrap();

        assert_eq!(contexts.len(), 1);
        assert!(contexts[0].contains("docs/guide.pdf"));
        assert!(contexts[0].contains("rarekeyword page two"));
    }

    #[tokio::test]
    async fn test_hybrid_search_can_filter_by_path_prefix() {
        let dir = tempfile::tempdir().unwrap();
        let db = connect_db(dir.path()).await.unwrap();

        replace_source_rows(
            &db,
            &[
                sample_row("docs/a.txt", "rarekeyword docs", -10.0),
                sample_row("notes/b.txt", "rarekeyword notes", 100.0),
            ],
            &["docs/a.txt".to_string(), "notes/b.txt".to_string()],
        )
        .await
        .unwrap();

        let filter = build_retrieval_filter(None, Some("docs/"), None, None).unwrap();
        let table = db.open_table(DEFAULT_TABLE_NAME).execute().await.unwrap();
        let batches: Vec<RecordBatch> = table
            .query()
            .full_text_search(FullTextSearchQuery::new("rarekeyword".to_string()))
            .nearest_to(&[-10.0, -10.0])
            .unwrap()
            .only_if(filter)
            .limit(5)
            .execute()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let contexts = extract_contexts(&batches).unwrap();

        assert_eq!(contexts.len(), 1);
        assert!(contexts[0].contains("docs/a.txt"));
        assert!(!contexts[0].contains("notes/b.txt"));
    }

    #[tokio::test]
    async fn test_hybrid_search_can_filter_by_format() {
        let dir = tempfile::tempdir().unwrap();
        let db = connect_db(dir.path()).await.unwrap();

        replace_source_rows(
            &db,
            &[
                sample_row("docs/a.md", "rarekeyword markdown", -10.0),
                sample_row("docs/b.txt", "rarekeyword text", 100.0),
            ],
            &["docs/a.md".to_string(), "docs/b.txt".to_string()],
        )
        .await
        .unwrap();

        let filter = build_retrieval_filter(None, None, None, Some("markdown")).unwrap();
        let table = db.open_table(DEFAULT_TABLE_NAME).execute().await.unwrap();
        let batches: Vec<RecordBatch> = table
            .query()
            .full_text_search(FullTextSearchQuery::new("rarekeyword".to_string()))
            .nearest_to(&[-10.0, -10.0])
            .unwrap()
            .only_if(filter)
            .limit(5)
            .execute()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let contexts = extract_contexts(&batches).unwrap();

        assert_eq!(contexts.len(), 1);
        assert!(contexts[0].contains("docs/a.md"));
        assert!(!contexts[0].contains("docs/b.txt"));
    }

    #[test]
    fn test_strip_thinking() {
        let input = "<think>hidden</think>\nFinal answer.";
        assert_eq!(strip_thinking(input).trim(), "Final answer.");
    }

    #[test]
    fn test_build_retrieval_filter_combines_clauses() {
        let filter = build_retrieval_filter(
            Some("docs/it's.txt"),
            Some("docs/_v1%/"),
            Some(3),
            Some("markdown"),
        )
        .unwrap();

        assert_eq!(
            filter,
            "source_path = 'docs/it''s.txt' AND source_path LIKE 'docs/\\_v1\\%/%' ESCAPE '\\' AND page = 3 AND format = 'markdown'"
        );
    }

    #[test]
    fn test_collect_indexed_source_summaries_aggregates_metadata() {
        let schema = build_schema(2);
        let batch = build_record_batch(
            schema,
            &[
                ChunkRow {
                    metadata: r#"{"source_absolute_path":"/tmp/notes.md"}"#.to_string(),
                    ..sample_row("notes.md", "hello world", 1.0)
                },
                ChunkRow {
                    page: 1,
                    ..sample_row("paper.pdf", "page one", 2.0)
                },
                ChunkRow {
                    page: 2,
                    ..sample_row("paper.pdf", "page two", 3.0)
                },
                sample_row("image.png", "caption text", 4.0),
            ],
        )
        .unwrap();

        let sources = collect_indexed_source_summaries(&[batch.clone()]).unwrap();

        assert_eq!(sources.len(), 3);
        assert_eq!(sources[0].source_path, "image.png");
        assert_eq!(sources[0].format, "image");
        assert_eq!(sources[1].source_path, "notes.md");
        assert_eq!(sources[1].chunks, 1);
        assert_eq!(
            sources[1].source_absolute_path.as_deref(),
            Some("/tmp/notes.md")
        );
        assert_eq!(sources[2].source_path, "paper.pdf");
        assert_eq!(sources[2].format, "pdf");
        assert_eq!(sources[2].chunks, 2);
        assert_eq!(sources[2].page_count, 2);

        let detailed = collect_indexed_sources(&[batch]).unwrap();
        assert_eq!(detailed[1].chars, "hello world".chars().count());
    }

    #[test]
    fn test_collect_store_stats_counts_content() {
        let schema = build_schema(2);
        let batch = build_record_batch(
            schema,
            &[
                sample_row("notes.md", "hello world", 1.0),
                ChunkRow {
                    page: 1,
                    ..sample_row("paper.pdf", "page one", 2.0)
                },
                ChunkRow {
                    page: 2,
                    ..sample_row("paper.pdf", "page two", 3.0)
                },
                sample_row("image.png", "caption text", 4.0),
            ],
        )
        .unwrap();

        let stats = collect_store_stats(&[batch], 5).unwrap();

        assert_eq!(stats.total_chunks, 4);
        assert_eq!(stats.unique_sources, 3);
        assert_eq!(stats.content_kinds.text_files, 1);
        assert_eq!(stats.content_kinds.pdf_files, 1);
        assert_eq!(stats.content_kinds.image_files, 1);
        assert_eq!(stats.pdf_pages, 2);
        assert_eq!(stats.top_sources[0].source_path, "paper.pdf");
        assert_eq!(stats.top_sources[0].chunks, 2);
    }

    #[test]
    fn test_ensure_metadata_writes_file_without_temp_leftovers() {
        let dir = tempfile::tempdir().unwrap();
        let store = dir.path().join("store");
        fs::create_dir_all(store.join("meta")).unwrap();

        ensure_metadata(&store, "embed-x", 768, 1000, 200).unwrap();

        let metadata = load_metadata(&store).unwrap();
        assert_eq!(metadata.embed_model, "embed-x");
        assert_eq!(metadata.embedding_dim, 768);

        let entries = fs::read_dir(store.join("meta"))
            .unwrap()
            .map(|entry| entry.unwrap().file_name().into_string().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(entries, vec!["store.toml"]);
    }
}
