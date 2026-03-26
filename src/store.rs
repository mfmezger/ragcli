use anyhow::{bail, Context, Result};
use arrow_array::types::Float32Type;
use arrow_array::{FixedSizeListArray, Int32Array, RecordBatch, RecordBatchIterator, StringArray};
use arrow_schema::{DataType, Field, Schema};
use lancedb::database::CreateTableMode;
use lancedb::index::scalar::FtsIndexBuilder;
use lancedb::index::Index;
use lancedb::index::IndexType;
use lancedb::{connect, Connection, Table};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub const DEFAULT_TABLE_NAME: &str = "chunks";
pub const DEFAULT_FTS_COLUMN: &str = "chunk_text";
const STORE_SCHEMA_VERSION: u32 = 1;

#[derive(Debug)]
pub struct ChunkRow {
    pub id: String,
    pub source_path: String,
    pub chunk_text: String,
    pub chunk_hash: String,
    pub page: i32,
    pub chunk_index: i32,
    pub metadata: String,
    pub embedding: Vec<f32>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct StoreMetadata {
    pub schema_version: u32,
    pub embed_model: String,
    pub embedding_dim: usize,
    pub chunk_size: usize,
    pub chunk_overlap: usize,
}

#[derive(Debug, Default)]
pub struct ContentKindCounts {
    pub text_files: usize,
    pub pdf_files: usize,
    pub image_files: usize,
    pub other_files: usize,
}

#[derive(Debug, Default)]
pub struct SourceChunkStat {
    pub source_path: String,
    pub chunks: usize,
    pub chars: usize,
    pub estimated_tokens: usize,
}

#[derive(Debug, Default)]
pub struct StoreStats {
    pub total_chunks: usize,
    pub unique_sources: usize,
    pub pdf_pages: usize,
    pub total_chars: usize,
    pub estimated_tokens: usize,
    pub min_chunk_chars: usize,
    pub max_chunk_chars: usize,
    pub content_kinds: ContentKindCounts,
    pub top_sources: Vec<SourceChunkStat>,
}

impl StoreMetadata {
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

pub async fn connect_db(store: &Path) -> Result<Connection> {
    let db_uri = store.join("lancedb").to_string_lossy().to_string();
    connect(&db_uri)
        .execute()
        .await
        .context("connect to LanceDB")
}

pub fn metadata_path(store: &Path) -> PathBuf {
    store.join("meta").join("store.toml")
}

pub fn load_metadata(store: &Path) -> Result<StoreMetadata> {
    let path = metadata_path(store);
    let raw = fs::read_to_string(&path)
        .with_context(|| format!("read store metadata: {}", path.display()))?;
    let metadata: StoreMetadata = toml::from_str(&raw)
        .with_context(|| format!("parse store metadata: {}", path.display()))?;
    Ok(metadata)
}

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
    fs::write(path, raw)?;
    Ok(())
}

pub async fn replace_source_rows(
    db: &Connection,
    rows: &[ChunkRow],
    source_paths: &[String],
) -> Result<()> {
    let source_paths: BTreeSet<String> = source_paths.iter().cloned().collect();
    let table = open_or_create_table(db, rows).await?;

    for source_path in source_paths {
        table
            .delete(&format!("source_path = {}", sql_string(&source_path)))
            .await?;
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
            Arc::new(pages),
            Arc::new(chunk_indices),
            Arc::new(metadata),
            Arc::new(vectors),
        ],
    )?)
}

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

            match classify_source_kind(source) {
                SourceKind::Pdf => {
                    if page_col.value(i) > 0 {
                        pdf_pages.insert((source.to_string(), page_col.value(i)));
                    }
                }
                _ => {}
            }
        }
    }

    stats.unique_sources = source_stats.len();
    stats.pdf_pages = pdf_pages.len();

    for source in source_stats.keys() {
        match classify_source_kind(source) {
            SourceKind::Pdf => stats.content_kinds.pdf_files += 1,
            SourceKind::Image => stats.content_kinds.image_files += 1,
            SourceKind::Text => stats.content_kinds.text_files += 1,
            SourceKind::Other => stats.content_kinds.other_files += 1,
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

fn sql_string(value: &str) -> String {
    format!("'{}'", value.replace('\'', "''"))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SourceKind {
    Text,
    Pdf,
    Image,
    Other,
}

fn classify_source_kind(source: &str) -> SourceKind {
    let ext = Path::new(source)
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("");

    if ext.eq_ignore_ascii_case("pdf") {
        SourceKind::Pdf
    } else if ["png", "jpg", "jpeg", "webp"]
        .iter()
        .any(|candidate| ext.eq_ignore_ascii_case(candidate))
    {
        SourceKind::Image
    } else if ["md", "markdown", "txt", "rst"]
        .iter()
        .any(|candidate| ext.eq_ignore_ascii_case(candidate))
    {
        SourceKind::Text
    } else {
        SourceKind::Other
    }
}

fn estimate_token_count(text: &str) -> usize {
    let chars = text.chars().count();
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
            page: 0,
            chunk_index: 0,
            metadata: "{}".to_string(),
            embedding: vec![value, value],
        }
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

    #[test]
    fn test_strip_thinking() {
        let input = "<think>hidden</think>\nFinal answer.";
        assert_eq!(strip_thinking(input).trim(), "Final answer.");
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
}
