//! File ingestion and chunking for the local RAG store.

use crate::models::{Embedder, VisionCaptioner};
use crate::source_kind::SourceKind;
use crate::store::ChunkRow;
use anyhow::{Context, Result};
use globset::{Glob, GlobSet, GlobSetBuilder};
use indicatif::{ProgressBar, ProgressStyle};
use pdf_extract::extract_text_by_pages;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use walkdir::{DirEntry, WalkDir};

/// Aggregated indexing counters for a single ingest run.
pub struct IndexStats {
    /// Number of files indexed successfully.
    pub indexed_files: usize,
    /// Number of files skipped due to errors or unsupported processing.
    pub skipped_files: usize,
    /// Total number of chunks written.
    pub total_chunks: usize,
    /// Human-readable errors encountered during indexing.
    pub errors: Vec<String>,
}

/// Result of indexing a path into chunk rows.
pub struct IngestResult {
    /// Chunk rows ready to write into the store.
    pub rows: Vec<ChunkRow>,
    /// Unique source paths that were seen during ingestion.
    pub source_paths: Vec<String>,
    /// Embedding dimension inferred from the first embedded chunk, if any.
    pub embedding_dim: Option<usize>,
    /// Aggregate indexing statistics.
    pub stats: IndexStats,
}

#[derive(Debug, Clone)]
struct ChunkContent {
    text: String,
    metadata: Value,
}

#[derive(Debug, Clone)]
struct ExtractedUnit {
    page_num: i32,
    chunks: Vec<ChunkContent>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PdfParser {
    Native,
    Liteparse,
}

#[derive(Debug)]
struct IngestOptions {
    excludes: GlobSet,
    include_hidden: bool,
}

/// Walks a file or directory, extracts supported content, and returns chunk rows.
pub async fn ingest_path(
    path: &Path,
    chunk_size: usize,
    overlap: usize,
    embedder: &Embedder,
    vision: Option<&VisionCaptioner>,
    pdf_parser: PdfParser,
    exclude_patterns: &[String],
    include_hidden: bool,
    existing_fingerprints: &BTreeMap<String, String>,
    force: bool,
) -> Result<IngestResult> {
    let options = build_ingest_options(exclude_patterns, include_hidden)?;
    let files = collect_files(path, &options)?;
    let progress = build_progress_bar(files.len() as u64);
    let mut rows = Vec::new();
    let mut source_paths = BTreeSet::new();
    let mut dim = None;
    let mut stats = IndexStats {
        indexed_files: 0,
        skipped_files: 0,
        total_chunks: 0,
        errors: Vec::new(),
    };

    for file in files {
        progress.set_message(format!("Reading {}", display_name(&file)));
        let kind = SourceKind::from_path(&file);
        if kind == SourceKind::Unsupported {
            progress.inc(1);
            continue;
        }

        let source_path = file.display().to_string();
        let source_fingerprint = match fingerprint_path(&file) {
            Ok(fingerprint) => fingerprint,
            Err(err) => {
                stats.skipped_files += 1;
                stats.errors.push(format!("{}: {}", file.display(), err));
                progress.inc(1);
                continue;
            }
        };

        if !force
            && existing_fingerprints
                .get(&source_path)
                .is_some_and(|fingerprint| fingerprint == &source_fingerprint)
        {
            stats.skipped_files += 1;
            progress.set_message(format!("Skipping unchanged {}", display_name(&file)));
            progress.inc(1);
            continue;
        }

        source_paths.insert(source_path);

        match extract_units(
            &file, kind, chunk_size, overlap, vision, pdf_parser, &progress,
        )
        .await
        {
            Ok(units) => match embed_units(
                &file,
                &source_fingerprint,
                units,
                embedder,
                &mut dim,
                &progress,
            )
            .await
            {
                Ok(mut file_rows) => {
                    stats.indexed_files += 1;
                    stats.total_chunks += file_rows.len();
                    rows.append(&mut file_rows);
                }
                Err(err) => {
                    stats.skipped_files += 1;
                    stats.errors.push(format!("{}: {}", file.display(), err));
                }
            },
            Err(err) => {
                stats.skipped_files += 1;
                stats.errors.push(format!("{}: {}", file.display(), err));
            }
        }

        progress.inc(1);
    }

    progress.finish_with_message(format!(
        "Indexed {} file(s), wrote {} chunk(s)",
        stats.indexed_files, stats.total_chunks
    ));

    Ok(IngestResult {
        rows,
        source_paths: source_paths.into_iter().collect(),
        embedding_dim: dim,
        stats,
    })
}

async fn extract_units(
    path: &Path,
    kind: SourceKind,
    chunk_size: usize,
    overlap: usize,
    vision: Option<&VisionCaptioner>,
    pdf_parser: PdfParser,
    progress: &ProgressBar,
) -> Result<Vec<ExtractedUnit>> {
    match kind {
        SourceKind::Text | SourceKind::Markdown => {
            let text = load_text_file(path)?;
            let chunks = if kind == SourceKind::Text {
                chunk_plain_text(&text, chunk_size, overlap)
            } else {
                chunk_markdown(&text, chunk_size, overlap)
            };
            Ok(vec![ExtractedUnit {
                page_num: 0,
                chunks,
            }])
        }
        SourceKind::Html => {
            let text = load_html_file(path)?;
            Ok(vec![ExtractedUnit {
                page_num: 0,
                chunks: chunk_document_text(
                    &text,
                    kind.format_label().expect("html has format label"),
                    None,
                    chunk_size,
                    overlap,
                ),
            }])
        }
        SourceKind::Csv { delimiter } => {
            let text = load_delimited_file(path, delimiter)?;
            Ok(vec![ExtractedUnit {
                page_num: 0,
                chunks: chunk_document_text(
                    &text,
                    kind.format_label().expect("csv has format label"),
                    None,
                    chunk_size,
                    overlap,
                ),
            }])
        }
        SourceKind::Code { language } => {
            let text = load_text_file(path)?;
            Ok(vec![ExtractedUnit {
                page_num: 0,
                chunks: chunk_document_text(
                    &text,
                    kind.format_label().expect("code has format label"),
                    Some(language),
                    chunk_size,
                    overlap,
                ),
            }])
        }
        SourceKind::Pdf => {
            progress.set_message(format!("Extracting {}", display_name(path)));
            let pages = extract_pdf_pages(path, pdf_parser)?;
            Ok(pages
                .into_iter()
                .enumerate()
                .map(|(page_idx, page_text)| {
                    let page_num = (page_idx + 1) as i32;
                    ExtractedUnit {
                        page_num,
                        chunks: chunk_pdf_page(&page_text, page_num, chunk_size, overlap),
                    }
                })
                .collect())
        }
        SourceKind::Image => {
            let vision = vision.context("image skipped because no vision model is configured")?;
            progress.set_message(format!("Captioning {}", display_name(path)));
            let caption = vision.caption_image(path).await?;
            Ok(vec![ExtractedUnit {
                page_num: 0,
                chunks: vec![ChunkContent {
                    text: build_image_retrieval_text(path, &caption),
                    metadata: json!({
                        "format": SourceKind::Image
                            .format_label()
                            .expect("image has format label"),
                    }),
                }],
            }])
        }
        SourceKind::Unsupported => Ok(Vec::new()),
    }
}

async fn embed_units(
    path: &Path,
    source_fingerprint: &str,
    units: Vec<ExtractedUnit>,
    embedder: &Embedder,
    dim: &mut Option<usize>,
    progress: &ProgressBar,
) -> Result<Vec<ChunkRow>> {
    let mut rows = Vec::new();
    for unit in units {
        if unit.page_num > 0 {
            progress.set_message(format!(
                "Embedding {} (page {})",
                display_name(path),
                unit.page_num
            ));
        } else {
            progress.set_message(format!("Embedding {}", display_name(path)));
        }
        let mut unit_rows = embed_chunks(
            path,
            source_fingerprint,
            unit.page_num,
            unit.chunks,
            embedder,
            dim,
        )
        .await?;
        rows.append(&mut unit_rows);
    }
    Ok(rows)
}

async fn embed_chunks(
    path: &Path,
    source_fingerprint: &str,
    page_num: i32,
    chunks: Vec<ChunkContent>,
    embedder: &Embedder,
    dim: &mut Option<usize>,
) -> Result<Vec<ChunkRow>> {
    let mut rows = Vec::new();
    for (idx, chunk) in chunks.into_iter().enumerate() {
        let embedding = embedder.embed(&chunk.text).await?;
        if dim.is_none() {
            *dim = Some(embedding.len());
        }
        let chunk_hash = hash_text(path, &chunk.text, page_num, idx as i32);
        let mut metadata = chunk.metadata;
        if let Some(obj) = metadata.as_object_mut() {
            obj.insert("source".to_string(), json!(path.display().to_string()));
            obj.insert("page".to_string(), json!(page_num));
            obj.insert("source_fingerprint".to_string(), json!(source_fingerprint));
        }
        let format = metadata
            .get("format")
            .and_then(Value::as_str)
            .context("chunk metadata missing format")?
            .to_string();
        rows.push(ChunkRow {
            id: chunk_hash.clone(),
            source_path: path.display().to_string(),
            chunk_text: chunk.text,
            chunk_hash,
            format,
            page: page_num,
            chunk_index: idx as i32,
            metadata: metadata.to_string(),
            embedding,
        });
    }
    Ok(rows)
}

fn build_ingest_options(
    exclude_patterns: &[String],
    include_hidden: bool,
) -> Result<IngestOptions> {
    let mut builder = GlobSetBuilder::new();
    for pattern in exclude_patterns {
        builder
            .add(Glob::new(pattern).with_context(|| format!("invalid exclude glob: {pattern}"))?);
    }

    Ok(IngestOptions {
        excludes: builder.build().context("build exclude glob set")?,
        include_hidden,
    })
}

fn collect_files(path: &Path, options: &IngestOptions) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    if path.is_file() {
        if should_index_path(path, options) {
            files.push(path.to_path_buf());
        }
        return Ok(files);
    }

    for entry in WalkDir::new(path)
        .into_iter()
        .filter_entry(|entry| should_traverse_entry(entry, options))
    {
        let entry = entry?;
        if entry.file_type().is_file() && should_index_path(entry.path(), options) {
            files.push(entry.path().to_path_buf());
        }
    }
    Ok(files)
}

fn should_traverse_entry(entry: &DirEntry, options: &IngestOptions) -> bool {
    if entry.depth() == 0 {
        return true;
    }
    if options.excludes.is_match(entry.path()) {
        return false;
    }

    options.include_hidden || !is_hidden_path(entry.path())
}

fn should_index_path(path: &Path, options: &IngestOptions) -> bool {
    !options.excludes.is_match(path) && (options.include_hidden || !is_hidden_path(path))
}

fn is_hidden_path(path: &Path) -> bool {
    path.file_name()
        .map(|name| name.to_string_lossy().starts_with('.'))
        .unwrap_or(false)
}
fn fingerprint_path(path: &Path) -> Result<String> {
    let metadata =
        fs::metadata(path).with_context(|| format!("read file metadata: {}", path.display()))?;
    let modified = metadata
        .modified()
        .with_context(|| format!("read file modified time: {}", path.display()))?;
    let modified = modified
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    Ok(format!("{}-{modified}", metadata.len()))
}

fn load_text_file(path: &Path) -> Result<String> {
    let bytes = fs::read(path)?;
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

fn load_html_file(path: &Path) -> Result<String> {
    let html = load_text_file(path)?;
    let rendered =
        html2text::from_read(html.as_bytes(), usize::MAX).context("render html as text")?;
    Ok(rendered.trim().to_string())
}

fn load_delimited_file(path: &Path, delimiter: u8) -> Result<String> {
    let mut reader = csv::ReaderBuilder::new()
        .delimiter(delimiter)
        .flexible(true)
        .from_path(path)
        .with_context(|| format!("open delimited file: {}", path.display()))?;

    let headers = reader
        .headers()
        .with_context(|| format!("read headers from {}", path.display()))?
        .iter()
        .map(|header| header.trim().to_string())
        .collect::<Vec<_>>();

    let mut lines = Vec::new();
    if !headers.is_empty() {
        lines.push(format!("Columns: {}", headers.join(", ")));
    }

    for record in reader.records() {
        let record = record.with_context(|| format!("read row from {}", path.display()))?;
        let cells = record
            .iter()
            .enumerate()
            .map(|(idx, value)| {
                let label = headers
                    .get(idx)
                    .map(String::as_str)
                    .filter(|header| !header.is_empty());
                match label {
                    Some(header) => format!("{}: {}", header, value.trim()),
                    None => value.trim().to_string(),
                }
            })
            .collect::<Vec<_>>();
        lines.push(cells.join(" | "));
    }

    Ok(lines.join("\n"))
}

fn extract_pdf_pages(path: &Path, parser: PdfParser) -> Result<Vec<String>> {
    match parser {
        PdfParser::Native => extract_text_by_pages(path)
            .with_context(|| format!("extract pdf text: {}", path.display())),
        PdfParser::Liteparse => extract_pdf_pages_with_liteparse(path),
    }
}

fn extract_pdf_pages_with_liteparse(path: &Path) -> Result<Vec<String>> {
    let output = Command::new("lit")
        .arg("parse")
        .arg(path)
        .arg("--format")
        .arg("json")
        .arg("--quiet")
        .output()
        .with_context(|| format!("run liteparse CLI (`lit parse`) for {}", path.display()))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("liteparse failed for {}: {}", path.display(), stderr.trim());
    }

    parse_liteparse_pages(&output.stdout)
        .with_context(|| format!("parse liteparse output for {}", path.display()))
}

fn parse_liteparse_pages(bytes: &[u8]) -> Result<Vec<String>> {
    let value: Value = serde_json::from_slice(bytes).context("decode liteparse JSON")?;
    if let Some(pages) = value.get("pages").and_then(Value::as_array) {
        let extracted = pages
            .iter()
            .filter_map(|page| page.get("text").and_then(Value::as_str))
            .map(str::trim)
            .filter(|text| !text.is_empty())
            .map(str::to_owned)
            .collect::<Vec<_>>();
        if !extracted.is_empty() {
            return Ok(extracted);
        }
    }

    if let Some(text) = value.get("text").and_then(Value::as_str) {
        let pages = split_liteparse_text_pages(text);
        if !pages.is_empty() {
            return Ok(pages);
        }
    }

    anyhow::bail!("liteparse output did not contain page text")
}

fn split_liteparse_text_pages(text: &str) -> Vec<String> {
    text.split('\u{c}')
        .map(str::trim)
        .filter(|page| !page.is_empty())
        .map(str::to_owned)
        .collect()
}

fn build_progress_bar(total_files: u64) -> ProgressBar {
    let bar = ProgressBar::new(total_files.max(1));
    let style = ProgressStyle::with_template(
        "{spinner:.green} [{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}",
    )
    .expect("valid progress template")
    .progress_chars("=>-");
    bar.set_style(style);
    bar
}

fn display_name(path: &Path) -> String {
    path.file_name()
        .and_then(|name| name.to_str())
        .map(str::to_owned)
        .unwrap_or_else(|| path.to_string_lossy().into_owned())
}

fn build_image_retrieval_text(path: &Path, caption: &str) -> String {
    format!("File: {}\nCaption: {}", display_name(path), caption.trim())
}

fn chunk_document_text(
    text: &str,
    format: &str,
    language: Option<&str>,
    chunk_size: usize,
    overlap: usize,
) -> Vec<ChunkContent> {
    chunk_text(text, chunk_size, overlap)
        .into_iter()
        .map(|text| {
            let mut metadata = json!({
                "format": format,
            });
            if let Some(obj) = metadata.as_object_mut() {
                if let Some(language) = language {
                    obj.insert("language".to_string(), json!(language));
                }
            }
            ChunkContent { text, metadata }
        })
        .collect()
}

fn chunk_plain_text(text: &str, chunk_size: usize, overlap: usize) -> Vec<ChunkContent> {
    chunk_document_text(text, "text", None, chunk_size, overlap)
}

fn chunk_markdown(text: &str, chunk_size: usize, overlap: usize) -> Vec<ChunkContent> {
    let mut sections = Vec::new();
    let mut heading_stack: Vec<String> = Vec::new();
    let mut buffer = Vec::new();

    for line in text.lines() {
        if let Some((level, heading)) = parse_markdown_heading(line) {
            push_markdown_section(&mut sections, &heading_stack, &buffer, chunk_size, overlap);
            buffer.clear();
            heading_stack.truncate(level.saturating_sub(1));
            heading_stack.push(heading);
            continue;
        }
        buffer.push(line.to_string());
    }

    push_markdown_section(&mut sections, &heading_stack, &buffer, chunk_size, overlap);
    sections
}

fn chunk_pdf_page(
    text: &str,
    page_num: i32,
    chunk_size: usize,
    overlap: usize,
) -> Vec<ChunkContent> {
    let paragraphs = split_blocks(text);
    if paragraphs.is_empty() {
        return Vec::new();
    }

    let mut sections: Vec<(Option<String>, Vec<String>)> = Vec::new();
    let mut current_heading: Option<String> = None;
    let mut current_blocks = Vec::new();

    for block in paragraphs {
        if looks_like_pdf_heading(&block) {
            if !current_blocks.is_empty() {
                sections.push((current_heading.clone(), current_blocks));
                current_blocks = Vec::new();
            }
            current_heading = Some(block.trim().to_string());
            continue;
        }
        current_blocks.push(block);
    }

    if !current_blocks.is_empty() || current_heading.is_some() {
        sections.push((current_heading, current_blocks));
    }

    let mut chunks = Vec::new();
    for (heading, blocks) in sections {
        let prefix = build_pdf_prefix(page_num, heading.as_deref());
        for text in chunk_blocks(&blocks, Some(prefix.as_str()), chunk_size, overlap) {
            chunks.push(ChunkContent {
                text,
                metadata: json!({
                    "format": SourceKind::Pdf.format_label().expect("pdf has format label"),
                    "section": heading,
                }),
            });
        }
    }
    chunks
}

/// Splits plain text into overlapping character-based chunks.
pub fn chunk_text(text: &str, chunk_size: usize, overlap: usize) -> Vec<String> {
    if text.trim().is_empty() {
        return Vec::new();
    }

    let chars: Vec<char> = text.chars().collect();
    let mut out = Vec::new();
    let mut start = 0usize;

    while start < chars.len() {
        let end = (start + chunk_size).min(chars.len());
        let chunk: String = chars[start..end].iter().collect();
        let trimmed = chunk.trim();
        if !trimmed.is_empty() {
            out.push(trimmed.to_string());
        }
        if end >= chars.len() {
            break;
        }
        start = end.saturating_sub(overlap);
        if start == end {
            break;
        }
    }

    out
}

fn push_markdown_section(
    sections: &mut Vec<ChunkContent>,
    heading_stack: &[String],
    buffer: &[String],
    chunk_size: usize,
    overlap: usize,
) {
    let body = buffer.join("\n");
    let body_blocks = split_blocks(&body);
    if body_blocks.is_empty() && heading_stack.is_empty() {
        return;
    }

    let heading_path = if heading_stack.is_empty() {
        None
    } else {
        Some(heading_stack.join(" > "))
    };
    let prefix = heading_path
        .as_ref()
        .map(|path| format!("Headings: {}", path));

    if body_blocks.is_empty() {
        if let Some(prefix) = prefix {
            sections.push(ChunkContent {
                text: prefix,
                metadata: json!({
                    "format": SourceKind::Markdown
                        .format_label()
                        .expect("markdown has format label"),
                    "headings": heading_stack,
                }),
            });
        }
        return;
    }

    for text in chunk_blocks(&body_blocks, prefix.as_deref(), chunk_size, overlap) {
        sections.push(ChunkContent {
            text,
            metadata: json!({
                "format": SourceKind::Markdown
                    .format_label()
                    .expect("markdown has format label"),
                "headings": heading_stack,
            }),
        });
    }
}

fn parse_markdown_heading(line: &str) -> Option<(usize, String)> {
    let trimmed = line.trim_start();
    let mut level = 0usize;
    for ch in trimmed.chars() {
        if ch == '#' {
            level += 1;
        } else {
            break;
        }
    }
    if level == 0 || level > 6 {
        return None;
    }

    let rest = trimmed[level..].trim();
    if rest.is_empty() {
        return None;
    }

    Some((level, rest.to_string()))
}

fn split_blocks(text: &str) -> Vec<String> {
    let mut blocks = Vec::new();
    let mut current = Vec::new();

    for line in text.lines() {
        if line.trim().is_empty() {
            if !current.is_empty() {
                blocks.push(current.join("\n").trim().to_string());
                current.clear();
            }
            continue;
        }
        current.push(line.to_string());
    }

    if !current.is_empty() {
        blocks.push(current.join("\n").trim().to_string());
    }

    blocks
        .into_iter()
        .filter(|block| !block.is_empty())
        .collect()
}

fn chunk_blocks(
    blocks: &[String],
    prefix: Option<&str>,
    chunk_size: usize,
    overlap: usize,
) -> Vec<String> {
    if blocks.is_empty() {
        return Vec::new();
    }

    let prefix = prefix.map(str::trim).filter(|value| !value.is_empty());
    let prefix_len = prefix.map(char_len).unwrap_or(0);
    let separator_len = usize::from(prefix.is_some()) * 2;
    let body_limit = chunk_size.saturating_sub(prefix_len + separator_len).max(1);

    let mut normalized = Vec::new();
    for block in blocks {
        if char_len(block) <= body_limit {
            normalized.push(block.trim().to_string());
            continue;
        }
        normalized.extend(chunk_text(block, body_limit, overlap));
    }

    let mut out = Vec::new();
    let mut current = Vec::new();
    let mut current_len = 0usize;

    for block in normalized {
        let block_len = char_len(&block);
        let next_len = if current.is_empty() {
            block_len
        } else {
            current_len + 2 + block_len
        };

        if !current.is_empty() && next_len > body_limit {
            out.push(render_chunk(prefix, &current));
            current = overlap_blocks(&current, overlap);
            current_len = current_chunk_len(&current);
        }

        if current.is_empty() {
            current_len = block_len;
            current.push(block);
        } else {
            current_len += 2 + block_len;
            current.push(block);
        }
    }

    if !current.is_empty() {
        out.push(render_chunk(prefix, &current));
    }

    out
}

fn render_chunk(prefix: Option<&str>, blocks: &[String]) -> String {
    let body = blocks.join("\n\n");
    match prefix {
        Some(prefix) if !body.is_empty() => format!("{}\n\n{}", prefix, body),
        Some(prefix) => prefix.to_string(),
        None => body,
    }
}

fn overlap_blocks(blocks: &[String], overlap: usize) -> Vec<String> {
    if overlap == 0 || blocks.is_empty() {
        return Vec::new();
    }

    let mut kept = Vec::new();
    let mut total = 0usize;
    for block in blocks.iter().rev() {
        let block_len = char_len(block);
        let next_total = if kept.is_empty() {
            block_len
        } else {
            total + 2 + block_len
        };
        if next_total > overlap && !kept.is_empty() {
            break;
        }
        total = next_total;
        kept.push(block.clone());
        if total >= overlap {
            break;
        }
    }
    kept.reverse();
    kept
}

fn current_chunk_len(blocks: &[String]) -> usize {
    blocks
        .iter()
        .enumerate()
        .map(|(idx, block)| char_len(block) + usize::from(idx > 0) * 2)
        .sum()
}

fn build_pdf_prefix(page_num: i32, heading: Option<&str>) -> String {
    match heading {
        Some(heading) => format!("Page: {}\nSection: {}", page_num, heading.trim()),
        None => format!("Page: {}", page_num),
    }
}

fn looks_like_pdf_heading(block: &str) -> bool {
    let trimmed = block.trim();
    if trimmed.is_empty() || trimmed.contains('\n') {
        return false;
    }

    let len = char_len(trimmed);
    if len > 80 {
        return false;
    }

    if trimmed.ends_with('.') || trimmed.ends_with(',') || trimmed.ends_with(';') {
        return false;
    }

    if starts_with_section_marker(trimmed) {
        return true;
    }

    let letters: Vec<char> = trimmed.chars().filter(|ch| ch.is_alphabetic()).collect();
    if letters.is_empty() {
        return false;
    }

    let uppercase = letters.iter().filter(|ch| ch.is_uppercase()).count();
    let uppercase_ratio = uppercase as f32 / letters.len() as f32;
    uppercase_ratio > 0.7 || trimmed == trimmed.to_uppercase() || is_title_like(trimmed)
}

fn starts_with_section_marker(text: &str) -> bool {
    let Some(first) = text.split_whitespace().next() else {
        return false;
    };
    let marker = first.trim_end_matches(['.', ')', ':']);
    if marker.is_empty() {
        return false;
    }
    marker.chars().all(|ch| ch.is_ascii_digit() || ch == '.')
}

fn is_title_like(text: &str) -> bool {
    let words: Vec<&str> = text
        .split_whitespace()
        .map(|word| word.trim_matches(|ch: char| !ch.is_alphanumeric()))
        .filter(|word| !word.is_empty())
        .collect();
    if words.is_empty() || words.len() > 10 {
        return false;
    }

    let titled = words
        .iter()
        .filter(|word| {
            let mut chars = word.chars();
            matches!(chars.next(), Some(ch) if ch.is_uppercase())
        })
        .count();
    titled * 2 >= words.len()
}

fn char_len(text: &str) -> usize {
    text.chars().count()
}

fn hash_text(path: &Path, text: &str, page: i32, idx: i32) -> String {
    let mut hasher = Sha256::new();
    hasher.update(path.display().to_string().as_bytes());
    hasher.update(text.as_bytes());
    hasher.update(page.to_le_bytes());
    hasher.update(idx.to_le_bytes());
    to_hex(&hasher.finalize())
}

fn to_hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push_str(&format!("{:02x}", b));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::thread;

    fn one_shot_json_server(status_line: &str, body: &'static str) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind test server");
        let addr = listener.local_addr().unwrap();
        let status_line = status_line.to_string();

        thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept request");
            let mut buf = [0_u8; 4096];
            let _ = stream.read(&mut buf);
            let response = format!(
                "HTTP/1.1 {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                status_line,
                body.len(),
                body
            );
            stream
                .write_all(response.as_bytes())
                .expect("write response");
        });

        format!("http://{}", addr)
    }

    #[test]
    fn test_chunk_text_basic() {
        let chunks = chunk_text("abcdefghij", 5, 2);
        assert_eq!(chunks, vec!["abcde", "defgh", "ghij"]);
    }

    #[test]
    fn test_chunk_markdown_preserves_heading_path() {
        let text = "# Guide\n\nIntro text.\n\n## Install\n\nStep one.\n\nStep two.";

        let chunks = chunk_markdown(text, 80, 20);

        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].text, "Headings: Guide\n\nIntro text.");
        assert_eq!(
            chunks[1].text,
            "Headings: Guide > Install\n\nStep one.\n\nStep two."
        );
    }

    #[test]
    fn test_chunk_markdown_splits_large_section_on_block_boundaries() {
        let text = "# Guide\n\nParagraph one.\n\nParagraph two.\n\nParagraph three.";

        let chunks = chunk_markdown(text, 40, 12);

        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].text, "Headings: Guide\n\nParagraph one.");
        assert_eq!(
            chunks[1].text,
            "Headings: Guide\n\nParagraph one.\n\nParagraph two."
        );
        assert_eq!(
            chunks[2].text,
            "Headings: Guide\n\nParagraph two.\n\nParagraph three."
        );
    }

    #[test]
    fn test_chunk_pdf_page_preserves_page_and_section_context() {
        let text =
            "INTRODUCTION\n\nFirst paragraph.\n\nSecond paragraph.\n\nMETHODS\n\nMethod details.";

        let chunks = chunk_pdf_page(text, 3, 80, 20);

        assert_eq!(chunks.len(), 2);
        assert_eq!(
            chunks[0].text,
            "Page: 3\nSection: INTRODUCTION\n\nFirst paragraph.\n\nSecond paragraph."
        );
        assert_eq!(
            chunks[1].text,
            "Page: 3\nSection: METHODS\n\nMethod details."
        );
    }

    #[test]
    fn test_load_text_file_is_lossy() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("note.txt");
        fs::write(&path, vec![0x66, 0x6f, 0x80, 0x6f]).unwrap();

        let text = load_text_file(&path).unwrap();
        assert!(text.contains('f'));
        assert!(text.contains('\u{fffd}'));
    }

    #[test]
    fn test_parse_liteparse_pages_from_pages_array() {
        let json = br#"{
            "pages": [
                {"page": 1, "text": "First page"},
                {"page": 2, "text": "Second page"}
            ]
        }"#;

        let pages = parse_liteparse_pages(json).unwrap();
        assert_eq!(pages, vec!["First page", "Second page"]);
    }

    #[test]
    fn test_parse_liteparse_pages_from_form_feed_text() {
        let json = br#"{
            "text": "First page\fSecond page"
        }"#;

        let pages = parse_liteparse_pages(json).unwrap();
        assert_eq!(pages, vec!["First page", "Second page"]);
    }

    #[test]
    fn test_parse_liteparse_pages_errors_without_page_text() {
        let json = br#"{"pages": [{"page": 1, "text": "   "}], "text": "   "}"#;
        let err = parse_liteparse_pages(json).unwrap_err().to_string();
        assert!(err.contains("did not contain page text"));
    }

    #[test]
    fn test_collect_files_handles_single_file_and_directory() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("a.txt");
        let nested_dir = dir.path().join("nested");
        let nested = nested_dir.join("b.md");
        fs::create_dir_all(&nested_dir).unwrap();
        fs::write(&file, "a").unwrap();
        fs::write(&nested, "b").unwrap();

        let options = build_ingest_options(&[], true).unwrap();
        assert_eq!(collect_files(&file, &options).unwrap(), vec![file.clone()]);

        let mut files = collect_files(dir.path(), &options).unwrap();
        files.sort();
        assert_eq!(files, vec![file, nested]);
    }

    #[test]
    fn test_collect_files_skips_hidden_and_excluded_paths() {
        let dir = tempfile::tempdir().unwrap();
        let visible = dir.path().join("visible.txt");
        let hidden_dir = dir.path().join(".git");
        let hidden_file = hidden_dir.join("config");
        let excluded_dir = dir.path().join("target");
        let excluded_file = excluded_dir.join("out.txt");
        fs::create_dir_all(&hidden_dir).unwrap();
        fs::create_dir_all(&excluded_dir).unwrap();
        fs::write(&visible, "ok").unwrap();
        fs::write(&hidden_file, "hidden").unwrap();
        fs::write(&excluded_file, "skip").unwrap();

        let options = build_ingest_options(&["**/target/**".to_string()], false).unwrap();
        let files = collect_files(dir.path(), &options).unwrap();

        assert_eq!(files, vec![visible]);
    }

    #[test]
    fn test_display_name_and_image_retrieval_text() {
        let path = Path::new("/tmp/example-image.png");
        assert_eq!(display_name(path), "example-image.png");
        assert_eq!(
            build_image_retrieval_text(path, "  a cat on a mat  "),
            "File: example-image.png\nCaption: a cat on a mat"
        );
    }

    #[test]
    fn test_parse_markdown_heading_and_split_blocks() {
        assert_eq!(
            parse_markdown_heading("### Title"),
            Some((3, "Title".to_string()))
        );
        assert_eq!(parse_markdown_heading("####### nope"), None);
        assert_eq!(parse_markdown_heading("#   "), None);

        assert_eq!(
            split_blocks("one\n\n two\nthree \n\n\n four"),
            vec!["one", "two\nthree", "four"]
        );
    }

    #[test]
    fn test_chunk_blocks_with_prefix_and_overlap() {
        let blocks = vec!["Alpha".to_string(), "Beta".to_string(), "Gamma".to_string()];

        let chunks = chunk_blocks(&blocks, Some("Headings: Guide"), 28, 6);

        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0], "Headings: Guide\n\nAlpha\n\nBeta");
        assert_eq!(chunks[1], "Headings: Guide\n\nBeta\n\nGamma");
    }

    #[test]
    fn test_chunk_plain_text_adds_text_metadata() {
        let chunks = chunk_plain_text("abcdef", 3, 1);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].text, "abc");
        assert_eq!(chunks[0].metadata["format"], "text");
    }

    #[test]
    fn test_chunk_document_text_adds_language_metadata() {
        let chunks = chunk_document_text("fn main() {}", "code", Some("rust"), 100, 0);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].metadata["format"], "code");
        assert_eq!(chunks[0].metadata["language"], "rust");
    }

    #[test]
    fn test_load_html_and_delimited_file() {
        let dir = tempfile::tempdir().unwrap();
        let html = dir.path().join("index.html");
        let csv = dir.path().join("people.csv");
        fs::write(
            &html,
            "<html><body><h1>Title</h1><p>Hello <b>world</b>.</p></body></html>",
        )
        .unwrap();
        fs::write(&csv, "name,role\nMina,Engineer\nKai,Designer\n").unwrap();

        let html_text = load_html_file(&html).unwrap();
        assert!(html_text.contains("Title"));
        assert!(html_text.contains("Hello world."));

        let csv_text = load_delimited_file(&csv, b',').unwrap();
        assert!(csv_text.contains("Columns: name, role"));
        assert!(csv_text.contains("name: Mina | role: Engineer"));
    }

    #[test]
    fn test_pdf_heading_helpers_and_prefix() {
        assert!(looks_like_pdf_heading("INTRODUCTION"));
        assert!(looks_like_pdf_heading("1.2 Background"));
        assert!(!looks_like_pdf_heading("This is a sentence."));
        assert!(starts_with_section_marker("2.1 Methods"));
        assert!(!starts_with_section_marker("Methods"));
        assert!(is_title_like("Project Overview"));
        assert!(!is_title_like(
            "a very long phrase with too many lowercase words perhaps"
        ));
        assert_eq!(
            build_pdf_prefix(4, Some("Methods")),
            "Page: 4\nSection: Methods"
        );
        assert_eq!(build_pdf_prefix(4, None), "Page: 4");
    }

    #[test]
    fn test_overlap_blocks_char_len_hash_and_hex() {
        let kept = overlap_blocks(
            &["one".to_string(), "two".to_string(), "three".to_string()],
            6,
        );
        assert_eq!(kept, vec!["three"]);
        assert_eq!(char_len("hé🙂"), 3);

        let path = Path::new("notes.txt");
        let first = hash_text(path, "hello", 1, 0);
        let second = hash_text(path, "hello", 1, 0);
        let different = hash_text(path, "hello", 2, 0);
        assert_eq!(first, second);
        assert_ne!(first, different);
        assert_eq!(to_hex(&[0x0f, 0xa0]), "0fa0");
    }

    #[test]
    fn test_fingerprint_path_changes_when_file_changes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("note.txt");
        fs::write(&path, "hello").unwrap();
        let first = fingerprint_path(&path).unwrap();

        std::thread::sleep(std::time::Duration::from_millis(2));
        fs::write(&path, "hello world").unwrap();
        let second = fingerprint_path(&path).unwrap();

        assert_ne!(first, second);
    }

    #[tokio::test]
    async fn test_ingest_path_skips_image_without_vision_model() {
        let dir = tempfile::tempdir().unwrap();
        let image = dir.path().join("photo.png");
        fs::write(&image, b"image-bytes").unwrap();

        let embedder = Embedder::new("http://unused".to_string(), "embed".to_string());
        let result = ingest_path(
            dir.path(),
            100,
            0,
            &embedder,
            None,
            PdfParser::Native,
            &[],
            false,
            &BTreeMap::new(),
            false,
        )
        .await
        .unwrap();

        assert!(result.rows.is_empty());
        assert_eq!(result.stats.indexed_files, 0);
        assert_eq!(result.stats.skipped_files, 1);
        assert!(result.stats.errors[0].contains("no vision model is configured"));
        assert_eq!(result.source_paths, vec![image.display().to_string()]);
    }

    #[tokio::test]
    async fn test_ingest_path_reports_pdf_extraction_error() {
        let dir = tempfile::tempdir().unwrap();
        let pdf = dir.path().join("broken.pdf");
        fs::write(&pdf, b"not a real pdf").unwrap();

        let embedder = Embedder::new("http://unused".to_string(), "embed".to_string());
        let result = ingest_path(
            &pdf,
            100,
            0,
            &embedder,
            None,
            PdfParser::Native,
            &[],
            false,
            &BTreeMap::new(),
            false,
        )
        .await
        .unwrap();

        assert!(result.rows.is_empty());
        assert_eq!(result.stats.indexed_files, 0);
        assert_eq!(result.stats.skipped_files, 1);
        assert!(result.stats.errors[0].contains("broken.pdf"));
    }

    #[tokio::test]
    async fn test_ingest_path_indexes_html_file() {
        let dir = tempfile::tempdir().unwrap();
        let html = dir.path().join("index.html");
        fs::write(
            &html,
            "<html><body><h1>Guide</h1><p>Hello world.</p></body></html>",
        )
        .unwrap();

        let base_url = one_shot_json_server("200 OK", r#"{"embeddings":[[0.1,0.2]]}"#);
        let embedder = Embedder::new(base_url, "embed".to_string());
        let result = ingest_path(
            &html,
            100,
            0,
            &embedder,
            None,
            PdfParser::Native,
            &[],
            false,
            &BTreeMap::new(),
            false,
        )
        .await
        .unwrap();

        assert_eq!(result.stats.indexed_files, 1);
        assert_eq!(result.stats.skipped_files, 0);
        assert_eq!(result.rows.len(), 1);
        assert_eq!(result.rows[0].page, 0);
        assert!(result.rows[0].chunk_text.contains("Guide"));
        assert!(result.rows[0].chunk_text.contains("Hello world."));
        assert!(result.rows[0].metadata.contains("\"format\":\"html\""));
    }

    #[tokio::test]
    async fn test_ingest_path_skips_unchanged_files_without_embedding() {
        let dir = tempfile::tempdir().unwrap();
        let text = dir.path().join("note.txt");
        fs::write(&text, "hello world").unwrap();

        let mut existing_fingerprints = BTreeMap::new();
        existing_fingerprints.insert(text.display().to_string(), fingerprint_path(&text).unwrap());

        let embedder = Embedder::new("http://127.0.0.1:9".to_string(), "embed".to_string());
        let result = ingest_path(
            &text,
            100,
            0,
            &embedder,
            None,
            PdfParser::Native,
            &[],
            false,
            &existing_fingerprints,
            false,
        )
        .await
        .unwrap();

        assert!(result.rows.is_empty());
        assert!(result.source_paths.is_empty());
        assert_eq!(result.stats.indexed_files, 0);
        assert_eq!(result.stats.skipped_files, 1);
        assert!(result.stats.errors.is_empty());
    }

    #[tokio::test]
    async fn test_ingest_path_reports_text_embedding_error() {
        let dir = tempfile::tempdir().unwrap();
        let text = dir.path().join("note.txt");
        fs::write(&text, "hello world").unwrap();

        let base_url = one_shot_json_server("500 Internal Server Error", r#"{"error":"boom"}"#);
        let embedder = Embedder::new(base_url, "embed".to_string());
        let result = ingest_path(
            &text,
            100,
            0,
            &embedder,
            None,
            PdfParser::Native,
            &[],
            false,
            &BTreeMap::new(),
            false,
        )
        .await
        .unwrap();

        assert!(result.rows.is_empty());
        assert_eq!(result.stats.indexed_files, 0);
        assert_eq!(result.stats.skipped_files, 1);
        assert!(result.stats.errors[0].contains("Ollama embed API error"));
    }
}
