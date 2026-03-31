//! File ingestion and chunking for the local RAG store.

use crate::models::{Embedder, VisionCaptioner};
use crate::store::ChunkRow;
use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use pdf_extract::extract_text_by_pages;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use walkdir::WalkDir;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FileKind {
    Text,
    Markdown,
    Pdf,
    Image,
    Unsupported,
}

#[derive(Debug, Clone)]
struct ChunkContent {
    text: String,
    metadata: Value,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PdfParser {
    Native,
    Liteparse,
}

/// Walks a file or directory, extracts supported content, and returns chunk rows.
pub async fn ingest_path(
    path: &Path,
    chunk_size: usize,
    overlap: usize,
    embedder: &Embedder,
    vision: Option<&VisionCaptioner>,
    pdf_parser: PdfParser,
) -> Result<IngestResult> {
    let files = collect_files(path)?;
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
        match file_kind(&file) {
            FileKind::Text => {
                source_paths.insert(file.display().to_string());
                match load_text_file(&file) {
                    Ok(text) => {
                        progress.set_message(format!("Embedding {}", display_name(&file)));
                        let chunks = chunk_plain_text(&text, chunk_size, overlap);
                        match embed_chunks(&file, 0, chunks, embedder, &mut dim).await {
                            Ok(mut file_rows) => {
                                stats.indexed_files += 1;
                                stats.total_chunks += file_rows.len();
                                rows.append(&mut file_rows);
                            }
                            Err(err) => {
                                stats.skipped_files += 1;
                                stats.errors.push(format!("{}: {}", file.display(), err));
                            }
                        }
                    }
                    Err(err) => {
                        stats.skipped_files += 1;
                        stats.errors.push(format!("{}: {}", file.display(), err));
                    }
                }
            }
            FileKind::Markdown => {
                source_paths.insert(file.display().to_string());
                match load_text_file(&file) {
                    Ok(text) => {
                        progress.set_message(format!("Embedding {}", display_name(&file)));
                        let chunks = chunk_markdown(&text, chunk_size, overlap);
                        match embed_chunks(&file, 0, chunks, embedder, &mut dim).await {
                            Ok(mut file_rows) => {
                                stats.indexed_files += 1;
                                stats.total_chunks += file_rows.len();
                                rows.append(&mut file_rows);
                            }
                            Err(err) => {
                                stats.skipped_files += 1;
                                stats.errors.push(format!("{}: {}", file.display(), err));
                            }
                        }
                    }
                    Err(err) => {
                        stats.skipped_files += 1;
                        stats.errors.push(format!("{}: {}", file.display(), err));
                    }
                }
            }
            FileKind::Pdf => {
                source_paths.insert(file.display().to_string());
                progress.set_message(format!("Extracting {}", display_name(&file)));
                match extract_pdf_pages(&file, pdf_parser) {
                    Ok(pages) => {
                        let mut file_rows = Vec::new();
                        let mut file_failed = None;
                        for (page_idx, page_text) in pages.into_iter().enumerate() {
                            let page_num = (page_idx + 1) as i32;
                            progress.set_message(format!(
                                "Embedding {} (page {})",
                                display_name(&file),
                                page_num
                            ));
                            let chunks = chunk_pdf_page(&page_text, page_num, chunk_size, overlap);
                            match embed_chunks(&file, page_num, chunks, embedder, &mut dim).await {
                                Ok(mut page_rows) => file_rows.append(&mut page_rows),
                                Err(err) => {
                                    file_failed = Some(err);
                                    break;
                                }
                            }
                        }

                        if let Some(err) = file_failed {
                            stats.skipped_files += 1;
                            stats.errors.push(format!("{}: {}", file.display(), err));
                        } else {
                            stats.indexed_files += 1;
                            stats.total_chunks += file_rows.len();
                            rows.append(&mut file_rows);
                        }
                    }
                    Err(err) => {
                        stats.skipped_files += 1;
                        stats.errors.push(format!("{}: {}", file.display(), err));
                    }
                }
            }
            FileKind::Image => {
                source_paths.insert(file.display().to_string());
                let Some(vision) = vision else {
                    stats.skipped_files += 1;
                    stats.errors.push(format!(
                        "{}: image skipped because no vision model is configured",
                        file.display()
                    ));
                    continue;
                };

                progress.set_message(format!("Captioning {}", display_name(&file)));
                match vision.caption_image(&file).await {
                    Ok(caption) => {
                        progress.set_message(format!("Embedding {}", display_name(&file)));
                        let chunks = vec![ChunkContent {
                            text: build_image_retrieval_text(&file, &caption),
                            metadata: json!({
                                "format": "image",
                            }),
                        }];
                        match embed_chunks(&file, 0, chunks, embedder, &mut dim).await {
                            Ok(mut file_rows) => {
                                stats.indexed_files += 1;
                                stats.total_chunks += file_rows.len();
                                rows.append(&mut file_rows);
                            }
                            Err(err) => {
                                stats.skipped_files += 1;
                                stats.errors.push(format!("{}: {}", file.display(), err));
                            }
                        }
                    }
                    Err(err) => {
                        stats.skipped_files += 1;
                        stats.errors.push(format!("{}: {}", file.display(), err));
                    }
                }
            }
            FileKind::Unsupported => {}
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

async fn embed_chunks(
    path: &Path,
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
        }
        rows.push(ChunkRow {
            id: chunk_hash.clone(),
            source_path: path.display().to_string(),
            chunk_text: chunk.text,
            chunk_hash,
            page: page_num,
            chunk_index: idx as i32,
            metadata: metadata.to_string(),
            embedding,
        });
    }
    Ok(rows)
}

fn collect_files(path: &Path) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    if path.is_file() {
        files.push(path.to_path_buf());
        return Ok(files);
    }

    for entry in WalkDir::new(path) {
        let entry = entry?;
        if entry.file_type().is_file() {
            files.push(entry.path().to_path_buf());
        }
    }
    Ok(files)
}

fn file_kind(path: &Path) -> FileKind {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();
    match ext.as_str() {
        "md" | "markdown" => FileKind::Markdown,
        "txt" | "rst" => FileKind::Text,
        "pdf" => FileKind::Pdf,
        "png" | "jpg" | "jpeg" | "webp" => FileKind::Image,
        _ => FileKind::Unsupported,
    }
}

fn load_text_file(path: &Path) -> Result<String> {
    let bytes = fs::read(path)?;
    Ok(String::from_utf8_lossy(&bytes).into_owned())
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
        .with_context(|| {
            format!(
                "run liteparse CLI (`lit parse`) for {}",
                path.display()
            )
        })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!(
            "liteparse failed for {}: {}",
            path.display(),
            stderr.trim()
        );
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

fn chunk_plain_text(text: &str, chunk_size: usize, overlap: usize) -> Vec<ChunkContent> {
    chunk_text(text, chunk_size, overlap)
        .into_iter()
        .map(|text| ChunkContent {
            text,
            metadata: json!({
                "format": "text",
            }),
        })
        .collect()
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
                    "format": "pdf",
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
                    "format": "markdown",
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
                "format": "markdown",
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
}
