use crate::models::{Embedder, VisionCaptioner};
use crate::store::ChunkRow;
use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use pdf_extract::extract_text_by_pages;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

pub struct IndexStats {
    pub indexed_files: usize,
    pub skipped_files: usize,
    pub total_chunks: usize,
    pub errors: Vec<String>,
}

pub struct IngestResult {
    pub rows: Vec<ChunkRow>,
    pub source_paths: Vec<String>,
    pub embedding_dim: Option<usize>,
    pub stats: IndexStats,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FileKind {
    Text,
    Pdf,
    Image,
    Unsupported,
}

pub async fn ingest_path(
    path: &Path,
    chunk_size: usize,
    overlap: usize,
    embedder: &Embedder,
    vision: Option<&VisionCaptioner>,
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
                        let chunks = chunk_text(&text, chunk_size, overlap);
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
                match extract_text_by_pages(&file)
                    .with_context(|| format!("extract pdf text: {}", file.display()))
                {
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
                            let chunks = chunk_text(&page_text, chunk_size, overlap);
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
                        let chunks = vec![build_image_retrieval_text(&file, &caption)];
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
    chunks: Vec<String>,
    embedder: &Embedder,
    dim: &mut Option<usize>,
) -> Result<Vec<ChunkRow>> {
    let mut rows = Vec::new();
    for (idx, chunk) in chunks.into_iter().enumerate() {
        let embedding = embedder.embed(&chunk).await?;
        if dim.is_none() {
            *dim = Some(embedding.len());
        }
        let chunk_hash = hash_text(path, &chunk, page_num, idx as i32);
        rows.push(ChunkRow {
            id: chunk_hash.clone(),
            source_path: path.display().to_string(),
            chunk_text: chunk,
            chunk_hash,
            page: page_num,
            chunk_index: idx as i32,
            metadata: serde_json::json!({
                "source": path.display().to_string(),
                "page": page_num,
            })
            .to_string(),
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
        "md" | "markdown" | "txt" | "rst" => FileKind::Text,
        "pdf" => FileKind::Pdf,
        "png" | "jpg" | "jpeg" | "webp" => FileKind::Image,
        _ => FileKind::Unsupported,
    }
}

fn load_text_file(path: &Path) -> Result<String> {
    let bytes = fs::read(path)?;
    Ok(String::from_utf8_lossy(&bytes).into_owned())
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
    fn test_load_text_file_is_lossy() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("note.txt");
        fs::write(&path, vec![0x66, 0x6f, 0x80, 0x6f]).unwrap();

        let text = load_text_file(&path).unwrap();
        assert!(text.contains('f'));
        assert!(text.contains('\u{fffd}'));
    }
}
