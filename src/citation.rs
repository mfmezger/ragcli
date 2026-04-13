//! Citation rendering helpers that produce labeled, human-readable context lines
//! from retrieval candidates for use in generation prompts.

use crate::retrieval::RetrievalCandidate;

/// Formats a list of candidates as labeled context blocks ready for inclusion in a prompt.
///
/// Each candidate is assigned a numeric label (1-based), and the chunk text is prefixed
/// with a location line (`source=<path>` or `source=<path>, page=<N>` when pagination
/// is available). Example output:
///
/// ```text
/// [1] source=docs/config.rs, page=3
/// Chunk text here...
/// ```
///
/// Returns a vector of one formatted string per candidate, in the same order as input.
pub fn labeled_contexts(candidates: &[RetrievalCandidate]) -> Vec<String> {
    candidates
        .iter()
        .enumerate()
        .map(|(idx, candidate)| {
            let label = idx + 1;
            let location = match candidate.page {
                page if page > 0 => format!("source={}, page={page}", candidate.source_path),
                _ => format!("source={}", candidate.source_path),
            };
            format!("[{label}] {location}\n{}", candidate.chunk_text)
        })
        .collect()
}

/// Formats a list of candidates as compact inline citations for reference display.
///
/// Each candidate produces a single line of the form `[N] <source>` or
/// `[N] <source> (page: N)` when pagination is present. Unlike [`labeled_contexts`],
/// this does not include the chunk text — it is purely a citation index.
pub fn render_citations(candidates: &[RetrievalCandidate]) -> Vec<String> {
    candidates
        .iter()
        .enumerate()
        .map(|(idx, candidate)| match candidate.page {
            page if page > 0 => {
                format!("[{}] {} (page: {})", idx + 1, candidate.source_path, page)
            }
            _ => format!("[{}] {}", idx + 1, candidate.source_path),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::retrieval::RetrievalCandidate;

    fn candidate(source_path: &str, page: i32, chunk_text: &str) -> RetrievalCandidate {
        RetrievalCandidate {
            id: String::new(),
            source_path: source_path.to_string(),
            chunk_text: chunk_text.to_string(),
            metadata: String::new(),
            page,
            chunk_index: 0,
            vector_score: None,
            keyword_score: None,
            fused_score: None,
            rerank_score: None,
        }
    }

    #[test]
    fn test_labeled_contexts_include_numeric_labels() {
        let contexts = labeled_contexts(&[candidate("src/config.rs", 0, "config text")]);
        assert_eq!(contexts, vec!["[1] source=src/config.rs\nconfig text"]);
    }

    #[test]
    fn test_render_citations_includes_page_when_present() {
        let citations = render_citations(&[candidate("book.pdf", 4, "page text")]);
        assert_eq!(citations, vec!["[1] book.pdf (page: 4)"]);
    }
}
