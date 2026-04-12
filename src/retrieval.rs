use std::cmp::Ordering;
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq)]
pub struct RetrievalCandidate {
    pub id: String,
    pub source_path: String,
    pub chunk_text: String,
    pub metadata: String,
    pub page: i32,
    pub chunk_index: i32,
    pub vector_score: Option<f32>,
    pub keyword_score: Option<f32>,
    pub fused_score: Option<f32>,
    pub rerank_score: Option<f32>,
}

impl RetrievalCandidate {
    pub fn best_score(&self) -> Option<f32> {
        self.rerank_score
            .or(self.fused_score)
            .or(self.vector_score)
            .or(self.keyword_score)
    }

    pub fn dedupe_key(&self) -> String {
        if !self.id.is_empty() {
            return self.id.clone();
        }

        format!(
            "{}::{}::{}::{}",
            self.source_path, self.page, self.chunk_index, self.chunk_text
        )
    }
}

pub fn merge_candidates(
    groups: impl IntoIterator<Item = Vec<RetrievalCandidate>>,
) -> Vec<RetrievalCandidate> {
    let mut merged: BTreeMap<String, RetrievalCandidate> = BTreeMap::new();

    for group in groups {
        for candidate in group {
            let key = candidate.dedupe_key();
            match merged.get_mut(&key) {
                Some(existing) => merge_candidate(existing, candidate),
                None => {
                    merged.insert(key, candidate);
                }
            }
        }
    }

    let mut out: Vec<_> = merged.into_values().collect();
    out.sort_by(compare_candidates);
    out
}

pub fn prune_candidates(
    candidates: Vec<RetrievalCandidate>,
    top_k: usize,
) -> Vec<RetrievalCandidate> {
    let mut candidates = candidates;
    candidates.sort_by(compare_candidates);

    let mut kept = Vec::new();
    let mut previous_score = None;
    for candidate in candidates.into_iter().take(top_k) {
        let current_score = candidate.best_score();
        if kept.len() >= 2 && should_autocut(previous_score, current_score) {
            break;
        }
        previous_score = current_score;
        kept.push(candidate);
    }

    kept
}

pub fn apply_rerank_order(
    candidates: &[RetrievalCandidate],
    reranked_ids: &[(String, f32)],
    top_n: usize,
) -> Vec<RetrievalCandidate> {
    let mut by_key: BTreeMap<String, RetrievalCandidate> = candidates
        .iter()
        .cloned()
        .map(|candidate| (candidate.dedupe_key(), candidate))
        .collect();
    let mut reranked = Vec::new();

    for (id, score) in reranked_ids.iter().take(top_n) {
        if let Some(mut candidate) = by_key.remove(id) {
            candidate.rerank_score = Some(*score);
            reranked.push(candidate);
        }
    }

    reranked.extend(by_key.into_values());
    reranked.sort_by(compare_candidates);
    reranked.truncate(top_n);
    reranked
}

fn merge_candidate(existing: &mut RetrievalCandidate, next: RetrievalCandidate) {
    existing.vector_score = max_option(existing.vector_score, next.vector_score);
    existing.keyword_score = max_option(existing.keyword_score, next.keyword_score);
    existing.fused_score = max_option(existing.fused_score, next.fused_score);
    existing.rerank_score = max_option(existing.rerank_score, next.rerank_score);

    if existing.metadata.is_empty() && !next.metadata.is_empty() {
        existing.metadata = next.metadata;
    }
    if existing.id.is_empty() && !next.id.is_empty() {
        existing.id = next.id;
    }
}

fn max_option(left: Option<f32>, right: Option<f32>) -> Option<f32> {
    match (left, right) {
        (Some(left), Some(right)) => Some(left.max(right)),
        (Some(left), None) => Some(left),
        (None, Some(right)) => Some(right),
        (None, None) => None,
    }
}

fn should_autocut(previous_score: Option<f32>, current_score: Option<f32>) -> bool {
    match (previous_score, current_score) {
        (Some(previous), Some(current)) if previous > 0.0 => current / previous < 0.6,
        _ => false,
    }
}

fn compare_candidates(left: &RetrievalCandidate, right: &RetrievalCandidate) -> Ordering {
    let left_score = left.best_score().unwrap_or(f32::NEG_INFINITY);
    let right_score = right.best_score().unwrap_or(f32::NEG_INFINITY);

    right_score
        .partial_cmp(&left_score)
        .unwrap_or(Ordering::Equal)
        .then_with(|| left.source_path.cmp(&right.source_path))
        .then_with(|| left.chunk_index.cmp(&right.chunk_index))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(id: &str, source: &str, chunk_index: i32, score: f32) -> RetrievalCandidate {
        RetrievalCandidate {
            id: id.to_string(),
            source_path: source.to_string(),
            chunk_text: format!("chunk-{chunk_index}"),
            metadata: String::new(),
            page: 0,
            chunk_index,
            vector_score: Some(score),
            keyword_score: None,
            fused_score: None,
            rerank_score: None,
        }
    }

    #[test]
    fn test_merge_candidates_dedupes_and_keeps_best_score() {
        let merged = merge_candidates([
            vec![
                candidate("a", "src/a.rs", 0, 0.2),
                candidate("b", "src/b.rs", 1, 0.4),
            ],
            vec![candidate("a", "src/a.rs", 0, 0.9)],
        ]);

        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0].id, "a");
        assert_eq!(merged[0].vector_score, Some(0.9));
        assert_eq!(merged[1].id, "b");
    }

    #[test]
    fn test_best_score_prefers_more_specific_scores() {
        let candidate = RetrievalCandidate {
            id: "a".to_string(),
            source_path: "src/a.rs".to_string(),
            chunk_text: "chunk".to_string(),
            metadata: String::new(),
            page: 0,
            chunk_index: 0,
            vector_score: Some(0.1),
            keyword_score: Some(0.2),
            fused_score: Some(0.3),
            rerank_score: Some(0.4),
        };

        assert_eq!(candidate.best_score(), Some(0.4));
    }

    #[test]
    fn test_prune_candidates_autocuts_after_score_cliff() {
        let kept = prune_candidates(
            vec![
                candidate("a", "src/a.rs", 0, 0.95),
                candidate("b", "src/b.rs", 1, 0.9),
                candidate("c", "src/c.rs", 2, 0.4),
            ],
            3,
        );

        assert_eq!(kept.len(), 2);
        assert_eq!(kept[0].id, "a");
        assert_eq!(kept[1].id, "b");
    }

    #[test]
    fn test_apply_rerank_order_promotes_ranked_candidates() {
        let reranked = apply_rerank_order(
            &[
                candidate("a", "src/a.rs", 0, 0.2),
                candidate("b", "src/b.rs", 1, 0.9),
            ],
            &[("a".to_string(), 0.8), ("b".to_string(), 0.1)],
            2,
        );

        assert_eq!(reranked[0].id, "a");
        assert_eq!(reranked[0].rerank_score, Some(0.8));
    }
}
