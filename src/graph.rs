//! Placeholder query planners for the graph retrieval modes (`Local`, `Global`, `Mix`).
//!
//! These modes are stubs awaiting a future graph-indexing milestone. For now they
//! produce stub plans that decompose queries into keyword/semantic variants and fall
//! back to the hybrid retrieval pipeline without graph context.

use crate::cli::QueryModeArg;
use crate::rewrite::QueryRewriteSet;
use std::collections::BTreeSet;

/// Stub plan produced by [`placeholder_plan`] for a graph retrieval mode.
///
/// Contains an execution label describing which stub is active, human-readable notes
/// explaining the current limitation, and a list of query variants to use instead
/// of a proper graph expansion.
#[derive(Clone, Debug, PartialEq)]
pub struct GraphModeStubPlan {
    /// Identifier for which stub plan was selected. One of `local-placeholder`,
    /// `global-placeholder`, or `mix-placeholder`.
    pub execution_label: &'static str,
    /// Human-readable notes describing the stub behavior and pending graph work.
    pub notes: Vec<String>,
    /// Query variants (original + rewritten forms) to feed into hybrid retrieval.
    pub query_variants: Vec<String>,
}

/// Returns a placeholder plan for graph retrieval modes, producing query variants
/// and notes while the full graph indexing pipeline is pending.
///
/// - **`Local`**: narrows to the original question only, assuming graph edges are chunk-local.
/// - **`Global`**: broadens to original + keyword variant for broader coverage.
/// - **`Mix`**: combines original + semantic + keyword variants to explore both local and global.
pub fn placeholder_plan(mode: QueryModeArg, rewrite_set: &QueryRewriteSet) -> GraphModeStubPlan {
    let question = rewrite_set.original.as_str();
    let (execution_label, notes, query_variants) = match mode {
        QueryModeArg::Local => (
            "local-placeholder",
            vec![
                "local mode prioritizes chunk-centric retrieval while graph indexing is pending"
                    .to_string(),
                "current implementation narrows to direct fact-style variants before hybrid retrieval"
                    .to_string(),
            ],
            dedupe_queries([Some(question), None, None]),
        ),
        QueryModeArg::Global => (
            "global-placeholder",
            vec![
                "global mode reserves graph-centric retrieval for a later milestone"
                    .to_string(),
                "current implementation broadens the query set before hybrid retrieval"
                    .to_string(),
            ],
            dedupe_queries([
                Some(question),
                rewrite_set.keyword_variant.as_deref(),
                None,
            ]),
        ),
        QueryModeArg::Mix => (
            "mix-placeholder",
            vec![
                "mix mode will later combine graph neighborhoods with chunk retrieval"
                    .to_string(),
                "current implementation blends local and broad query variants before hybrid retrieval"
                    .to_string(),
            ],
            dedupe_queries([
                Some(question),
                rewrite_set.semantic_variant.as_deref(),
                rewrite_set.keyword_variant.as_deref(),
            ]),
        ),
        _ => (
            "hybrid",
            vec!["non-graph modes do not use the graph placeholder planner".to_string()],
            dedupe_queries([Some(question), None, None]),
        ),
    };

    GraphModeStubPlan {
        execution_label,
        notes,
        query_variants,
    }
}

/// Deduplicates an array of optional query strings, returning only unique non-empty variants
/// in the order they first appear.
fn dedupe_queries(queries: [Option<&str>; 3]) -> Vec<String> {
    let mut seen = BTreeSet::new();
    let mut variants = Vec::new();
    for query in queries.into_iter().flatten() {
        let trimmed = query.trim();
        if !trimmed.is_empty() && seen.insert(trimmed.to_string()) {
            variants.push(trimmed.to_string());
        }
    }
    variants
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rewrite_set() -> QueryRewriteSet {
        QueryRewriteSet {
            original: "How do config checks work?".to_string(),
            semantic_variant: Some("Explain config validation flow".to_string()),
            keyword_variant: Some("config validation metadata".to_string()),
            subqueries: Vec::new(),
        }
    }

    #[test]
    fn test_local_placeholder_uses_local_label() {
        let plan = placeholder_plan(QueryModeArg::Local, &rewrite_set());
        assert_eq!(plan.execution_label, "local-placeholder");
        assert!(plan
            .notes
            .iter()
            .any(|note| note.contains("chunk-centric retrieval")));
        assert_eq!(plan.query_variants, vec!["How do config checks work?"]);
    }

    #[test]
    fn test_global_placeholder_keeps_keyword_variant_without_semantic_variant() {
        let plan = placeholder_plan(QueryModeArg::Global, &rewrite_set());
        assert_eq!(plan.execution_label, "global-placeholder");
        assert_eq!(plan.query_variants.len(), 2);
        assert_eq!(
            plan.query_variants,
            vec![
                "How do config checks work?".to_string(),
                "config validation metadata".to_string(),
            ]
        );
    }

    #[test]
    fn test_mix_placeholder_combines_semantic_and_keyword_variants_in_order() {
        let plan = placeholder_plan(QueryModeArg::Mix, &rewrite_set());
        assert_eq!(plan.execution_label, "mix-placeholder");
        assert_eq!(
            plan.query_variants,
            vec![
                "How do config checks work?".to_string(),
                "Explain config validation flow".to_string(),
                "config validation metadata".to_string(),
            ]
        );
    }
}
