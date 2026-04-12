use crate::cli::QueryModeArg;
use crate::rewrite::QueryRewriteSet;
use std::collections::BTreeSet;

#[derive(Clone, Debug, PartialEq)]
pub struct GraphModeStubPlan {
    pub execution_label: &'static str,
    pub notes: Vec<String>,
    pub query_variants: Vec<String>,
}

pub fn placeholder_plan(
    mode: QueryModeArg,
    question: &str,
    rewrite_set: &QueryRewriteSet,
) -> GraphModeStubPlan {
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

fn dedupe_queries(queries: [Option<&str>; 3]) -> Vec<String> {
    let mut variants = BTreeSet::new();
    for query in queries.into_iter().flatten() {
        let trimmed = query.trim();
        if !trimmed.is_empty() {
            variants.insert(trimmed.to_string());
        }
    }
    variants.into_iter().collect()
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
        let plan = placeholder_plan(
            QueryModeArg::Local,
            "How do config checks work?",
            &rewrite_set(),
        );
        assert_eq!(plan.execution_label, "local-placeholder");
        assert!(plan
            .notes
            .iter()
            .any(|note| note.contains("chunk-centric retrieval")));
        assert_eq!(plan.query_variants, vec!["How do config checks work?"]);
    }

    #[test]
    fn test_global_placeholder_keeps_keyword_variant_without_semantic_variant() {
        let plan = placeholder_plan(
            QueryModeArg::Global,
            "How do config checks work?",
            &rewrite_set(),
        );
        assert_eq!(plan.execution_label, "global-placeholder");
        assert_eq!(plan.query_variants.len(), 2);
        assert!(plan
            .query_variants
            .iter()
            .any(|query| query.contains("metadata")));
        assert!(!plan
            .query_variants
            .iter()
            .any(|query| query.contains("validation flow")));
    }

    #[test]
    fn test_mix_placeholder_combines_semantic_and_keyword_variants() {
        let plan = placeholder_plan(
            QueryModeArg::Mix,
            "How do config checks work?",
            &rewrite_set(),
        );
        assert_eq!(plan.execution_label, "mix-placeholder");
        assert_eq!(plan.query_variants.len(), 3);
        assert!(plan
            .query_variants
            .iter()
            .any(|query| query.contains("validation flow")));
        assert!(plan
            .query_variants
            .iter()
            .any(|query| query.contains("metadata")));
    }
}
