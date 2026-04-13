use crate::jsonutil::parse_json;
use crate::models::Generator;
use anyhow::{Context, Result};
use serde::Deserialize;
use std::collections::BTreeSet;

#[derive(Clone, Debug, PartialEq)]
pub struct QueryRewriteSet {
    pub original: String,
    pub semantic_variant: Option<String>,
    pub keyword_variant: Option<String>,
    pub subqueries: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct QueryRewritePayload {
    semantic_variant: Option<String>,
    keyword_variant: Option<String>,
    #[serde(default)]
    subqueries: Vec<String>,
}

impl QueryRewriteSet {
    pub fn fallback(question: &str) -> Self {
        Self {
            original: question.to_string(),
            semantic_variant: None,
            keyword_variant: None,
            subqueries: Vec::new(),
        }
    }

    pub fn query_variants(&self) -> Vec<String> {
        let mut seen = BTreeSet::new();
        let mut variants = Vec::new();

        for candidate in [
            Some(self.original.as_str()),
            self.semantic_variant.as_deref(),
            self.keyword_variant.as_deref(),
        ] {
            let Some(candidate) = candidate.map(str::trim).filter(|value| !value.is_empty()) else {
                continue;
            };
            if seen.insert(candidate.to_string()) {
                variants.push(candidate.to_string());
            }
        }

        variants
    }
}

pub async fn rewrite_query_for_retrieval(
    generator: &Generator,
    question: &str,
) -> Result<QueryRewriteSet> {
    let response = generator
        .generate_json(
            "You rewrite retrieval queries for a local RAG CLI. Respond with strict JSON only and no markdown fences.",
            &format!(
                concat!(
                    "Return a JSON object with keys semantic_variant, keyword_variant, and subqueries. ",
                    "semantic_variant should restate the question for semantic retrieval. ",
                    "keyword_variant should contain sparse retrieval keywords. ",
                    "subqueries should be an array and may be empty. ",
                    "Question: {}"
                ),
                question,
            ),
            256,
        )
        .await
        .context("generate query rewrite JSON")?;
    parse_query_rewrite_set(&response, question)
}

pub fn parse_query_rewrite_set(raw: &str, question: &str) -> Result<QueryRewriteSet> {
    let payload = parse_query_rewrite_payload(raw)?;
    Ok(QueryRewriteSet {
        original: question.to_string(),
        semantic_variant: clean_optional(payload.semantic_variant),
        keyword_variant: clean_optional(payload.keyword_variant),
        subqueries: payload
            .subqueries
            .into_iter()
            .map(|item| item.trim().to_string())
            .filter(|item| !item.is_empty())
            .collect(),
    })
}

fn parse_query_rewrite_payload(raw: &str) -> Result<QueryRewritePayload> {
    parse_json(raw, "parse query rewrite payload")
}

/// Provided for backwards compatibility — prefer [`jsonutil::trim_json_fences`].
pub fn trim_json_fences(raw: &str) -> &str {
    crate::jsonutil::trim_json_fences(raw)
}

fn clean_optional(value: Option<String>) -> Option<String> {
    value
        .map(|item| item.trim().to_string())
        .filter(|item| !item.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_query_rewrite_set_accepts_json_fences() {
        let rewrite = parse_query_rewrite_set(
            "```json\n{\"semantic_variant\":\"Explain config resolution\",\"keyword_variant\":\"config resolve model\",\"subqueries\":[]}\n```",
            "How does config resolution work?",
        )
        .unwrap();

        assert_eq!(rewrite.original, "How does config resolution work?");
        assert_eq!(
            rewrite.semantic_variant.as_deref(),
            Some("Explain config resolution")
        );
        assert_eq!(
            rewrite.keyword_variant.as_deref(),
            Some("config resolve model")
        );
    }

    #[test]
    fn test_query_variants_dedupes_and_skips_empty_entries() {
        let rewrite = QueryRewriteSet {
            original: "question".to_string(),
            semantic_variant: Some("question".to_string()),
            keyword_variant: Some("keywords".to_string()),
            subqueries: Vec::new(),
        };

        assert_eq!(rewrite.query_variants(), vec!["question", "keywords"]);
    }

    #[test]
    fn test_parse_query_rewrite_set_errors_on_invalid_json() {
        let err: String = parse_query_rewrite_set("not json", "question")
            .unwrap_err()
            .to_string();
        assert!(err.contains("parse query rewrite payload"));
    }

    #[test]
    fn test_trim_json_fences_leaves_plain_json_untouched() {
        assert_eq!(trim_json_fences("{\"a\":1}"), "{\"a\":1}");
    }
}
