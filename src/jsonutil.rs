//! Shared JSON utilities for parsing LLM-generated payloads.
//!
//! LLM responses often include markdown fences and whitespace that must be stripped
//! before deserialization. This module provides consistent parsing with error messages
//! that include the raw content for easier debugging.

use anyhow::{Context, Result};
use serde::de::DeserializeOwned;

/// Strips markdown JSON code fences from LLM output.
///
/// Handles both fenced (` """json ... """ `) and unfenced responses.
/// Returns the inner JSON text, trimmed of surrounding whitespace.
pub fn trim_json_fences(raw: &str) -> &str {
    let trimmed = raw.trim();
    if !trimmed.starts_with("```") {
        return trimmed;
    }

    let without_prefix = trimmed
        .trim_start_matches("```")
        .trim_start()
        .trim_start_matches("json")
        .trim();

    without_prefix
        .strip_suffix("```")
        .unwrap_or(without_prefix)
        .trim()
}

/// Parses a JSON payload from an LLM response, stripping fences if present.
///
/// Includes a snippet of the raw content in error messages so failures are actionable.
pub fn parse_json<T: DeserializeOwned>(raw: &str, context: &str) -> Result<T> {
    let cleaned = trim_json_fences(raw);
    serde_json::from_str(cleaned).with_context(|| {
        let snippet = cleaned
            .chars()
            .take(200)
            .collect::<String>()
            .replace(['\n', '\r'], " ");
        format!("{context} — raw content (first 200 chars): {snippet}")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trim_json_fences_leaves_plain_json_untouched() {
        assert_eq!(trim_json_fences(r#"{"a":1}"#), r#"{"a":1}"#);
    }

    #[test]
    fn test_trim_json_fences_handles_json_fence() {
        let input = "```json\n{\"a\":1}\n```";
        assert_eq!(trim_json_fences(input), r#"{"a":1}"#);
    }

    #[test]
    fn test_trim_json_fences_handles_fence_without_json_tag() {
        let input = "```\n{\"a\":1}\n```";
        assert_eq!(trim_json_fences(input), r#"{"a":1}"#);
    }

    #[test]
    fn test_trim_json_fences_handles_extra_whitespace() {
        let input = "  ```json  \n  {\"a\":1}  \n  ```  ";
        assert_eq!(trim_json_fences(input), r#"{"a":1}"#);
    }

    #[test]
    fn test_trim_json_fences_handles_plain_json_with_whitespace() {
        assert_eq!(trim_json_fences("  {\"a\":1}  "), r#"{"a":1}"#);
    }

    #[test]
    fn test_trim_json_fences_handles_empty_input() {
        assert_eq!(trim_json_fences(""), "");
        assert_eq!(trim_json_fences("   "), "");
    }

    #[test]
    fn test_parse_json_includes_raw_snippet_in_error() {
        let err = parse_json::<serde_json::Value>("not json at all", "parse foo").unwrap_err();
        let err_msg = err.to_string();
        assert!(err_msg.contains("parse foo"));
        assert!(err_msg.contains("not json"));
    }

    #[test]
    fn test_parse_json_accepts_fenced_input() {
        #[derive(serde::Deserialize, Debug, PartialEq)]
        struct Payload {
            value: String,
        }

        let result = parse_json::<Payload>(
            "```json\n{\"value\":\"hello\"}
```",
            "test parse",
        )
        .unwrap();
        assert_eq!(result.value, "hello");
    }

    #[test]
    fn test_parse_json_accepts_plain_json() {
        #[derive(serde::Deserialize, Debug, PartialEq)]
        struct Payload {
            value: String,
        }

        let result = parse_json::<Payload>(r#"{"value":"hello"}"#, "test parse").unwrap();
        assert_eq!(result.value, "hello");
    }
}
