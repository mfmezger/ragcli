mod support;

use serde_json::Value;
use support::{run_ragcli, MockOllamaConfig, MockOllamaServer};

#[test]
fn test_query_returns_answer_with_citations_and_context() {
    let dir = tempfile::tempdir().unwrap();
    let docs = dir.path().join("docs");
    std::fs::create_dir_all(&docs).unwrap();
    let nebula = docs.join("nebula.md");
    let orchard = docs.join("orchard.md");
    std::fs::write(
        &nebula,
        "Project Nebula maps star catalogs for observatory search.",
    )
    .unwrap();
    std::fs::write(
        &orchard,
        "Orchard operations track apple harvest schedules and storage.",
    )
    .unwrap();

    let server = MockOllamaServer::start(MockOllamaConfig {
        chat_response: "Project Nebula maps star catalogs for observatory search [1].".to_string(),
        ..Default::default()
    });
    let env = [("RAGCLI_OLLAMA_URL", server.url())];

    let index = run_ragcli(
        dir.path(),
        &env,
        ["--name", "query-e2e", "index", docs.to_str().unwrap()],
    );
    index.assert_success();

    let query = run_ragcli(
        dir.path(),
        &env,
        [
            "--name",
            "query-e2e",
            "query",
            "What does Project Nebula map?",
            "--mode",
            "hybrid",
            "--top-k",
            "1",
            "--source",
            nebula.to_str().unwrap(),
            "--show-citations",
            "--show-context",
        ],
    );
    query.assert_success();

    assert!(query.stdout.contains("Citations:"), "{}", query.stdout);
    assert!(
        query.stdout.contains("Retrieved context:"),
        "{}",
        query.stdout
    );
    assert!(query.stdout.contains("nebula.md"), "{}", query.stdout);
    assert!(
        query
            .stdout
            .contains("Project Nebula maps star catalogs for observatory search."),
        "{}",
        query.stdout
    );
    assert!(query.stdout.contains("[1]"), "{}", query.stdout);
}

#[test]
fn test_query_json_reports_null_answer_when_no_context_is_retained() {
    let dir = tempfile::tempdir().unwrap();
    let docs = dir.path().join("docs");
    std::fs::create_dir_all(&docs).unwrap();
    let nebula = docs.join("nebula.md");
    std::fs::write(
        &nebula,
        "Project Nebula maps star catalogs for observatory search.",
    )
    .unwrap();

    let server = MockOllamaServer::start(MockOllamaConfig {
        chat_response: "UNSUPPORTED EMPTY CONTEXT ANSWER".to_string(),
        ..Default::default()
    });
    let env = [("RAGCLI_OLLAMA_URL", server.url())];

    let index = run_ragcli(
        dir.path(),
        &env,
        [
            "--name",
            "query-json-empty",
            "index",
            docs.to_str().unwrap(),
        ],
    );
    index.assert_success();

    let query = run_ragcli(
        dir.path(),
        &env,
        [
            "--name",
            "query-json-empty",
            "query",
            "What does Project Nebula map?",
            "--top-k",
            "0",
            "--json",
        ],
    );
    query.assert_success();

    let report: Value = query.json();
    assert_eq!(report["answer"], Value::Null);
    assert_eq!(report["hits"].as_array().unwrap().len(), 0);
    assert!(!query.stdout.contains("UNSUPPORTED EMPTY CONTEXT ANSWER"));
}
