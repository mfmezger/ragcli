mod support;

use support::run_ragcli;

#[test]
#[ignore = "requires a running Ollama with the configured models installed"]
fn test_live_ollama_index_and_query_smoke() {
    if std::env::var_os("RAGCLI_LIVE_E2E").is_none() {
        eprintln!("skipping live smoke test; set RAGCLI_LIVE_E2E=1 to enable");
        return;
    }

    let dir = tempfile::tempdir().unwrap();
    let docs = dir.path().join("docs");
    std::fs::create_dir_all(&docs).unwrap();
    let note = docs.join("nebula.txt");
    std::fs::write(
        &note,
        "Project Nebula maps star catalogs for observatory search.",
    )
    .unwrap();

    let ollama_url =
        std::env::var("RAGCLI_OLLAMA_URL").unwrap_or_else(|_| "http://localhost:11434".to_string());
    let env = [("RAGCLI_OLLAMA_URL", ollama_url.as_str())];

    let index = run_ragcli(
        dir.path(),
        &env,
        ["--name", "live-smoke", "index", note.to_str().unwrap()],
    );
    index.assert_success();
    assert!(index.stdout.contains("Index complete:"));

    let query = run_ragcli(
        dir.path(),
        &env,
        [
            "--name",
            "live-smoke",
            "query",
            "What does Project Nebula map?",
            "--mode",
            "hybrid",
            "--top-k",
            "1",
        ],
    );
    query.assert_success();
    assert!(!query.stdout.trim().is_empty());
}
