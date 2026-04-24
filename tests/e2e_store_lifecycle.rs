mod support;

use support::{run_ragcli, MockOllamaConfig, MockOllamaServer};

#[test]
fn test_index_sources_and_stat_cover_store_lifecycle() {
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

    let server = MockOllamaServer::start(MockOllamaConfig::default());
    let env = [("RAGCLI_OLLAMA_URL", server.url())];

    let index = run_ragcli(
        dir.path(),
        &env,
        ["--name", "lifecycle", "index", docs.to_str().unwrap()],
    );
    index.assert_success();
    assert!(index.stdout.contains("Index complete: 2 files, 2 chunks"));

    let sources = run_ragcli(
        dir.path(),
        &env,
        ["--name", "lifecycle", "sources", "--json"],
    );
    sources.assert_success();
    let sources_json = sources.json();
    assert_eq!(sources_json["total_sources"], 2);
    assert_eq!(sources_json["sources"].as_array().unwrap().len(), 2);
    let source_paths = sources_json["sources"]
        .as_array()
        .unwrap()
        .iter()
        .map(|source| source["source_path"].as_str().unwrap())
        .collect::<Vec<_>>();
    assert!(source_paths.iter().any(|path| path.ends_with("nebula.md")));
    assert!(source_paths.iter().any(|path| path.ends_with("orchard.md")));

    let stat = run_ragcli(
        dir.path(),
        &env,
        ["--name", "lifecycle", "stat", "--json"],
    );
    stat.assert_success();
    let stat_json = stat.json();
    assert_eq!(stat_json["stats"]["total_chunks"], 2);
    assert_eq!(stat_json["stats"]["unique_sources"], 2);
    assert_eq!(stat_json["stats"]["content_kinds"]["text_files"], 2);
    assert_eq!(stat_json["metadata"]["embedding_dim"], 4);
}

#[test]
fn test_prune_apply_removes_missing_sources_end_to_end() {
    let dir = tempfile::tempdir().unwrap();
    let docs = dir.path().join("docs");
    std::fs::create_dir_all(&docs).unwrap();
    let stale = docs.join("stale.md");
    std::fs::write(&stale, "Project Quartz records calibration baselines.").unwrap();

    let server = MockOllamaServer::start(MockOllamaConfig::default());
    let env = [("RAGCLI_OLLAMA_URL", server.url())];

    let index = run_ragcli(
        dir.path(),
        &env,
        ["--name", "maintenance", "index", stale.to_str().unwrap()],
    );
    index.assert_success();

    std::fs::remove_file(&stale).unwrap();

    let dry_run = run_ragcli(
        dir.path(),
        &env,
        ["--name", "maintenance", "prune", "--json"],
    );
    dry_run.assert_success();
    let dry_run_json = dry_run.json();
    assert_eq!(dry_run_json["dry_run"], true);
    assert_eq!(dry_run_json["deleted_sources"], 0);
    assert_eq!(dry_run_json["stale_sources"].as_array().unwrap().len(), 1);
    assert!(dry_run_json["stale_sources"][0]["source_path"]
        .as_str()
        .unwrap()
        .ends_with("stale.md"));

    let applied = run_ragcli(
        dir.path(),
        &env,
        ["--name", "maintenance", "prune", "--apply", "--json"],
    );
    applied.assert_success();
    let applied_json = applied.json();
    assert_eq!(applied_json["dry_run"], false);
    assert_eq!(applied_json["deleted_sources"], 1);
    assert_eq!(applied_json["deleted_chunks"], 1);

    let sources = run_ragcli(
        dir.path(),
        &env,
        ["--name", "maintenance", "sources", "--json"],
    );
    sources.assert_success();
    assert_eq!(sources.json()["total_sources"], 0);
}
