mod support;

use support::{run_ragcli, MockOllamaConfig, MockOllamaServer};

#[test]
fn test_doctor_json_reports_store_layout_and_installed_models() {
    let dir = tempfile::tempdir().unwrap();
    let server = MockOllamaServer::start(MockOllamaConfig::default());
    let env = [("RAGCLI_OLLAMA_URL", server.url())];

    let doctor = run_ragcli(
        dir.path(),
        &env,
        ["--name", "doctor-e2e", "doctor", "--json"],
    );
    doctor.assert_success();
    let report = doctor.json();

    assert_eq!(report["ollama_reachable"], true);
    assert_eq!(report["installed_models"]["embed_model_installed"], true);
    assert_eq!(report["installed_models"]["chat_model_installed"], true);
    assert_eq!(report["installed_models"]["vision_model_installed"], true);
    assert_eq!(report["store"]["status"], "exists");
    assert_eq!(report["config"]["status"], "exists");
    assert_eq!(report["metadata"]["status"], "missing");
    assert_eq!(report["subdirectories"].as_array().unwrap().len(), 4);
}
