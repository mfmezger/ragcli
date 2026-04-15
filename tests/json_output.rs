use serde_json::Value;
use std::process::Command;

#[test]
fn test_config_show_json_outputs_machine_readable_report() {
    let dir = tempfile::tempdir().unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_ragcli"))
        .args(["config", "show", "--json"])
        .env("XDG_CONFIG_HOME", dir.path())
        .env("RAGCLI_CHAT_MODEL", "test-chat")
        .output()
        .unwrap();

    assert!(output.status.success());

    let stdout = String::from_utf8(output.stdout).unwrap();
    let report: Value = serde_json::from_str(&stdout).unwrap();

    assert_eq!(report["config"]["models"]["chat"], "test-chat");
    assert_eq!(report["sources"]["models_chat"]["kind"], "Env");
    assert_eq!(
        report["sources"]["models_chat"]["env_var"],
        "RAGCLI_CHAT_MODEL"
    );
    assert_eq!(
        report["active_overrides"][0],
        "models.chat <- RAGCLI_CHAT_MODEL"
    );
}
