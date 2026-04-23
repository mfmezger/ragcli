use serde_json::Value;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::process::Command;
use std::thread;

fn one_shot_ok_server(body: &'static str) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();

    thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut buf = [0_u8; 4096];
        let _ = stream.read(&mut buf);
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            body.len(),
            body
        );
        stream.write_all(response.as_bytes()).unwrap();
    });

    format!("http://{}", addr)
}

fn otlp_stub_server() -> (String, std::sync::mpsc::Receiver<String>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let (tx, rx) = std::sync::mpsc::channel();

    thread::spawn(move || {
        if let Ok((mut stream, _)) = listener.accept() {
            let mut buf = [0_u8; 8192];
            let size = stream.read(&mut buf).unwrap_or_default();
            let request = String::from_utf8_lossy(&buf[..size]).to_string();
            let _ = tx.send(request);
            let response = "HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
            stream.write_all(response.as_bytes()).unwrap();
        }
    });

    (format!("http://{}", addr), rx)
}

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

#[test]
fn test_doctor_json_reports_telemetry_configuration() {
    let dir = tempfile::tempdir().unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_ragcli"))
        .args(["doctor", "--json"])
        .env("XDG_CONFIG_HOME", dir.path())
        .env("OTEL_SERVICE_NAME", "ragcli-test")
        .env("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf")
        .env("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:6006")
        .env("OTEL_EXPORTER_OTLP_HEADERS", "api_key=secret")
        .env("OTEL_EXPORTER_OTLP_TIMEOUT", "2500")
        .output()
        .unwrap();

    assert!(output.status.success());

    let stdout = String::from_utf8(output.stdout).unwrap();
    let report: Value = serde_json::from_str(&stdout).unwrap();

    assert_eq!(report["telemetry"]["enabled"], true);
    assert_eq!(report["telemetry"]["service_name"], "ragcli-test");
    assert_eq!(report["telemetry"]["protocol"], "http/protobuf");
    assert_eq!(
        report["telemetry"]["endpoint"],
        "http://localhost:6006/v1/traces"
    );
    assert_eq!(report["telemetry"]["timeout_ms"], 2500);
    assert_eq!(report["telemetry"]["headers_configured"], true);
    assert_eq!(report["telemetry_error"], Value::Null);
}

#[test]
fn test_doctor_json_reports_invalid_telemetry_configuration_without_failing() {
    let dir = tempfile::tempdir().unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_ragcli"))
        .args(["doctor", "--json"])
        .env("XDG_CONFIG_HOME", dir.path())
        .env("OTEL_SERVICE_NAME", "ragcli-test")
        .env("OTEL_EXPORTER_OTLP_PROTOCOL", "http/json")
        .env("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:6006")
        .output()
        .unwrap();

    assert!(output.status.success());

    let stdout = String::from_utf8(output.stdout).unwrap();
    let report: Value = serde_json::from_str(&stdout).unwrap();

    assert_eq!(report["telemetry"]["enabled"], true);
    assert_eq!(report["telemetry"]["service_name"], "ragcli-test");
    assert_eq!(report["telemetry"]["protocol"], "http/protobuf");
    assert_eq!(
        report["telemetry"]["endpoint"],
        "http://localhost:6006/v1/traces"
    );
    assert!(report["telemetry_error"]
        .as_str()
        .unwrap()
        .contains("unsupported OTEL_EXPORTER_OTLP_PROTOCOL value"));
}

#[test]
fn test_cli_starts_with_otlp_enabled_and_emits_trace_request() {
    let dir = tempfile::tempdir().unwrap();
    let ollama = one_shot_ok_server(r#"{"models":[]}"#);
    let (otlp_endpoint, requests) = otlp_stub_server();

    let output = Command::new(env!("CARGO_BIN_EXE_ragcli"))
        .args(["doctor"])
        .env("XDG_CONFIG_HOME", dir.path())
        .env("RAGCLI_OLLAMA_URL", &ollama)
        .env("OTEL_SERVICE_NAME", "ragcli-test")
        .env("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf")
        .env("OTEL_EXPORTER_OTLP_ENDPOINT", &otlp_endpoint)
        .output()
        .unwrap();

    assert!(output.status.success());

    let request = requests
        .recv_timeout(std::time::Duration::from_secs(5))
        .expect("expected OTLP request");
    assert!(request.starts_with("POST /v1/traces HTTP/1.1"));
}
