#![allow(dead_code)]

use serde_json::Value;
use std::ffi::OsStr;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::Path;
use std::process::Command;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::thread::{self, JoinHandle};

const TELEMETRY_ENVS: [&str; 5] = [
    "OTEL_SERVICE_NAME",
    "OTEL_EXPORTER_OTLP_ENDPOINT",
    "OTEL_EXPORTER_OTLP_HEADERS",
    "OTEL_EXPORTER_OTLP_PROTOCOL",
    "OTEL_EXPORTER_OTLP_TIMEOUT",
];

pub struct CliOutput {
    pub stdout: String,
    pub stderr: String,
    pub success: bool,
}

impl CliOutput {
    pub fn assert_success(&self) {
        assert!(
            self.success,
            "command failed\nstdout:\n{}\nstderr:\n{}",
            self.stdout,
            self.stderr
        );
    }

    #[allow(dead_code)]
    pub fn json(&self) -> Value {
        serde_json::from_str(&self.stdout).unwrap_or_else(|err| {
            panic!(
                "stdout was not valid json: {err}\nstdout:\n{}\nstderr:\n{}",
                self.stdout, self.stderr
            )
        })
    }
}

pub fn run_ragcli<I, S>(config_home: &Path, extra_env: &[(&str, &str)], args: I) -> CliOutput
where
    I: IntoIterator<Item = S>,
    S: AsRef<OsStr>,
{
    let mut command = Command::new(env!("CARGO_BIN_EXE_ragcli"));
    command.args(args);
    command.env("XDG_CONFIG_HOME", config_home);
    for (key, value) in extra_env {
        command.env(key, value);
    }
    for key in TELEMETRY_ENVS {
        command.env_remove(key);
    }

    let output = command.output().unwrap();
    CliOutput {
        stdout: String::from_utf8(output.stdout).unwrap(),
        stderr: String::from_utf8(output.stderr).unwrap(),
        success: output.status.success(),
    }
}

pub struct MockOllamaConfig {
    pub chat_response: String,
    pub tags_models: Vec<String>,
}

impl Default for MockOllamaConfig {
    fn default() -> Self {
        Self {
            chat_response: "Project Nebula maps star catalogs for observatory search [1]."
                .to_string(),
            tags_models: vec![
                "nomic-embed-text-v2-moe:latest".to_string(),
                "qwen3.5:4b".to_string(),
            ],
        }
    }
}

pub struct MockOllamaServer {
    base_url: String,
    socket_addr: SocketAddr,
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl MockOllamaServer {
    pub fn start(config: MockOllamaConfig) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind mock ollama");
        let socket_addr = listener.local_addr().expect("mock ollama addr");
        let stop = Arc::new(AtomicBool::new(false));
        let stop_thread = Arc::clone(&stop);
        let handle = thread::spawn(move || serve(listener, stop_thread, config));

        Self {
            base_url: format!("http://{socket_addr}"),
            socket_addr,
            stop,
            handle: Some(handle),
        }
    }

    pub fn url(&self) -> &str {
        &self.base_url
    }
}

impl Drop for MockOllamaServer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        let _ = TcpStream::connect(self.socket_addr);
        if let Some(handle) = self.handle.take() {
            handle.join().expect("join mock ollama thread");
        }
    }
}

#[derive(Debug)]
struct HttpRequest {
    method: String,
    path: String,
    body: String,
}

fn serve(listener: TcpListener, stop: Arc<AtomicBool>, config: MockOllamaConfig) {
    loop {
        let (mut stream, _) = listener.accept().expect("accept mock ollama request");
        if stop.load(Ordering::SeqCst) {
            break;
        }

        let request = match read_request(&mut stream) {
            Some(request) => request,
            None => continue,
        };
        let (status, body) = route_request(&config, &request);
        write_response(&mut stream, status, &body);
    }
}

fn read_request(stream: &mut TcpStream) -> Option<HttpRequest> {
    let mut buffer = Vec::new();
    let mut chunk = [0_u8; 1024];
    let headers_end;

    loop {
        let read = stream.read(&mut chunk).ok()?;
        if read == 0 {
            return None;
        }
        buffer.extend_from_slice(&chunk[..read]);
        if let Some(position) = find_bytes(&buffer, b"\r\n\r\n") {
            headers_end = position + 4;
            break;
        }
    }

    let header_text = String::from_utf8_lossy(&buffer[..headers_end]);
    let mut lines = header_text.lines();
    let request_line = lines.next()?;
    let mut parts = request_line.split_whitespace();
    let method = parts.next()?.to_string();
    let path = parts.next()?.to_string();
    let content_length = lines
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            if name.eq_ignore_ascii_case("content-length") {
                value.trim().parse::<usize>().ok()
            } else {
                None
            }
        })
        .unwrap_or(0);

    while buffer.len() < headers_end + content_length {
        let read = stream.read(&mut chunk).ok()?;
        if read == 0 {
            break;
        }
        buffer.extend_from_slice(&chunk[..read]);
    }

    let body_end = headers_end.saturating_add(content_length).min(buffer.len());
    let body = String::from_utf8_lossy(&buffer[headers_end..body_end]).into();
    Some(HttpRequest { method, path, body })
}

fn route_request(config: &MockOllamaConfig, request: &HttpRequest) -> (&'static str, String) {
    match (request.method.as_str(), request.path.as_str()) {
        ("GET", "/api/tags") => {
            let body = serde_json::json!({
                "models": config
                    .tags_models
                    .iter()
                    .map(|name| serde_json::json!({ "name": name }))
                    .collect::<Vec<_>>()
            });
            ("200 OK", body.to_string())
        }
        ("POST", "/api/embed") => {
            let input = serde_json::from_str::<Value>(&request.body)
                .ok()
                .and_then(|value| value.get("input").and_then(Value::as_str).map(str::to_owned))
                .unwrap_or_default();
            let body = serde_json::json!({
                "embeddings": [compute_embedding(&input)],
            });
            ("200 OK", body.to_string())
        }
        ("POST", "/api/chat") => {
            let body = serde_json::json!({
                "message": {
                    "content": config.chat_response,
                }
            });
            ("200 OK", body.to_string())
        }
        _ => (
            "404 Not Found",
            serde_json::json!({"error": "not found"}).to_string(),
        ),
    }
}

fn compute_embedding(input: &str) -> Vec<f32> {
    let normalized = input.to_ascii_lowercase();
    let mut embedding = vec![0.0, 0.0, 0.0, 0.0];
    if normalized.contains("nebula") {
        embedding[0] = 1.0;
    }
    if normalized.contains("orchard") {
        embedding[1] = 1.0;
    }
    if normalized.contains("quartz") {
        embedding[2] = 1.0;
    }
    if embedding.iter().all(|value| *value == 0.0) {
        embedding[3] = 1.0;
    }
    embedding
}

fn write_response(stream: &mut TcpStream, status: &str, body: &str) {
    let response = format!(
        "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        body.len(),
        body
    );
    stream
        .write_all(response.as_bytes())
        .expect("write mock ollama response");
}

fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|window| window == needle)
}
