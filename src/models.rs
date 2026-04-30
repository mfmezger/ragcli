//! Clients for Ollama embedding, chat, and vision APIs.

use anyhow::{Context, Result};
use base64::Engine;
use reqwest::header::CONTENT_TYPE;
use reqwest::Client;
use serde::Deserialize;
use std::path::Path;
use std::time::{Duration, Instant};
use tokio::time::sleep;
use tracing::{field, Instrument};

/// Embedding client backed by Ollama's `/api/embed` endpoint.
pub struct Embedder {
    client: HttpClient,
    base_url: String,
    model: String,
}

/// Text generation client backed by Ollama's `/api/chat` endpoint.
pub struct Generator {
    client: HttpClient,
    base_url: String,
    model: String,
}

/// Vision captioning client that turns images into retrieval text.
pub struct VisionCaptioner {
    client: HttpClient,
    base_url: String,
    model: String,
}

/// Minimal Ollama API client used for health checks and model discovery.
pub struct OllamaClient {
    client: HttpClient,
    base_url: String,
}

enum HttpClient {
    Reqwest(Client),
}

#[derive(Debug, Deserialize)]
struct EmbedResponse {
    embeddings: Vec<Vec<f32>>,
}

#[derive(Debug, Deserialize)]
struct ChatResponse {
    message: ChatMessage,
}

#[derive(Debug, Deserialize)]
struct ChatMessage {
    content: String,
}

#[derive(Debug, Deserialize)]
struct TagsResponse {
    models: Vec<TagModel>,
}

#[derive(Debug, Deserialize)]
struct TagModel {
    name: String,
}

impl OllamaClient {
    /// Creates a new Ollama client for the given base URL.
    pub fn new(base_url: String) -> Self {
        Self {
            client: HttpClient::with_timeout(Duration::from_secs(10)),
            base_url,
        }
    }

    /// Returns the installed model names reported by Ollama.
    pub async fn list_models(&self) -> Result<Vec<String>> {
        let span = tracing::info_span!(
            "ollama_request",
            backend = "ollama",
            operation = "tags",
            endpoint_host = %endpoint_host(&self.base_url),
            success = field::Empty,
            response_models = field::Empty,
            duration_ms = field::Empty,
        );
        let started = Instant::now();
        let result = async {
            let url = format!("{}/api/tags", self.base_url.trim_end_matches('/'));
            let (status, body) = self
                .client
                .get_text(url)
                .await
                .context("call Ollama tags API")?;
            if !status.is_success() {
                anyhow::bail!("Ollama tags API error: {} - {}", status, body);
            }

            let parsed: TagsResponse =
                serde_json::from_str(&body).context("parse Ollama tags response")?;
            Ok(parsed
                .models
                .into_iter()
                .map(|model| model.name)
                .collect::<Vec<_>>())
        }
        .instrument(span.clone())
        .await;

        span.record("success", result.is_ok());
        span.record("duration_ms", started.elapsed().as_millis() as u64);
        if let Ok(models) = &result {
            span.record("response_models", models.len());
        }
        result
    }
}

impl Embedder {
    /// Creates a new embedding client.
    pub fn new(base_url: String, model: String) -> Self {
        Self {
            client: HttpClient::with_timeout(Duration::from_secs(60)),
            base_url,
            model,
        }
    }

    /// Embeds a single text input and returns its vector.
    pub async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let span = tracing::info_span!(
            "ollama_request",
            backend = "ollama",
            operation = "embed",
            model = %self.model,
            endpoint_host = %endpoint_host(&self.base_url),
            input_bytes = text.len(),
            success = field::Empty,
            embedding_dim = field::Empty,
            duration_ms = field::Empty,
        );
        let started = Instant::now();
        let result = async {
            let url = format!("{}/api/embed", self.base_url.trim_end_matches('/'));
            let request_body = serde_json::json!({
                "model": self.model,
                "input": text,
            });
            let (status, body) = self
                .client
                .post_json(url, request_body)
                .await
                .context("call Ollama embed API")?;
            if !status.is_success() {
                return Err(ollama_error("embed", status, &body, &self.model));
            }

            let parsed: EmbedResponse =
                serde_json::from_str(&body).context("parse Ollama embed response")?;
            let embedding = parsed
                .embeddings
                .into_iter()
                .next()
                .context("Ollama embed response did not include an embedding")?;
            Ok(embedding)
        }
        .instrument(span.clone())
        .await;

        span.record("success", result.is_ok());
        span.record("duration_ms", started.elapsed().as_millis() as u64);
        if let Ok(embedding) = &result {
            span.record("embedding_dim", embedding.len());
        }
        result
    }
}

impl Generator {
    /// Creates a new generation client.
    pub fn new(base_url: String, model: String) -> Self {
        Self {
            client: HttpClient::with_timeout(Duration::from_secs(180)),
            base_url,
            model,
        }
    }

    /// Generates an answer grounded in retrieved contexts.
    pub async fn generate_answer(
        &self,
        contexts: &[String],
        question: &str,
        max_tokens: usize,
    ) -> Result<String> {
        self.generate_with_prompts(
            "You are a helpful assistant. Answer only from the provided evidence. Cite supported claims with bracketed evidence labels like [1] or [2]. If the evidence is insufficient, say you don't know. Respond with only the final answer and no chain-of-thought.",
            &build_user_prompt(contexts, question),
            max_tokens,
        )
        .await
    }

    /// Generates a strict-JSON response for structured tasks.
    pub async fn generate_json(
        &self,
        system_prompt: &str,
        user_prompt: &str,
        max_tokens: usize,
    ) -> Result<String> {
        self.generate_with_prompts(system_prompt, user_prompt, max_tokens)
            .await
    }

    async fn generate_with_prompts(
        &self,
        system_prompt: &str,
        user_prompt: &str,
        max_tokens: usize,
    ) -> Result<String> {
        let span = tracing::info_span!(
            "ollama_request",
            backend = "ollama",
            operation = "chat",
            model = %self.model,
            endpoint_host = %endpoint_host(&self.base_url),
            system_prompt_bytes = system_prompt.len(),
            user_prompt_bytes = user_prompt.len(),
            max_tokens,
            success = field::Empty,
            response_bytes = field::Empty,
            duration_ms = field::Empty,
        );
        let started = Instant::now();
        let result = async {
            let url = format!("{}/api/chat", self.base_url.trim_end_matches('/'));
            let request_body = serde_json::json!({
                "model": self.model,
                "stream": false,
                "think": false,
                "options": {
                    "num_predict": max_tokens,
                },
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    }
                ]
            });
            let (status, body) = self
                .client
                .post_json(url, request_body)
                .await
                .context("call Ollama chat API")?;
            if !status.is_success() {
                return Err(ollama_error("chat", status, &body, &self.model));
            }

            let parsed: ChatResponse =
                serde_json::from_str(&body).context("parse Ollama chat response")?;
            Ok(parsed.message.content)
        }
        .instrument(span.clone())
        .await;

        span.record("success", result.is_ok());
        span.record("duration_ms", started.elapsed().as_millis() as u64);
        if let Ok(content) = &result {
            span.record("response_bytes", content.len());
        }
        result
    }
}

impl VisionCaptioner {
    /// Creates a new vision captioning client.
    pub fn new(base_url: String, model: String) -> Self {
        Self {
            client: HttpClient::with_timeout(Duration::from_secs(180)),
            base_url,
            model,
        }
    }

    /// Produces a retrieval-oriented caption for an image file.
    pub async fn caption_image(&self, image_path: &Path) -> Result<String> {
        let span = tracing::info_span!(
            "ollama_request",
            backend = "ollama",
            operation = "vision",
            model = %self.model,
            endpoint_host = %endpoint_host(&self.base_url),
            image_bytes = field::Empty,
            success = field::Empty,
            response_bytes = field::Empty,
            duration_ms = field::Empty,
        );
        let started = Instant::now();
        let result = async {
            let bytes = std::fs::read(image_path)
                .with_context(|| format!("read image: {}", image_path.display()))?;
            span.record("image_bytes", bytes.len());
            let image_b64 = base64::engine::general_purpose::STANDARD.encode(bytes);
            let url = format!("{}/api/chat", self.base_url.trim_end_matches('/'));

            let request_body = serde_json::json!({
                "model": self.model,
                "stream": false,
                "think": false,
                "options": {
                    "num_predict": 128,
                },
                "messages": [
                    {
                        "role": "user",
                        "content": "Describe this image for retrieval. Focus on the subjects, setting, and notable visual details.",
                        "images": [image_b64],
                    }
                ]
            });
            let (status, body) = self
                .client
                .post_json(url, request_body)
                .await
                .context("call Ollama vision chat API")?;
            if !status.is_success() {
                return Err(ollama_error("vision", status, &body, &self.model));
            }

            let parsed: ChatResponse =
                serde_json::from_str(&body).context("parse Ollama vision response")?;
            Ok(parsed.message.content)
        }
        .instrument(span.clone())
        .await;

        span.record("success", result.is_ok());
        span.record("duration_ms", started.elapsed().as_millis() as u64);
        if let Ok(content) = &result {
            span.record("response_bytes", content.len());
        }
        result
    }
}

const MAX_RETRIES: u32 = 2;
const INITIAL_BACKOFF: Duration = Duration::from_millis(500);

impl HttpClient {
    fn with_timeout(timeout: Duration) -> Self {
        Self::Reqwest(
            Client::builder()
                .timeout(timeout)
                .build()
                .expect("build reqwest client"),
        )
    }

    async fn get_text(&self, url: String) -> Result<(reqwest::StatusCode, String)> {
        for attempt in 0..MAX_RETRIES {
            let result = self.get_text_once(&url).await;
            if should_retry(&result) {
                let delay = backoff_duration(attempt);
                log_retry("GET", &url, &result, attempt, delay);
                sleep(delay).await;
                continue;
            }
            return result;
        }
        self.get_text_once(&url).await
    }

    async fn get_text_once(&self, url: &str) -> Result<(reqwest::StatusCode, String)> {
        let response = match self {
            Self::Reqwest(client) => client
                .get(url)
                .send()
                .await
                .context("send HTTP GET request")?,
        };
        let status = response.status();
        let body = response.text().await.context("read HTTP response body")?;
        Ok((status, body))
    }

    async fn post_json(
        &self,
        url: String,
        body: serde_json::Value,
    ) -> Result<(reqwest::StatusCode, String)> {
        let json_body = body.to_string();
        for attempt in 0..MAX_RETRIES {
            let result = self.post_json_once(&url, &json_body).await;
            if should_retry(&result) {
                let delay = backoff_duration(attempt);
                log_retry("POST", &url, &result, attempt, delay);
                sleep(delay).await;
                continue;
            }
            return result;
        }
        self.post_json_once(&url, &json_body).await
    }

    async fn post_json_once(
        &self,
        url: &str,
        json_body: &str,
    ) -> Result<(reqwest::StatusCode, String)> {
        let response = match self {
            Self::Reqwest(client) => client
                .post(url)
                .header(CONTENT_TYPE, "application/json")
                .body(json_body.to_string())
                .send()
                .await
                .context("send HTTP POST request")?,
        };
        let status = response.status();
        let response_body = response.text().await.context("read HTTP response body")?;
        Ok((status, response_body))
    }
}

fn should_retry(result: &Result<(reqwest::StatusCode, String)>) -> bool {
    match result {
        Err(err) => is_connection_error(err),
        Ok((status, _)) => status.is_server_error(),
    }
}

fn backoff_duration(attempt: u32) -> Duration {
    INITIAL_BACKOFF * 2u32.saturating_pow(attempt)
}

fn log_retry(
    method: &str,
    url: &str,
    result: &Result<(reqwest::StatusCode, String)>,
    attempt: u32,
    delay: Duration,
) {
    let retry = attempt + 1;
    let total_attempts = MAX_RETRIES + 1;
    let delay_ms = delay.as_millis() as u64;
    match result {
        Err(err) => tracing::warn!(
            method = %method,
            url = %url,
            retry,
            max_retries = MAX_RETRIES,
            total_attempts,
            delay_ms,
            error = %err,
            "Ollama request failed; retrying"
        ),
        Ok((status, _)) => tracing::warn!(
            method = %method,
            url = %url,
            retry,
            max_retries = MAX_RETRIES,
            total_attempts,
            delay_ms,
            status = %status,
            "Ollama request returned retryable status; retrying"
        ),
    }
}

fn is_connection_error(err: &anyhow::Error) -> bool {
    err.chain().any(|cause| {
        if let Some(reqwest_err) = cause.downcast_ref::<reqwest::Error>() {
            return reqwest_err.is_connect() || reqwest_err.is_timeout();
        }
        if let Some(io_err) = cause.downcast_ref::<std::io::Error>() {
            return matches!(
                io_err.kind(),
                std::io::ErrorKind::BrokenPipe
                    | std::io::ErrorKind::ConnectionRefused
                    | std::io::ErrorKind::ConnectionReset
                    | std::io::ErrorKind::TimedOut
            );
        }

        let msg = cause.to_string().to_ascii_lowercase();
        msg.contains("connection refused")
            || msg.contains("connection reset")
            || msg.contains("broken pipe")
            || msg.contains("connect error")
    })
}

/// Converts a raw Ollama HTTP error into a user-friendly `anyhow::Error`.
///
/// When the response body signals that the requested model is not installed,
/// the returned error includes the exact `ollama pull <model>` command the
/// user needs to run and a tip to use `ragcli doctor` to audit all configured
/// models at once.  All other errors fall back to a compact representation of
/// the HTTP status and body.
fn ollama_error(
    operation: &str,
    status: reqwest::StatusCode,
    body: &str,
    model: &str,
) -> anyhow::Error {
    // Ollama returns {"error": "model 'x' not found, try pulling it first"}
    // when the model has never been pulled.
    let ollama_msg = serde_json::from_str::<serde_json::Value>(body)
        .ok()
        .and_then(|v| v.get("error")?.as_str().map(str::to_owned));

    if let Some(msg) = ollama_msg {
        let lower = msg.to_lowercase();
        if lower.contains("not found") || lower.contains("try pulling") {
            return anyhow::anyhow!(
                "Ollama model '{}' is not installed.\n  \
                 Install it with: ollama pull {}\n  \
                 Run 'ragcli doctor' to check the status of all configured models.",
                model,
                model,
            );
        }
        if is_context_length_message(&msg) {
            return anyhow::anyhow!(
                "Ollama {} API error: {} - input length exceeds the model context length",
                operation,
                status
            );
        }
        return anyhow::anyhow!("Ollama {} API error: {} - {}", operation, status, msg);
    }

    anyhow::anyhow!("Ollama {} API error: {} - {}", operation, status, body)
}

/// Returns true when an error came from an Ollama model context-window limit.
pub fn is_context_length_error(err: &anyhow::Error) -> bool {
    err.chain()
        .any(|cause| is_context_length_message(&cause.to_string()))
}

fn is_context_length_message(message: &str) -> bool {
    let lower = message.to_lowercase();
    lower.contains("input length exceeds") && lower.contains("context length")
}

fn endpoint_host(base_url: &str) -> String {
    reqwest::Url::parse(base_url)
        .ok()
        .map(|url| match (url.host_str(), url.port()) {
            (Some(host), Some(port)) => format!("{host}:{port}"),
            (Some(host), None) => host.to_string(),
            _ => base_url.to_string(),
        })
        .unwrap_or_else(|| base_url.to_string())
}

fn build_user_prompt(contexts: &[String], question: &str) -> String {
    let mut ctx = String::new();
    for item in contexts {
        ctx.push_str(item);
        ctx.push_str("\n\n");
    }

    format!("Context:\n{}\nQuestion: {}", ctx, question)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::thread;

    fn one_shot_server(status_line: &str, body: &'static str) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind test server");
        let addr = listener.local_addr().unwrap();
        let status_line = status_line.to_string();

        thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept request");
            let mut buf = [0_u8; 4096];
            let _ = stream.read(&mut buf);
            let response = format!(
                "HTTP/1.1 {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                status_line,
                body.len(),
                body
            );
            stream
                .write_all(response.as_bytes())
                .expect("write response");
        });

        format!("http://{}", addr)
    }

    #[test]
    fn test_ollama_error_model_not_found_suggests_pull_command() {
        let err = ollama_error(
            "embed",
            reqwest::StatusCode::NOT_FOUND,
            r#"{"error":"model 'nomic-embed-text:latest' not found, try pulling it first"}"#,
            "nomic-embed-text:latest",
        );
        let msg = err.to_string();
        assert!(
            msg.contains("is not installed"),
            "expected 'is not installed' in: {msg}"
        );
        assert!(
            msg.contains("ollama pull nomic-embed-text:latest"),
            "expected pull command in: {msg}"
        );
        assert!(
            msg.contains("ragcli doctor"),
            "expected doctor hint in: {msg}"
        );
    }

    #[test]
    fn test_ollama_error_model_not_found_via_try_pulling_phrase() {
        // Some Ollama versions vary the exact wording; ensure the 'try pulling' phrase
        // also triggers the friendly path.
        let err = ollama_error(
            "chat",
            reqwest::StatusCode::NOT_FOUND,
            r#"{"error":"try pulling the model first"}"#,
            "llama3:latest",
        );
        let msg = err.to_string();
        assert!(
            msg.contains("ollama pull llama3:latest"),
            "expected pull command in: {msg}"
        );
    }

    #[test]
    fn test_ollama_error_context_length_is_detectable() {
        let err = ollama_error(
            "embed",
            reqwest::StatusCode::BAD_REQUEST,
            r#"{"error":"the input length exceeds the context length"}"#,
            "embed-model",
        );

        assert!(is_context_length_error(&err));
        assert!(err.to_string().contains("model context length"));
    }

    #[test]
    fn test_ollama_error_generic_json_error_preserves_message() {
        let err = ollama_error(
            "chat",
            reqwest::StatusCode::INTERNAL_SERVER_ERROR,
            r#"{"error":"out of memory"}"#,
            "llama3:latest",
        );
        let msg = err.to_string();
        assert!(
            msg.contains("out of memory"),
            "expected original message in: {msg}"
        );
        assert!(
            !msg.contains("ollama pull"),
            "should not suggest pull for unrelated errors: {msg}"
        );
    }

    #[test]
    fn test_ollama_error_non_json_body_falls_back_to_raw() {
        let err = ollama_error(
            "vision",
            reqwest::StatusCode::BAD_GATEWAY,
            "Bad Gateway",
            "llava:latest",
        );
        let msg = err.to_string();
        assert!(msg.contains("Bad Gateway"), "expected raw body in: {msg}");
        assert!(msg.contains("502"), "expected status code in: {msg}");
    }

    #[tokio::test]
    async fn test_embed_returns_friendly_error_when_model_not_found() {
        let base_url = one_shot_server(
            "404 Not Found",
            r#"{"error":"model 'nomic-embed-text:latest' not found, try pulling it first"}"#,
        );
        let embedder = Embedder::new(base_url, "nomic-embed-text:latest".to_string());
        let err = embedder.embed("hello").await.unwrap_err().to_string();
        assert!(
            err.contains("is not installed"),
            "expected friendly message in: {err}"
        );
        assert!(
            err.contains("ollama pull"),
            "expected pull command in: {err}"
        );
    }

    #[tokio::test]
    async fn test_generate_answer_returns_friendly_error_when_model_not_found() {
        let base_url = one_shot_server(
            "404 Not Found",
            r#"{"error":"model 'llama3:latest' not found, try pulling it first"}"#,
        );
        let generator = Generator::new(base_url, "llama3:latest".to_string());
        let err = generator
            .generate_answer(&[], "test", 64)
            .await
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("is not installed"),
            "expected friendly message in: {err}"
        );
        assert!(
            err.contains("ollama pull"),
            "expected pull command in: {err}"
        );
    }

    #[test]
    fn test_build_user_prompt_includes_all_contexts() {
        let prompt = build_user_prompt(&["First".to_string(), "Second".to_string()], "Why?");
        assert!(prompt.contains("Context:"));
        assert!(prompt.contains("First\n\nSecond"));
        assert!(prompt.ends_with("Question: Why?"));
    }

    #[tokio::test]
    async fn test_list_models_success() {
        let base_url = one_shot_server(
            "200 OK",
            r#"{"models":[{"name":"embed-x"},{"name":"chat-y"}]}"#,
        );

        let client = OllamaClient::new(base_url);
        let models = client.list_models().await.unwrap();
        assert_eq!(models, vec!["embed-x", "chat-y"]);
    }

    #[tokio::test]
    async fn test_embed_success() {
        let base_url = one_shot_server("200 OK", r#"{"embeddings":[[0.25,0.5,0.75]]}"#);

        let embedder = Embedder::new(base_url, "embed-model".to_string());
        let embedding = embedder.embed("hello").await.unwrap();
        assert_eq!(embedding, vec![0.25, 0.5, 0.75]);
    }

    #[tokio::test]
    async fn test_embed_errors_when_response_has_no_embedding() {
        let base_url = one_shot_server("200 OK", r#"{"embeddings":[]}"#);

        let embedder = Embedder::new(base_url, "embed-model".to_string());
        let err = embedder.embed("hello").await.unwrap_err().to_string();
        assert!(err.contains("did not include an embedding"));
    }

    #[tokio::test]
    async fn test_caption_image_success() {
        let base_url =
            one_shot_server("200 OK", r#"{"message":{"content":"A small orange cat."}}"#);
        let captioner = VisionCaptioner::new(base_url, "vision-model".to_string());
        let dir = tempfile::tempdir().unwrap();
        let image = dir.path().join("cat.png");
        std::fs::write(&image, b"not-a-real-png").unwrap();

        let caption = captioner.caption_image(&image).await.unwrap();
        assert_eq!(caption, "A small orange cat.");
    }

    #[tokio::test]
    async fn test_generate_answer_success() {
        let base_url = one_shot_server(
            "200 OK",
            r#"{"message":{"content":"A cat sits on a windowsill. [1]"}}"#,
        );
        let generator = Generator::new(base_url, "llama3.2".to_string());

        let answer = generator
            .generate_answer(
                &[String::from(
                    "[1] source=note.txt\nA cat sits on a windowsill.",
                )],
                "What animal is described?",
                64,
            )
            .await
            .expect("generate Ollama chat response");

        assert_eq!(answer, "A cat sits on a windowsill. [1]");
    }

    #[test]
    fn test_should_retry_returns_true_for_server_errors() {
        let result: Result<(reqwest::StatusCode, String)> = Ok((
            reqwest::StatusCode::INTERNAL_SERVER_ERROR,
            "error".to_string(),
        ));
        assert!(should_retry(&result));
    }

    #[test]
    fn test_should_retry_returns_false_for_client_errors() {
        let result: Result<(reqwest::StatusCode, String)> =
            Ok((reqwest::StatusCode::NOT_FOUND, "error".to_string()));
        assert!(!should_retry(&result));
    }

    #[test]
    fn test_should_retry_returns_true_for_connection_errors() {
        let result: Result<(reqwest::StatusCode, String)> =
            Err(anyhow::anyhow!("error sending request: connection refused"));
        assert!(should_retry(&result));
    }

    #[test]
    fn test_backoff_duration_doubles_each_attempt() {
        assert_eq!(backoff_duration(0), Duration::from_millis(500));
        assert_eq!(backoff_duration(1), Duration::from_millis(1000));
        assert_eq!(backoff_duration(2), Duration::from_millis(2000));
    }

    #[test]
    fn test_is_connection_error_detects_known_patterns() {
        assert!(is_connection_error(&anyhow::anyhow!("connection refused")));
        assert!(is_connection_error(&anyhow::anyhow!("connection reset by peer")));
        assert!(is_connection_error(&anyhow::anyhow!("broken pipe")));
        assert!(is_connection_error(&anyhow::anyhow!(std::io::Error::from(
            std::io::ErrorKind::ConnectionRefused
        ))));
        assert!(!is_connection_error(&anyhow::anyhow!("error sending request")));
        assert!(!is_connection_error(&anyhow::anyhow!("model not found")));
    }
}
