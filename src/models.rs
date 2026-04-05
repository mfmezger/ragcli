//! Clients for Ollama embedding, chat, and vision APIs.

use anyhow::{Context, Result};
use base64::Engine;
use reqwest::header::CONTENT_TYPE;
use reqwest::Client;
use serde::Deserialize;
use std::path::Path;
use std::time::Duration;
#[cfg(test)]
use std::path::PathBuf;
#[cfg(test)]
use reqwest_middleware::ClientWithMiddleware;

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
    #[cfg(test)]
    Middleware(ClientWithMiddleware),
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
        let url = format!("{}/api/tags", self.base_url.trim_end_matches('/'));
        let (status, body) = self.client.get_text(url).await.context("call Ollama tags API")?;
        if !status.is_success() {
            anyhow::bail!("Ollama tags API error: {} - {}", status, body);
        }

        let parsed: TagsResponse =
            serde_json::from_str(&body).context("parse Ollama tags response")?;
        Ok(parsed.models.into_iter().map(|model| model.name).collect())
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
            anyhow::bail!("Ollama embed API error: {} - {}", status, body);
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

    #[cfg(test)]
    fn new_with_client(base_url: String, model: String, client: ClientWithMiddleware) -> Self {
        Self {
            client: HttpClient::Middleware(client),
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
        let url = format!("{}/api/chat", self.base_url.trim_end_matches('/'));
        let system_prompt = "You are a helpful assistant. Use the provided context to answer the question. If the answer is not in the context, say you don't know. Respond with only the final answer and no chain-of-thought.";
        let user_prompt = build_user_prompt(contexts, question);

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
            anyhow::bail!("Ollama chat API error: {} - {}", status, body);
        }

        let parsed: ChatResponse =
            serde_json::from_str(&body).context("parse Ollama chat response")?;
        Ok(parsed.message.content)
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
        let bytes = std::fs::read(image_path)
            .with_context(|| format!("read image: {}", image_path.display()))?;
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
            anyhow::bail!("Ollama vision API error: {} - {}", status, body);
        }

        let parsed: ChatResponse =
            serde_json::from_str(&body).context("parse Ollama vision response")?;
        Ok(parsed.message.content)
    }
}

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
        let response = match self {
            Self::Reqwest(client) => client
                .get(url)
                .send()
                .await
                .context("send HTTP GET request")?,
            #[cfg(test)]
            Self::Middleware(client) => client
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
        let body = body.to_string();
        let response = match self {
            Self::Reqwest(client) => client
                .post(url)
                .header(CONTENT_TYPE, "application/json")
                .body(body.clone())
                .send()
                .await
                .context("send HTTP POST request")?,
            #[cfg(test)]
            Self::Middleware(client) => client
                .post(url)
                .header(CONTENT_TYPE, "application/json")
                .body(body)
                .send()
                .await
                .context("send HTTP POST request")?,
        };

        let status = response.status();
        let body = response.text().await.context("read HTTP response body")?;
        Ok((status, body))
    }
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
    use reqwest_middleware::ClientBuilder;
    use reqwest_vcr::{VCRMiddleware, VCRMode};
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::thread;

    fn replay_client(cassette_name: &str) -> ClientWithMiddleware {
        let cassette = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("cassettes")
            .join(cassette_name);
        let middleware = VCRMiddleware::try_from(cassette)
            .expect("load VCR cassette")
            .with_mode(VCRMode::Replay)
            .with_modify_request(|request| {
                request.headers.clear();
            });

        ClientBuilder::new(reqwest::Client::new())
            .with(middleware)
            .build()
    }

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
            stream.write_all(response.as_bytes()).expect("write response");
        });

        format!("http://{}", addr)
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
        let base_url = one_shot_server(
            "200 OK",
            r#"{"embeddings":[[0.25,0.5,0.75]]}"#,
        );

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
        let base_url = one_shot_server(
            "200 OK",
            r#"{"message":{"content":"A small orange cat."}}"#,
        );
        let captioner = VisionCaptioner::new(base_url, "vision-model".to_string());
        let dir = tempfile::tempdir().unwrap();
        let image = dir.path().join("cat.png");
        std::fs::write(&image, b"not-a-real-png").unwrap();

        let caption = captioner.caption_image(&image).await.unwrap();
        assert_eq!(caption, "A small orange cat.");
    }

    #[tokio::test]
    async fn test_generate_answer_replays_from_cassette() {
        let generator = Generator::new_with_client(
            "http://ollama.test".to_string(),
            "llama3.2".to_string(),
            replay_client("ollama-chat-success.vcr.json"),
        );

        let answer = generator
            .generate_answer(
                &[String::from("A cat sits on a windowsill.")],
                "What animal is described?",
                64,
            )
            .await
            .expect("replay Ollama chat response");

        assert_eq!(answer, "A cat sits on a windowsill.");
    }
}
