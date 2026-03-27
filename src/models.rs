//! Clients for Ollama embedding, chat, and vision APIs.

use anyhow::{Context, Result};
use base64::Engine;
use reqwest::Client;
use serde::Deserialize;
use std::path::Path;
use std::time::Duration;

/// Embedding client backed by Ollama's `/api/embed` endpoint.
pub struct Embedder {
    client: Client,
    base_url: String,
    model: String,
}

/// Text generation client backed by Ollama's `/api/chat` endpoint.
pub struct Generator {
    client: Client,
    base_url: String,
    model: String,
}

/// Vision captioning client that turns images into retrieval text.
pub struct VisionCaptioner {
    client: Client,
    base_url: String,
    model: String,
}

/// Minimal Ollama API client used for health checks and model discovery.
pub struct OllamaClient {
    client: Client,
    base_url: String,
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
            client: Client::builder()
                .timeout(Duration::from_secs(10))
                .build()
                .expect("build reqwest client"),
            base_url,
        }
    }

    /// Returns the installed model names reported by Ollama.
    pub async fn list_models(&self) -> Result<Vec<String>> {
        let url = format!("{}/api/tags", self.base_url.trim_end_matches('/'));
        let response = self
            .client
            .get(url)
            .send()
            .await
            .context("call Ollama tags API")?;

        let status = response.status();
        let body = response.text().await.context("read Ollama tags response")?;
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
            client: Client::builder()
                .timeout(Duration::from_secs(60))
                .build()
                .expect("build reqwest client"),
            base_url,
            model,
        }
    }

    /// Embeds a single text input and returns its vector.
    pub async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let url = format!("{}/api/embed", self.base_url.trim_end_matches('/'));
        let response = self
            .client
            .post(url)
            .json(&serde_json::json!({
                "model": self.model,
                "input": text,
            }))
            .send()
            .await
            .context("call Ollama embed API")?;

        let status = response.status();
        let body = response
            .text()
            .await
            .context("read Ollama embed response")?;
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
            client: Client::builder()
                .timeout(Duration::from_secs(180))
                .build()
                .expect("build reqwest client"),
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

        let response = self
            .client
            .post(url)
            .json(&serde_json::json!({
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
            }))
            .send()
            .await
            .context("call Ollama chat API")?;

        let status = response.status();
        let body = response.text().await.context("read Ollama chat response")?;
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
            client: Client::builder()
                .timeout(Duration::from_secs(180))
                .build()
                .expect("build reqwest client"),
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

        let response = self
            .client
            .post(url)
            .json(&serde_json::json!({
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
            }))
            .send()
            .await
            .context("call Ollama vision chat API")?;

        let status = response.status();
        let body = response
            .text()
            .await
            .context("read Ollama vision response")?;
        if !status.is_success() {
            anyhow::bail!("Ollama vision API error: {} - {}", status, body);
        }

        let parsed: ChatResponse =
            serde_json::from_str(&body).context("parse Ollama vision response")?;
        Ok(parsed.message.content)
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
