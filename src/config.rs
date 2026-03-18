use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

const DEFAULT_STORE_NAME: &str = "default";

#[derive(Debug, Serialize, Deserialize)]
pub struct Config {
    #[serde(default)]
    pub models: ModelsConfig,
    #[serde(default)]
    pub ollama: OllamaConfig,
    #[serde(default)]
    pub chunk: ChunkConfig,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct ModelsConfig {
    pub embed: String,
    pub chat: String,
    pub vision: String,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct OllamaConfig {
    pub base_url: String,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct ChunkConfig {
    pub size: usize,
    pub overlap: usize,
}

impl Default for ModelsConfig {
    fn default() -> Self {
        Self {
            embed: "nomic-embed-text-v2-moe:latest".to_string(),
            chat: "qwen3.5:4b".to_string(),
            vision: "qwen3.5:4b".to_string(),
        }
    }
}

impl Default for OllamaConfig {
    fn default() -> Self {
        Self {
            base_url: "http://localhost:11434".to_string(),
        }
    }
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            size: 1000,
            overlap: 200,
        }
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            models: ModelsConfig::default(),
            ollama: OllamaConfig::default(),
            chunk: ChunkConfig::default(),
        }
    }
}

pub fn base_dir() -> Result<PathBuf> {
    let xdg_dirs = xdg::BaseDirectories::with_prefix("ragcli")?;
    Ok(xdg_dirs.get_config_home())
}

pub fn store_dir(name: Option<&str>) -> Result<PathBuf> {
    let store_name = name.unwrap_or(DEFAULT_STORE_NAME);
    Ok(base_dir()?.join(store_name))
}

pub fn config_path(store: &Path) -> PathBuf {
    store.join("config.toml")
}

pub fn ensure_store_layout(store: &Path) -> Result<()> {
    fs::create_dir_all(store.join("lancedb"))?;
    fs::create_dir_all(store.join("meta"))?;
    fs::create_dir_all(store.join("cache"))?;
    fs::create_dir_all(store.join("models"))?;
    Ok(())
}

pub fn load_or_create_config(store: &Path) -> Result<Config> {
    let path = config_path(store);
    if path.exists() {
        let raw = fs::read_to_string(&path)?;
        let cfg: Config = toml::from_str(&raw)?;
        return Ok(cfg);
    }

    let cfg = Config::default();
    let raw = toml::to_string_pretty(&cfg)?;
    fs::write(&path, raw)?;
    Ok(cfg)
}

pub fn resolve_model_name(_store: &Path, model: &str) -> String {
    model.to_string()
}

pub fn normalize_chunk_settings(size: usize, overlap: usize) -> (usize, usize) {
    if size == 0 {
        return (1, 0);
    }
    let mut overlap = overlap.min(size.saturating_sub(1));
    if overlap >= size {
        overlap = size / 2;
    }
    (size, overlap)
}

pub fn status(exists: bool) -> &'static str {
    if exists {
        "exists"
    } else {
        "missing"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_create_and_reload() {
        let dir = tempfile::tempdir().unwrap();
        let store = dir.path().join("store");
        fs::create_dir_all(&store).unwrap();

        let cfg = load_or_create_config(&store).unwrap();
        assert_eq!(cfg.models.embed, "nomic-embed-text-v2-moe:latest");
        assert_eq!(cfg.models.chat, "qwen3.5:4b");
        assert_eq!(cfg.models.vision, "qwen3.5:4b");
        assert!(config_path(&store).exists());

        let cfg2 = load_or_create_config(&store).unwrap();
        assert_eq!(cfg2.models.embed, "nomic-embed-text-v2-moe:latest");
        assert_eq!(cfg2.chunk.size, 1000);
    }

    #[test]
    fn test_resolve_model_name() {
        let dir = tempfile::tempdir().unwrap();
        let store = dir.path().join("store");
        fs::create_dir_all(store.join("models")).unwrap();

        let model = resolve_model_name(&store, "nomic-embed-text");
        assert_eq!(model, "nomic-embed-text");
    }

    #[test]
    fn test_normalize_chunk_settings() {
        let (size, overlap) = normalize_chunk_settings(0, 10);
        assert_eq!(size, 1);
        assert_eq!(overlap, 0);

        let (size2, overlap2) = normalize_chunk_settings(10, 50);
        assert_eq!(size2, 10);
        assert_eq!(overlap2, 9);
    }
}
