//! Configuration loading, persistence, and environment overrides.

use crate::fsutil::write_atomic;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const DEFAULT_STORE_NAME: &str = "default";
pub const ENV_OLLAMA_URL: &str = "RAGCLI_OLLAMA_URL";
pub const ENV_EMBED_MODEL: &str = "RAGCLI_EMBED_MODEL";
pub const ENV_CHAT_MODEL: &str = "RAGCLI_CHAT_MODEL";
pub const ENV_VISION_MODEL: &str = "RAGCLI_VISION_MODEL";

/// Runtime configuration for a single store.
#[derive(Debug, Serialize, Deserialize)]
pub struct Config {
    /// Model names used for embedding, chat, and vision tasks.
    #[serde(default)]
    pub models: ModelsConfig,
    /// Ollama connection settings.
    #[serde(default)]
    pub ollama: OllamaConfig,
    /// Chunking settings used during indexing.
    #[serde(default)]
    pub chunk: ChunkConfig,
}

/// Model configuration values.
#[derive(Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct ModelsConfig {
    /// Embedding model name.
    pub embed: String,
    /// Chat model name.
    pub chat: String,
    /// Vision-capable model name.
    pub vision: String,
}

/// Ollama connection settings.
#[derive(Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct OllamaConfig {
    /// Base URL for the Ollama server.
    pub base_url: String,
}

/// Chunking settings used during indexing.
#[derive(Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct ChunkConfig {
    /// Target chunk size in characters.
    pub size: usize,
    /// Overlap between adjacent chunks in characters.
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

/// Indicates where a resolved config value came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum ConfigValueSource {
    /// The value came from the store config file.
    File,
    /// The value came from the named environment variable.
    Env(&'static str),
}

/// Tracks the source of config values after environment overrides are applied.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ConfigSources {
    /// Source of `ollama.base_url`.
    pub ollama_base_url: ConfigValueSource,
    /// Source of `models.embed`.
    pub models_embed: ConfigValueSource,
    /// Source of `models.chat`.
    pub models_chat: ConfigValueSource,
    /// Source of `models.vision`.
    pub models_vision: ConfigValueSource,
}

impl Default for ConfigSources {
    fn default() -> Self {
        Self {
            ollama_base_url: ConfigValueSource::File,
            models_embed: ConfigValueSource::File,
            models_chat: ConfigValueSource::File,
            models_vision: ConfigValueSource::File,
        }
    }
}

impl ConfigSources {
    /// Returns human-readable descriptions of active environment overrides.
    pub fn overrides(&self) -> Vec<String> {
        let mut overrides = Vec::new();
        if let ConfigValueSource::Env(var) = self.ollama_base_url {
            overrides.push(format!("ollama.base_url <- {}", var));
        }
        if let ConfigValueSource::Env(var) = self.models_embed {
            overrides.push(format!("models.embed <- {}", var));
        }
        if let ConfigValueSource::Env(var) = self.models_chat {
            overrides.push(format!("models.chat <- {}", var));
        }
        if let ConfigValueSource::Env(var) = self.models_vision {
            overrides.push(format!("models.vision <- {}", var));
        }
        overrides
    }
}

/// Returns the base configuration directory for `ragcli`.
pub fn base_dir() -> Result<PathBuf> {
    let xdg_dirs = xdg::BaseDirectories::with_prefix("ragcli")?;
    Ok(xdg_dirs.get_config_home())
}

/// Returns the directory for a named store, or the default store when omitted.
pub fn store_dir(name: Option<&str>) -> Result<PathBuf> {
    let store_name = name.unwrap_or(DEFAULT_STORE_NAME);
    Ok(base_dir()?.join(store_name))
}

/// Returns the path to the store's `config.toml` file.
pub fn config_path(store: &Path) -> PathBuf {
    store.join("config.toml")
}

/// Ensures the on-disk directory layout for a store exists.
pub fn ensure_store_layout(store: &Path) -> Result<()> {
    fs::create_dir_all(store.join("lancedb"))?;
    fs::create_dir_all(store.join("meta"))?;
    fs::create_dir_all(store.join("cache"))?;
    fs::create_dir_all(store.join("models"))?;
    Ok(())
}

/// Loads the store config from disk, creating a default file when needed.
pub fn load_or_create_file_config(store: &Path) -> Result<Config> {
    let path = config_path(store);
    if path.exists() {
        let raw = fs::read_to_string(&path)?;
        let cfg: Config = toml::from_str(&raw)?;
        return Ok(cfg);
    }

    let cfg = Config::default();
    let raw = toml::to_string_pretty(&cfg)?;
    write_atomic(&path, &raw)?;
    Ok(cfg)
}

/// Writes a config file for the given store.
pub fn save_config(store: &Path, cfg: &Config) -> Result<()> {
    let raw = toml::to_string_pretty(cfg)?;
    write_atomic(&config_path(store), &raw)?;
    Ok(())
}

/// Loads the effective config after applying environment overrides.
pub fn load_or_create_config(store: &Path) -> Result<Config> {
    Ok(load_or_create_file_config(store)?.apply_env_overrides())
}

/// Loads the effective config together with the source of each resolved value.
pub fn load_or_create_config_with_sources(store: &Path) -> Result<(Config, ConfigSources)> {
    let cfg = load_or_create_file_config(store)?;
    Ok(cfg.apply_env_overrides_with_sources())
}

/// Resolves a model name for runtime use.
///
/// This currently returns the configured model name unchanged.
pub fn resolve_model_name(_store: &Path, model: &str) -> String {
    model.to_string()
}

impl Config {
    /// Applies environment variable overrides to the config.
    pub fn apply_env_overrides(self) -> Self {
        self.apply_env_overrides_with_sources().0
    }

    /// Applies environment variable overrides and returns the source of each value.
    pub fn apply_env_overrides_with_sources(mut self) -> (Self, ConfigSources) {
        let mut sources = ConfigSources::default();

        if let Ok(value) = env::var(ENV_OLLAMA_URL) {
            self.ollama.base_url = value;
            sources.ollama_base_url = ConfigValueSource::Env(ENV_OLLAMA_URL);
        }
        if let Ok(value) = env::var(ENV_EMBED_MODEL) {
            self.models.embed = value;
            sources.models_embed = ConfigValueSource::Env(ENV_EMBED_MODEL);
        }
        if let Ok(value) = env::var(ENV_CHAT_MODEL) {
            self.models.chat = value;
            sources.models_chat = ConfigValueSource::Env(ENV_CHAT_MODEL);
        }
        if let Ok(value) = env::var(ENV_VISION_MODEL) {
            self.models.vision = value;
            sources.models_vision = ConfigValueSource::Env(ENV_VISION_MODEL);
        }

        (self, sources)
    }

    /// Updates a supported config key from a dotted path and string value.
    pub fn set_path(&mut self, key: &str, value: &str) -> Result<()> {
        match key {
            "ollama.base_url" => self.ollama.base_url = value.to_string(),
            "models.embed" => self.models.embed = value.to_string(),
            "models.chat" => self.models.chat = value.to_string(),
            "models.vision" => self.models.vision = value.to_string(),
            "chunk.size" => self.chunk.size = value.parse()?,
            "chunk.overlap" => self.chunk.overlap = value.parse()?,
            _ => anyhow::bail!(
                "unknown config key: {} (expected one of ollama.base_url, models.embed, models.chat, models.vision, chunk.size, chunk.overlap)",
                key
            ),
        }
        Ok(())
    }
}

/// Normalizes chunk settings so they remain valid for indexing.
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

/// Formats a simple existence status string.
pub fn status(exists: bool) -> &'static str {
    if exists {
        "exists"
    } else {
        "missing"
    }
}

#[cfg(test)]
pub(crate) fn test_env_lock() -> &'static std::sync::Mutex<()> {
    use std::sync::{Mutex, OnceLock};

    static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    ENV_LOCK.get_or_init(|| Mutex::new(()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env_lock() -> &'static std::sync::Mutex<()> {
        test_env_lock()
    }

    #[test]
    fn test_config_create_and_reload() {
        let dir = tempfile::tempdir().unwrap();
        let store = dir.path().join("store");
        fs::create_dir_all(&store).unwrap();

        let cfg = load_or_create_file_config(&store).unwrap();
        assert_eq!(cfg.models.embed, "nomic-embed-text-v2-moe:latest");
        assert_eq!(cfg.models.chat, "qwen3.5:4b");
        assert_eq!(cfg.models.vision, "qwen3.5:4b");
        assert!(config_path(&store).exists());

        let cfg2 = load_or_create_file_config(&store).unwrap();
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

    #[test]
    fn test_set_path_updates_supported_keys() {
        let mut cfg = Config::default();

        cfg.set_path("ollama.base_url", "http://ollama:11434")
            .unwrap();
        cfg.set_path("models.embed", "embed-x").unwrap();
        cfg.set_path("models.chat", "chat-x").unwrap();
        cfg.set_path("models.vision", "vision-x").unwrap();
        cfg.set_path("chunk.size", "512").unwrap();
        cfg.set_path("chunk.overlap", "64").unwrap();

        assert_eq!(cfg.ollama.base_url, "http://ollama:11434");
        assert_eq!(cfg.models.embed, "embed-x");
        assert_eq!(cfg.models.chat, "chat-x");
        assert_eq!(cfg.models.vision, "vision-x");
        assert_eq!(cfg.chunk.size, 512);
        assert_eq!(cfg.chunk.overlap, 64);
    }

    #[test]
    fn test_apply_env_overrides_with_sources() {
        let _guard = env_lock().lock().unwrap();
        let dir = tempfile::tempdir().unwrap();
        let store = dir.path().join("store");
        fs::create_dir_all(&store).unwrap();
        let path = config_path(&store);
        fs::write(
            &path,
            r#"[ollama]
base_url = "http://localhost:11434"

[models]
embed = "embed-file"
chat = "chat-file"
vision = "vision-file"

[chunk]
size = 1000
overlap = 200
"#,
        )
        .unwrap();

        let previous_ollama_url = env::var_os(ENV_OLLAMA_URL);
        let previous_embed_model = env::var_os(ENV_EMBED_MODEL);
        let previous_chat_model = env::var_os(ENV_CHAT_MODEL);
        let previous_vision_model = env::var_os(ENV_VISION_MODEL);

        unsafe {
            env::set_var(ENV_OLLAMA_URL, "http://remote:11434");
            env::set_var(ENV_EMBED_MODEL, "embed-env");
            env::set_var(ENV_CHAT_MODEL, "chat-env");
            env::set_var(ENV_VISION_MODEL, "vision-env");
        }

        let (cfg, sources) = load_or_create_config_with_sources(&store).unwrap();

        assert_eq!(cfg.ollama.base_url, "http://remote:11434");
        assert_eq!(cfg.models.embed, "embed-env");
        assert_eq!(cfg.models.chat, "chat-env");
        assert_eq!(cfg.models.vision, "vision-env");
        assert_eq!(
            sources.ollama_base_url,
            ConfigValueSource::Env(ENV_OLLAMA_URL)
        );
        assert_eq!(
            sources.models_embed,
            ConfigValueSource::Env(ENV_EMBED_MODEL)
        );
        assert_eq!(sources.models_chat, ConfigValueSource::Env(ENV_CHAT_MODEL));
        assert_eq!(
            sources.models_vision,
            ConfigValueSource::Env(ENV_VISION_MODEL)
        );

        unsafe {
            match previous_ollama_url {
                Some(value) => env::set_var(ENV_OLLAMA_URL, value),
                None => env::remove_var(ENV_OLLAMA_URL),
            }
            match previous_embed_model {
                Some(value) => env::set_var(ENV_EMBED_MODEL, value),
                None => env::remove_var(ENV_EMBED_MODEL),
            }
            match previous_chat_model {
                Some(value) => env::set_var(ENV_CHAT_MODEL, value),
                None => env::remove_var(ENV_CHAT_MODEL),
            }
            match previous_vision_model {
                Some(value) => env::set_var(ENV_VISION_MODEL, value),
                None => env::remove_var(ENV_VISION_MODEL),
            }
        }
    }

    #[test]
    fn test_config_sources_overrides_formats_only_env_values() {
        let sources = ConfigSources {
            ollama_base_url: ConfigValueSource::Env(ENV_OLLAMA_URL),
            models_embed: ConfigValueSource::File,
            models_chat: ConfigValueSource::Env(ENV_CHAT_MODEL),
            models_vision: ConfigValueSource::File,
        };

        assert_eq!(
            sources.overrides(),
            vec![
                format!("ollama.base_url <- {}", ENV_OLLAMA_URL),
                format!("models.chat <- {}", ENV_CHAT_MODEL),
            ]
        );
    }

    #[test]
    fn test_ensure_store_layout_creates_expected_directories() {
        let dir = tempfile::tempdir().unwrap();
        let store = dir.path().join("store");

        ensure_store_layout(&store).unwrap();

        assert!(store.join("lancedb").exists());
        assert!(store.join("meta").exists());
        assert!(store.join("cache").exists());
        assert!(store.join("models").exists());
    }

    #[test]
    fn test_save_config_persists_updates() {
        let dir = tempfile::tempdir().unwrap();
        let store = dir.path().join("store");
        fs::create_dir_all(&store).unwrap();

        let mut cfg = Config::default();
        cfg.models.chat = "chat-updated".to_string();
        cfg.chunk.size = 256;
        save_config(&store, &cfg).unwrap();

        let reloaded = load_or_create_file_config(&store).unwrap();
        assert_eq!(reloaded.models.chat, "chat-updated");
        assert_eq!(reloaded.chunk.size, 256);
    }

    #[test]
    fn test_set_path_rejects_unknown_key() {
        let mut cfg = Config::default();
        let err = cfg.set_path("models.unknown", "x").unwrap_err().to_string();
        assert!(err.contains("unknown config key"));
    }

    #[test]
    fn test_status_reports_exists_and_missing() {
        assert_eq!(status(true), "exists");
        assert_eq!(status(false), "missing");
    }
}
