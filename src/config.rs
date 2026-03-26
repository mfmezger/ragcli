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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigValueSource {
    File,
    Env(&'static str),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfigSources {
    pub ollama_base_url: ConfigValueSource,
    pub models_embed: ConfigValueSource,
    pub models_chat: ConfigValueSource,
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

pub fn load_or_create_file_config(store: &Path) -> Result<Config> {
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

pub fn save_config(store: &Path, cfg: &Config) -> Result<()> {
    let raw = toml::to_string_pretty(cfg)?;
    fs::write(config_path(store), raw)?;
    Ok(())
}

pub fn load_or_create_config(store: &Path) -> Result<Config> {
    Ok(load_or_create_file_config(store)?.apply_env_overrides())
}

pub fn load_or_create_config_with_sources(store: &Path) -> Result<(Config, ConfigSources)> {
    let cfg = load_or_create_file_config(store)?;
    Ok(cfg.apply_env_overrides_with_sources())
}

pub fn resolve_model_name(_store: &Path, model: &str) -> String {
    model.to_string()
}

impl Config {
    pub fn apply_env_overrides(self) -> Self {
        self.apply_env_overrides_with_sources().0
    }

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
    use std::sync::{Mutex, OnceLock};

    fn env_lock() -> &'static Mutex<()> {
        static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        ENV_LOCK.get_or_init(|| Mutex::new(()))
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
}
