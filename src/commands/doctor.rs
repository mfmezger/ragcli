use crate::config::{self, ensure_store_layout, load_or_create_config, status, store_dir};
use crate::models::OllamaClient;
use crate::store;
use anyhow::{Context, Result};
use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

pub async fn run(name: Option<&str>) -> Result<()> {
    let store = store_dir(name)?;
    ensure_store_layout(&store)?;
    let cfg = load_or_create_config(&store)?;
    let base = config::base_dir()?;
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system time before unix epoch")?
        .as_secs();

    println!("Doctor report");
    println!("  time: {}", now);
    println!("  base: {} ({})", base.display(), status(base.exists()));
    println!("  store: {} ({})", store.display(), status(store.exists()));
    println!(
        "  config: {} ({})",
        config::config_path(&store).display(),
        status(config::config_path(&store).exists())
    );
    println!("  ollama url: {}", cfg.ollama.base_url);
    println!("  embed model: {}", cfg.models.embed);
    println!("  chat model: {}", cfg.models.chat);
    println!("  vision model: {}", cfg.models.vision);

    let ollama = OllamaClient::new(cfg.ollama.base_url.clone());
    match ollama.list_models().await {
        Ok(models) => {
            println!("  ollama: reachable");
            println!(
                "  embed model installed: {}",
                status(models.iter().any(|model| model == &cfg.models.embed))
            );
            println!(
                "  chat model installed: {}",
                status(models.iter().any(|model| model == &cfg.models.chat))
            );
            println!(
                "  vision model installed: {}",
                status(models.iter().any(|model| model == &cfg.models.vision))
            );
        }
        Err(err) => {
            println!("  ollama: unreachable ({})", err);
        }
    }

    let metadata_path = store::metadata_path(&store);
    println!(
        "  metadata: {} ({})",
        metadata_path.display(),
        status(metadata_path.exists())
    );
    if metadata_path.exists() {
        let metadata = fs::read_to_string(&metadata_path)?;
        println!("  metadata summary: {}", metadata.replace('\n', " "));
    }

    for sub in ["lancedb", "meta", "cache", "models"] {
        let path = store.join(sub);
        println!("  {}: {} ({})", sub, path.display(), status(path.exists()));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{sequential_json_server, with_test_env};

    #[tokio::test(flavor = "current_thread")]
    async fn test_run_succeeds_on_empty_store() {
        let dir = tempfile::tempdir().unwrap();
        with_test_env(dir.path(), None, || async {
            run(Some("empty")).await.unwrap();
        })
        .await;
    }

    #[tokio::test(flavor = "current_thread")]
    async fn test_run_succeeds_with_reachable_mock_ollama() {
        let dir = tempfile::tempdir().unwrap();
        let server = sequential_json_server(vec![
            r#"{"models":[{"name":"nomic-embed-text-v2-moe:latest"},{"name":"qwen3.5:4b"}]}"#,
        ]);

        with_test_env(dir.path(), Some(&server), || async {
            run(Some("reachable")).await.unwrap();
        })
        .await;
    }
}
