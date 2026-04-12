use crate::config::{
    self, ensure_store_layout, load_or_create_config_with_sources, load_or_create_file_config,
    save_config, store_dir,
};
use anyhow::Result;

pub async fn show(name: Option<&str>) -> Result<()> {
    let store = store_dir(name)?;
    ensure_store_layout(&store)?;
    let (cfg, sources) = load_or_create_config_with_sources(&store)?;

    println!("Store: {}", store.display());
    println!("Config: {}", config::config_path(&store).display());
    println!();
    println!("{}", toml::to_string_pretty(&cfg)?);

    let overrides = sources.overrides();
    if !overrides.is_empty() {
        println!("Active environment overrides:");
        for override_entry in overrides {
            println!("  {}", override_entry);
        }
    }

    Ok(())
}

pub async fn set(name: Option<&str>, key: String, value: String) -> Result<()> {
    let store = store_dir(name)?;
    ensure_store_layout(&store)?;
    let mut cfg = load_or_create_file_config(&store)?;
    cfg.set_path(&key, &value)?;
    save_config(&store, &cfg)?;

    println!("Updated {}", config::config_path(&store).display());
    println!("  {} = {}", key, value);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::load_or_create_file_config;
    use crate::test_support::with_test_env;

    #[tokio::test]
    async fn test_set_and_show_succeed_for_named_store() {
        let dir = tempfile::tempdir().unwrap();
        with_test_env(dir.path(), None, || async {
            set(
                Some("test-store"),
                "models.chat".to_string(),
                "chat-z".to_string(),
            )
            .await
            .unwrap();
            show(Some("test-store")).await.unwrap();

            let store = store_dir(Some("test-store")).unwrap();
            let cfg = load_or_create_file_config(&store).unwrap();
            assert_eq!(cfg.models.chat, "chat-z");
        })
        .await;
    }
}
