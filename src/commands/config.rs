use crate::config::{
    self, ensure_store_layout, load_or_create_config_with_sources, load_or_create_file_config,
    save_config, store_dir, Config, ConfigSources,
};
use crate::ui::{self, Panel};
use anyhow::Result;
use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct ConfigShowReport {
    pub store: String,
    pub config_path: String,
    pub config: Config,
    pub sources: ConfigSources,
    pub active_overrides: Vec<String>,
}

pub async fn show(name: Option<&str>, json: bool) -> Result<()> {
    let report = build_show_report(name)?;
    if json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        print_show_human(&report)?;
    }
    Ok(())
}

fn build_show_report(name: Option<&str>) -> Result<ConfigShowReport> {
    let store = store_dir(name)?;
    ensure_store_layout(&store)?;
    let (cfg, sources) = load_or_create_config_with_sources(&store)?;
    let active_overrides = sources.overrides();

    Ok(ConfigShowReport {
        store: store.display().to_string(),
        config_path: config::config_path(&store).display().to_string(),
        config: cfg,
        sources,
        active_overrides,
    })
}

fn print_show_human(report: &ConfigShowReport) -> Result<()> {
    ui::command_header("ragcli config show", "");

    let mut location = Panel::new("Config Location");
    location.kv("store", &report.store, 6);
    location.kv("file", &report.config_path, 6);
    location.render();

    println!();
    let mut config = Panel::new("Config");
    for line in toml::to_string_pretty(&report.config)?.lines() {
        config.push(format!("  {line}"));
    }
    config.render();

    if !report.active_overrides.is_empty() {
        println!();
        let mut overrides = Panel::new("Active Environment Overrides");
        for override_entry in &report.active_overrides {
            overrides.prose("env", override_entry, 4);
        }
        overrides.render();
    }

    Ok(())
}

pub async fn set(name: Option<&str>, key: String, value: String) -> Result<()> {
    let store = store_dir(name)?;
    ensure_store_layout(&store)?;
    let mut cfg = load_or_create_file_config(&store)?;
    cfg.set_path(&key, &value)?;
    save_config(&store, &cfg)?;

    let mut panel = Panel::new("Config Updated");
    panel.kv("file", config::config_path(&store).display().to_string(), 6);
    panel.kv("key", key, 6);
    panel.kv("value", value, 6);
    panel.kv("status", ui::ok("saved"), 6);
    panel.render();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::load_or_create_file_config;
    use crate::test_support::with_test_env;

    #[tokio::test(flavor = "current_thread")]
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
            show(Some("test-store"), false).await.unwrap();

            let store = store_dir(Some("test-store")).unwrap();
            let cfg = load_or_create_file_config(&store).unwrap();
            assert_eq!(cfg.models.chat, "chat-z");
        })
        .await;
    }

    #[tokio::test(flavor = "current_thread")]
    async fn test_build_show_report_serializes_to_json() {
        let dir = tempfile::tempdir().unwrap();
        with_test_env(dir.path(), None, || async {
            let report = build_show_report(Some("json-store")).unwrap();
            let json = serde_json::to_string(&report).unwrap();
            assert!(json.contains("\"config\""));
            assert!(json.contains("\"active_overrides\""));
        })
        .await;
    }
}
