use crate::config::{
    self, ensure_store_layout, load_or_create_config, status, store_dir, STORE_SUBDIRECTORIES,
};
use crate::models::OllamaClient;
use crate::store;
use anyhow::{Context, Result};
use serde::Serialize;
use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};
use time::format_description::well_known::Rfc3339;
use time::OffsetDateTime;

#[derive(Debug, Serialize)]
pub struct PathStatusReport {
    pub path: String,
    pub status: &'static str,
}

#[derive(Debug, Serialize)]
pub struct NamedPathStatusReport {
    pub name: String,
    pub path: String,
    pub status: &'static str,
}

#[derive(Debug, Serialize)]
pub struct ModelInstallReport {
    pub embed_model_installed: bool,
    pub chat_model_installed: bool,
    pub vision_model_installed: bool,
}

#[derive(Debug, Serialize)]
pub struct DoctorReport {
    pub time: u64,
    pub base: PathStatusReport,
    pub store: PathStatusReport,
    pub config: PathStatusReport,
    pub ollama_url: String,
    pub embed_model: String,
    pub chat_model: String,
    pub vision_model: String,
    pub ollama_reachable: bool,
    pub ollama_error: Option<String>,
    pub installed_models: Option<ModelInstallReport>,
    pub metadata: PathStatusReport,
    pub metadata_summary: Option<String>,
    pub metadata_error: Option<String>,
    pub subdirectories: Vec<NamedPathStatusReport>,
}

pub async fn run(name: Option<&str>, json: bool) -> Result<()> {
    let report = build_report(name).await?;
    if json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        print_human(&report);
    }
    Ok(())
}

async fn build_report(name: Option<&str>) -> Result<DoctorReport> {
    let store = store_dir(name)?;
    ensure_store_layout(&store)?;
    let cfg = load_or_create_config(&store)?;
    let base = config::base_dir()?;
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system time before unix epoch")?
        .as_secs();

    let ollama = OllamaClient::new(cfg.ollama.base_url.clone());
    let (ollama_reachable, ollama_error, installed_models) = match ollama.list_models().await {
        Ok(models) => (
            true,
            None,
            Some(ModelInstallReport {
                embed_model_installed: models.iter().any(|model| model == &cfg.models.embed),
                chat_model_installed: models.iter().any(|model| model == &cfg.models.chat),
                vision_model_installed: models.iter().any(|model| model == &cfg.models.vision),
            }),
        ),
        Err(err) => (false, Some(err.to_string()), None),
    };

    let metadata_path = store::metadata_path(&store);
    let (metadata_summary, metadata_error) = if metadata_path.exists() {
        match fs::read_to_string(&metadata_path) {
            Ok(contents) => (Some(contents.replace('\n', " ")), None),
            Err(err) => (None, Some(err.to_string())),
        }
    } else {
        (None, None)
    };

    let subdirectories = STORE_SUBDIRECTORIES
        .into_iter()
        .map(|sub| {
            let path = store.join(sub);
            NamedPathStatusReport {
                name: sub.to_string(),
                path: path.display().to_string(),
                status: status(path.exists()),
            }
        })
        .collect();

    Ok(DoctorReport {
        time: now,
        base: PathStatusReport {
            path: base.display().to_string(),
            status: status(base.exists()),
        },
        store: PathStatusReport {
            path: store.display().to_string(),
            status: status(store.exists()),
        },
        config: PathStatusReport {
            path: config::config_path(&store).display().to_string(),
            status: status(config::config_path(&store).exists()),
        },
        ollama_url: cfg.ollama.base_url,
        embed_model: cfg.models.embed,
        chat_model: cfg.models.chat,
        vision_model: cfg.models.vision,
        ollama_reachable,
        ollama_error,
        installed_models,
        metadata: PathStatusReport {
            path: metadata_path.display().to_string(),
            status: status(metadata_path.exists()),
        },
        metadata_summary,
        metadata_error,
        subdirectories,
    })
}

fn print_human(report: &DoctorReport) {
    println!("Doctor report");
    println!("  time: {}", format_unix_timestamp(report.time));
    println!("  base: {} ({})", report.base.path, report.base.status);
    println!("  store: {} ({})", report.store.path, report.store.status);
    println!(
        "  config: {} ({})",
        report.config.path, report.config.status
    );
    println!("  ollama url: {}", report.ollama_url);
    println!("  embed model: {}", report.embed_model);
    println!("  chat model: {}", report.chat_model);
    println!("  vision model: {}", report.vision_model);

    if report.ollama_reachable {
        println!("  ollama: reachable");
        if let Some(installed_models) = &report.installed_models {
            println!(
                "  embed model installed: {}",
                status(installed_models.embed_model_installed)
            );
            println!(
                "  chat model installed: {}",
                status(installed_models.chat_model_installed)
            );
            println!(
                "  vision model installed: {}",
                status(installed_models.vision_model_installed)
            );
        }
    } else if let Some(err) = &report.ollama_error {
        eprintln!("  ollama: unreachable ({})", err);
    }

    println!(
        "  metadata: {} ({})",
        report.metadata.path, report.metadata.status
    );
    if let Some(metadata_summary) = &report.metadata_summary {
        println!("  metadata summary: {}", metadata_summary);
    }
    if let Some(metadata_error) = &report.metadata_error {
        eprintln!("  metadata error: {}", metadata_error);
    }

    for subdirectory in &report.subdirectories {
        println!(
            "  {}: {} ({})",
            subdirectory.name, subdirectory.path, subdirectory.status
        );
    }
}

fn format_unix_timestamp(timestamp: u64) -> String {
    match OffsetDateTime::from_unix_timestamp(timestamp as i64)
        .ok()
        .and_then(|time| time.format(&Rfc3339).ok())
    {
        Some(formatted) => formatted,
        None => timestamp.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{sequential_json_server, with_test_env};

    #[test]
    fn test_format_unix_timestamp_uses_rfc3339_when_possible() {
        assert_eq!(format_unix_timestamp(0), "1970-01-01T00:00:00Z");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn test_build_report_succeeds_on_empty_store() {
        let dir = tempfile::tempdir().unwrap();
        with_test_env(dir.path(), None, || async {
            let report = build_report(Some("empty")).await.unwrap();
            assert_eq!(report.store.status, "exists");
            assert!(serde_json::to_string(&report)
                .unwrap()
                .contains("\"ollama_reachable\""));
        })
        .await;
    }

    #[tokio::test(flavor = "current_thread")]
    async fn test_build_report_succeeds_with_reachable_mock_ollama() {
        let dir = tempfile::tempdir().unwrap();
        let server = sequential_json_server(vec![
            r#"{"models":[{"name":"nomic-embed-text-v2-moe:latest"},{"name":"qwen3.5:4b"}]}"#,
        ]);

        with_test_env(dir.path(), Some(&server), || async {
            let report = build_report(Some("reachable")).await.unwrap();
            assert!(report.ollama_reachable);
            assert!(report.installed_models.is_some());
            assert_eq!(report.subdirectories[0].name, "lancedb");
        })
        .await;
    }

    #[tokio::test(flavor = "current_thread")]
    async fn test_build_report_keeps_running_when_metadata_read_fails() {
        let dir = tempfile::tempdir().unwrap();
        with_test_env(dir.path(), None, || async {
            let store = store_dir(Some("broken-metadata")).unwrap();
            ensure_store_layout(&store).unwrap();
            fs::create_dir_all(store::metadata_path(&store)).unwrap();

            let report = build_report(Some("broken-metadata")).await.unwrap();

            assert!(report.metadata_summary.is_none());
            assert!(report.metadata_error.is_some());
        })
        .await;
    }
}
