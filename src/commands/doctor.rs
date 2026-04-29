use crate::config::{
    self, ensure_store_layout, load_or_create_config, status, store_dir, STORE_SUBDIRECTORIES,
};
use crate::models::OllamaClient;
use crate::store;
use crate::telemetry::{TelemetryConfig, TelemetryStatus};
use anyhow::{Context, Result};
use console::style;
use serde::Serialize;
use std::collections::BTreeMap;
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
    pub telemetry: TelemetryStatus,
    pub telemetry_error: Option<String>,
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
    let (telemetry, telemetry_error) = match TelemetryConfig::from_env() {
        Ok(config) => (config.status(), None),
        Err(err) => (TelemetryStatus::from_env_lossy(), Some(err.to_string())),
    };
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
        telemetry,
        telemetry_error,
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
    // ── Header ────────────────────────────────────────────────────────────────
    println!(
        "{}  {}",
        style("ragcli doctor").bold().cyan(),
        style(format_unix_timestamp(report.time)).dim()
    );
    println!();

    // ── Paths ─────────────────────────────────────────────────────────────────
    section("Paths");
    path_row("base", &report.base, 8);
    path_row("store", &report.store, 8);
    path_row("config", &report.config, 8);
    path_row("metadata", &report.metadata, 8);
    if let Some(err) = &report.metadata_error {
        eprintln!("  {:<8}  {}", style("error").red().bold(), style(err).red());
    }
    if let Some(summary) = &report.metadata_summary {
        println!("  {:<8}  {}", style("summary").dim(), style(summary).dim());
    }
    println!();

    // ── Ollama ────────────────────────────────────────────────────────────────
    section("Ollama");
    kv("url", &report.ollama_url, 9);
    if report.ollama_reachable {
        kv_styled("reachable", &ok("yes"), 9);
    } else {
        kv_styled("reachable", &err("no"), 9);
        if let Some(e) = &report.ollama_error {
            eprintln!("  {:<9}  {}", style("error").red().bold(), style(e).red());
        }
    }
    println!();

    // ── Models ────────────────────────────────────────────────────────────────
    section("Models");
    match &report.installed_models {
        Some(inst) => {
            model_row("embed", &report.embed_model, inst.embed_model_installed, 6);
            model_row("chat", &report.chat_model, inst.chat_model_installed, 6);
            model_row(
                "vision",
                &report.vision_model,
                inst.vision_model_installed,
                6,
            );
        }
        None => {
            kv("embed", &report.embed_model, 6);
            kv("chat", &report.chat_model, 6);
            kv("vision", &report.vision_model, 6);
        }
    }
    println!();

    // ── Telemetry ─────────────────────────────────────────────────────────────
    section("Telemetry");
    let tel = &report.telemetry;
    if tel.enabled {
        kv_styled("enabled", &ok("yes"), 14);
    } else {
        kv_styled("enabled", &err("no"), 14);
    }
    kv("service name", &tel.service_name, 14);
    kv("protocol", &tel.protocol, 14);
    if let Some(endpoint) = &tel.endpoint {
        kv("endpoint", endpoint, 14);
    }
    if let Some(timeout_ms) = tel.timeout_ms {
        kv("timeout (ms)", &timeout_ms.to_string(), 14);
    }
    if tel.headers_configured {
        kv_styled("headers", &ok("configured"), 14);
    } else {
        kv_styled("headers", &err("not configured"), 14);
    }
    if let Some(e) = &report.telemetry_error {
        eprintln!(
            "  {:<14}  {}",
            style("config error").red().bold(),
            style(e).red()
        );
    }
    println!();

    // ── Store Layout ──────────────────────────────────────────────────────────
    section("Store Layout");
    let col = report
        .subdirectories
        .iter()
        .map(|s| s.name.len())
        .max()
        .unwrap_or(0);
    for sub in &report.subdirectories {
        let status_str = if sub.status == "exists" {
            ok("exists")
        } else {
            err("missing")
        };
        println!(
            "  {:<col$}  {}  {}",
            style(&sub.name).bold(),
            style(&sub.path).dim(),
            status_str,
            col = col
        );
    }

    print_hints(report);
}

fn print_hints(report: &DoctorReport) {
    let hints = doctor_hints(report);
    if hints.is_empty() {
        return;
    }

    println!();
    section("Hints");
    for hint in hints {
        println!("  {} {}", style("→").yellow().bold(), style(hint).yellow());
    }
}

fn doctor_hints(report: &DoctorReport) -> Vec<String> {
    let mut hints = Vec::new();

    if report.base.status != "exists"
        || report.store.status != "exists"
        || report.config.status != "exists"
        || report
            .subdirectories
            .iter()
            .any(|sub| sub.status != "exists")
    {
        hints.push(format!(
            "Store layout is incomplete. Run `ragcli doctor` again; if it stays missing, check write permissions under `{}`.",
            report.store.path
        ));
    }

    if report.metadata_error.is_some() {
        hints.push(format!(
            "Fix metadata access at `{}`; it should be a readable file, not a directory.",
            report.metadata.path
        ));
    } else if report.metadata.status != "exists" {
        hints.push(
            "No indexed metadata found yet. Run `ragcli index <path>` before querying.".to_string(),
        );
    }

    if report.ollama_reachable {
        if let Some(installed_models) = &report.installed_models {
            hints.extend(model_install_hints(report, installed_models));
        }
    } else {
        hints.push(format!(
            "Start Ollama with `ollama serve`, or set the correct URL with `RAGCLI_OLLAMA_URL` / `ragcli config set ollama.base_url <url>` (current: {}).",
            report.ollama_url
        ));
        hints.push(
            "After Ollama is reachable, rerun `ragcli doctor` to verify configured models."
                .to_string(),
        );
    }

    if report.telemetry_error.is_some() {
        hints.push(
            "Fix telemetry environment variables, or unset OTEL_EXPORTER_OTLP_* to disable telemetry."
                .to_string(),
        );
    } else if !report.telemetry.enabled {
        hints.push(
            "Telemetry is optional. For local traces, run Arize Phoenix and set `OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:6006`; for managed tracing like Logfire or Phoenix Cloud, set that provider's OTLP endpoint and headers."
                .to_string(),
        );
    } else if !report.telemetry.headers_configured {
        hints.push(
            "Telemetry export is enabled. Local collectors usually do not need headers; managed services like Logfire or Phoenix Cloud usually require `OTEL_EXPORTER_OTLP_HEADERS`."
                .to_string(),
        );
    }

    hints
}

fn model_install_hints(
    report: &DoctorReport,
    installed_models: &ModelInstallReport,
) -> Vec<String> {
    let mut missing_by_model: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    let configured = [
        (
            "embed",
            report.embed_model.as_str(),
            installed_models.embed_model_installed,
        ),
        (
            "chat",
            report.chat_model.as_str(),
            installed_models.chat_model_installed,
        ),
        (
            "vision",
            report.vision_model.as_str(),
            installed_models.vision_model_installed,
        ),
    ];

    for (role, model, installed) in configured {
        if !installed {
            missing_by_model.entry(model).or_default().push(role);
        }
    }

    missing_by_model
        .into_iter()
        .map(|(model, roles)| {
            format!(
                "Install the {} model for {}: `ollama pull {}`.",
                if roles.len() == 1 {
                    "missing"
                } else {
                    "shared"
                },
                roles.join("/"),
                model
            )
        })
        .collect()
}

/// Print a bold cyan section header.
fn section(title: &str) {
    println!("{}", style(title).bold().cyan());
}

/// Print a plain key/value row with aligned padding.
fn kv(label: &str, value: &str, width: usize) {
    println!("  {:<width$}  {}", style(label).dim(), value, width = width);
}

/// Print a key/value row where the value is already a styled string.
fn kv_styled(label: &str, value: &str, width: usize) {
    println!("  {:<width$}  {}", style(label).dim(), value, width = width);
}

/// Print a path row: label  path  ✓/✗ status.
fn path_row(label: &str, report: &PathStatusReport, width: usize) {
    let status_str = if report.status == "exists" {
        ok("exists")
    } else {
        err("missing")
    };
    println!(
        "  {:<width$}  {}  {}",
        style(label).dim(),
        style(&report.path).dim(),
        status_str,
        width = width
    );
}

/// Print a model row: label  model-name  ✓/✗ installed.
fn model_row(label: &str, model: &str, installed: bool, width: usize) {
    let status_str = if installed {
        ok("installed")
    } else {
        err("not installed")
    };
    println!(
        "  {:<width$}  {:<30}  {}",
        style(label).dim(),
        model,
        status_str,
        width = width
    );
}

/// Green ✓ prefix with text.
fn ok(text: &str) -> String {
    format!("{} {}", style("✓").green().bold(), style(text).green())
}

/// Red ✗ prefix with text.
fn err(text: &str) -> String {
    format!("{} {}", style("✗").red().bold(), style(text).red())
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
    use std::env;

    struct ScopedEnvVars {
        previous: Vec<(&'static str, Option<std::ffi::OsString>)>,
    }

    impl ScopedEnvVars {
        fn set(vars: &[(&'static str, &str)]) -> Self {
            let previous = vars
                .iter()
                .map(|(name, _)| (*name, env::var_os(name)))
                .collect();

            unsafe {
                for (name, value) in vars {
                    env::set_var(name, value);
                }
            }

            Self { previous }
        }
    }

    impl Drop for ScopedEnvVars {
        fn drop(&mut self) {
            unsafe {
                for (name, value) in self.previous.drain(..).rev() {
                    restore_var(name, value);
                }
            }
        }
    }

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
            assert!(!report.telemetry.enabled);
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

    #[tokio::test(flavor = "current_thread")]
    async fn test_build_report_includes_telemetry_env_configuration() {
        let dir = tempfile::tempdir().unwrap();
        with_test_env(dir.path(), None, || async {
            let _vars = ScopedEnvVars::set(&[
                (
                    crate::telemetry::ENV_OTEL_EXPORTER_OTLP_ENDPOINT,
                    "http://localhost:6006",
                ),
                (
                    crate::telemetry::ENV_OTEL_EXPORTER_OTLP_PROTOCOL,
                    "http/protobuf",
                ),
                (crate::telemetry::ENV_OTEL_SERVICE_NAME, "ragcli-test"),
                (
                    crate::telemetry::ENV_OTEL_EXPORTER_OTLP_HEADERS,
                    "api_key=secret",
                ),
                (crate::telemetry::ENV_OTEL_EXPORTER_OTLP_TIMEOUT, "2500"),
            ]);

            let report = build_report(Some("telemetry")).await.unwrap();
            assert!(report.telemetry.enabled);
            assert_eq!(report.telemetry.service_name, "ragcli-test");
            assert_eq!(report.telemetry.protocol, "http/protobuf");
            assert_eq!(
                report.telemetry.endpoint.as_deref(),
                Some("http://localhost:6006/v1/traces")
            );
            assert_eq!(report.telemetry.timeout_ms, Some(2500));
            assert!(report.telemetry.headers_configured);
            assert!(report.telemetry_error.is_none());
        })
        .await;
    }

    #[tokio::test(flavor = "current_thread")]
    async fn test_build_report_keeps_running_when_telemetry_env_is_invalid() {
        let dir = tempfile::tempdir().unwrap();
        with_test_env(dir.path(), None, || async {
            let _vars = ScopedEnvVars::set(&[
                (
                    crate::telemetry::ENV_OTEL_EXPORTER_OTLP_ENDPOINT,
                    "http://localhost:6006",
                ),
                (
                    crate::telemetry::ENV_OTEL_EXPORTER_OTLP_PROTOCOL,
                    "http/json",
                ),
                (crate::telemetry::ENV_OTEL_SERVICE_NAME, "ragcli-test"),
            ]);

            let report = build_report(Some("telemetry-invalid")).await.unwrap();
            assert!(report.telemetry.enabled);
            assert_eq!(report.telemetry.service_name, "ragcli-test");
            assert_eq!(report.telemetry.protocol, "http/protobuf");
            assert_eq!(
                report.telemetry.endpoint.as_deref(),
                Some("http://localhost:6006/v1/traces")
            );
            assert!(report
                .telemetry_error
                .as_deref()
                .unwrap()
                .contains("unsupported OTEL_EXPORTER_OTLP_PROTOCOL value"));
            assert!(serde_json::to_string(&report)
                .unwrap()
                .contains("\"telemetry_error\""));
        })
        .await;
    }

    #[test]
    fn test_scoped_env_vars_restore_on_panic() {
        let _guard = crate::config::test_env_lock().lock().unwrap();
        let previous_service_name = env::var_os(crate::telemetry::ENV_OTEL_SERVICE_NAME);

        let result = std::panic::catch_unwind(|| {
            let _vars = ScopedEnvVars::set(&[(crate::telemetry::ENV_OTEL_SERVICE_NAME, "boom")]);
            panic!("force unwind");
        });

        assert!(result.is_err());
        assert_eq!(
            env::var_os(crate::telemetry::ENV_OTEL_SERVICE_NAME),
            previous_service_name
        );
    }

    unsafe fn restore_var(name: &str, value: Option<std::ffi::OsString>) {
        match value {
            Some(value) => env::set_var(name, value),
            None => env::remove_var(name),
        }
    }
}
