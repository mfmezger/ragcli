use anyhow::{anyhow, Context, Result};
use opentelemetry::global;
use opentelemetry::trace::TracerProvider as _;
use opentelemetry::KeyValue;
use opentelemetry_otlp::{Protocol, SpanExporter, WithExportConfig, WithHttpConfig};
use opentelemetry_sdk::{trace::SdkTracerProvider, Resource};
use reqwest::blocking::Client;
use tracing_opentelemetry::OpenTelemetryLayer;
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

const DEFAULT_FILTER: &str = "info";
const DEFAULT_SERVICE_NAME: &str = env!("CARGO_PKG_NAME");
const SERVICE_VERSION_KEY: &str = "service.version";
const HTTP_TRACES_PATH: &str = "/v1/traces";

pub const ENV_OTEL_SERVICE_NAME: &str = "OTEL_SERVICE_NAME";
pub const ENV_OTEL_EXPORTER_OTLP_ENDPOINT: &str = "OTEL_EXPORTER_OTLP_ENDPOINT";
pub const ENV_OTEL_EXPORTER_OTLP_HEADERS: &str = "OTEL_EXPORTER_OTLP_HEADERS";
pub const ENV_OTEL_EXPORTER_OTLP_PROTOCOL: &str = "OTEL_EXPORTER_OTLP_PROTOCOL";
pub const ENV_OTEL_EXPORTER_OTLP_TIMEOUT: &str = "OTEL_EXPORTER_OTLP_TIMEOUT";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OtlpProtocol {
    HttpProtobuf,
    Grpc,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TelemetryConfig {
    pub enabled: bool,
    pub service_name: String,
    pub endpoint: Option<String>,
    pub protocol: OtlpProtocol,
    pub timeout_ms: Option<u64>,
    pub headers_configured: bool,
}

impl TelemetryConfig {
    pub fn from_env() -> Result<Self> {
        let endpoint = read_non_empty_env(ENV_OTEL_EXPORTER_OTLP_ENDPOINT);
        let protocol = match read_non_empty_env(ENV_OTEL_EXPORTER_OTLP_PROTOCOL).as_deref() {
            None => OtlpProtocol::HttpProtobuf,
            Some("http/protobuf") => OtlpProtocol::HttpProtobuf,
            Some("grpc") => OtlpProtocol::Grpc,
            Some(value) => {
                anyhow::bail!(
                    "unsupported {} value: {} (expected 'http/protobuf' or 'grpc')",
                    ENV_OTEL_EXPORTER_OTLP_PROTOCOL,
                    value
                )
            }
        };
        let timeout_ms = match read_non_empty_env(ENV_OTEL_EXPORTER_OTLP_TIMEOUT) {
            Some(value) => Some(value.parse().with_context(|| {
                format!("parse {} as milliseconds", ENV_OTEL_EXPORTER_OTLP_TIMEOUT)
            })?),
            None => None,
        };

        Ok(Self {
            enabled: endpoint.is_some(),
            service_name: read_non_empty_env(ENV_OTEL_SERVICE_NAME)
                .unwrap_or_else(|| DEFAULT_SERVICE_NAME.to_string()),
            endpoint,
            protocol,
            timeout_ms,
            headers_configured: read_non_empty_env(ENV_OTEL_EXPORTER_OTLP_HEADERS).is_some(),
        })
    }

    pub fn resolved_export_endpoint(&self) -> Option<String> {
        let endpoint = self.endpoint.as_deref()?;
        Some(match self.protocol {
            OtlpProtocol::HttpProtobuf => normalize_http_traces_endpoint(endpoint),
            OtlpProtocol::Grpc => endpoint.to_string(),
        })
    }
}

#[derive(Default)]
pub struct TelemetryGuard {
    tracer_provider: Option<SdkTracerProvider>,
}

impl TelemetryGuard {
    pub fn disabled() -> Self {
        Self::default()
    }

    pub fn shutdown(&mut self) -> Result<()> {
        if let Some(provider) = self.tracer_provider.take() {
            provider
                .shutdown()
                .map_err(|err| anyhow!(err.to_string()))
                .context("shutdown OTLP tracer provider")?;
        }
        Ok(())
    }
}

impl Drop for TelemetryGuard {
    fn drop(&mut self) {
        if let Err(err) = self.shutdown() {
            eprintln!("failed to shutdown telemetry: {err}");
        }
    }
}

pub fn init() -> Result<TelemetryGuard> {
    let filter = build_env_filter();
    let config = TelemetryConfig::from_env()?;
    let fmt_layer = fmt::layer().with_writer(std::io::stderr);

    if !config.enabled {
        tracing_subscriber::registry()
            .with(filter)
            .with(fmt_layer)
            .init();
        return Ok(TelemetryGuard::disabled());
    }

    let tracer_provider = build_tracer_provider(&config)?;
    let tracer = tracer_provider.tracer(DEFAULT_SERVICE_NAME.to_string());
    global::set_tracer_provider(tracer_provider.clone());

    tracing_subscriber::registry()
        .with(filter)
        .with(fmt_layer)
        .with(OpenTelemetryLayer::new(tracer))
        .init();

    Ok(TelemetryGuard {
        tracer_provider: Some(tracer_provider),
    })
}

fn build_tracer_provider(config: &TelemetryConfig) -> Result<SdkTracerProvider> {
    let resource = Resource::builder()
        .with_service_name(config.service_name.clone())
        .with_attributes([KeyValue::new(
            SERVICE_VERSION_KEY,
            env!("CARGO_PKG_VERSION"),
        )])
        .build();

    let endpoint = config
        .resolved_export_endpoint()
        .context("telemetry endpoint missing while OTLP export is enabled")?;

    let exporter = match config.protocol {
        OtlpProtocol::HttpProtobuf => SpanExporter::builder()
            .with_http()
            .with_http_client(Client::new())
            .with_protocol(Protocol::HttpBinary)
            .with_endpoint(endpoint)
            .with_timeout(timeout_or_default(config.timeout_ms))
            .build()
            .context("build OTLP HTTP trace exporter")?,
        OtlpProtocol::Grpc => SpanExporter::builder()
            .with_tonic()
            .with_protocol(Protocol::Grpc)
            .with_endpoint(endpoint)
            .with_timeout(timeout_or_default(config.timeout_ms))
            .build()
            .context("build OTLP gRPC trace exporter")?,
    };

    Ok(SdkTracerProvider::builder()
        .with_batch_exporter(exporter)
        .with_resource(resource)
        .build())
}

fn timeout_or_default(timeout_ms: Option<u64>) -> std::time::Duration {
    std::time::Duration::from_millis(timeout_ms.unwrap_or(10_000))
}

fn build_env_filter() -> EnvFilter {
    match std::env::var("RUST_LOG") {
        Ok(value) => EnvFilter::try_new(&value).unwrap_or_else(|err| {
            eprintln!("failed to parse RUST_LOG, using default '{DEFAULT_FILTER}' filter: {err}");
            EnvFilter::new(DEFAULT_FILTER)
        }),
        Err(std::env::VarError::NotPresent) => EnvFilter::new(DEFAULT_FILTER),
        Err(std::env::VarError::NotUnicode(_)) => {
            eprintln!(
                "failed to parse RUST_LOG, using default '{DEFAULT_FILTER}' filter: not valid unicode"
            );
            EnvFilter::new(DEFAULT_FILTER)
        }
    }
}

fn read_non_empty_env(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn normalize_http_traces_endpoint(endpoint: &str) -> String {
    if endpoint.ends_with(HTTP_TRACES_PATH) {
        endpoint.to_string()
    } else {
        format!("{}{HTTP_TRACES_PATH}", endpoint.trim_end_matches('/'))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::test_env_lock;
    use std::env;

    #[test]
    fn test_telemetry_config_disabled_without_endpoint() {
        let _guard = test_env_lock().lock().unwrap();
        let previous_endpoint = env::var_os(ENV_OTEL_EXPORTER_OTLP_ENDPOINT);
        let previous_service_name = env::var_os(ENV_OTEL_SERVICE_NAME);
        let previous_protocol = env::var_os(ENV_OTEL_EXPORTER_OTLP_PROTOCOL);

        unsafe {
            env::remove_var(ENV_OTEL_EXPORTER_OTLP_ENDPOINT);
            env::remove_var(ENV_OTEL_SERVICE_NAME);
            env::remove_var(ENV_OTEL_EXPORTER_OTLP_PROTOCOL);
        }

        let config = TelemetryConfig::from_env().unwrap();
        assert!(!config.enabled);
        assert_eq!(config.service_name, DEFAULT_SERVICE_NAME);
        assert_eq!(config.protocol, OtlpProtocol::HttpProtobuf);

        unsafe {
            restore_var(ENV_OTEL_EXPORTER_OTLP_ENDPOINT, previous_endpoint);
            restore_var(ENV_OTEL_SERVICE_NAME, previous_service_name);
            restore_var(ENV_OTEL_EXPORTER_OTLP_PROTOCOL, previous_protocol);
        }
    }

    #[test]
    fn test_telemetry_config_reads_protocol_and_timeout() {
        let _guard = test_env_lock().lock().unwrap();
        let previous_endpoint = env::var_os(ENV_OTEL_EXPORTER_OTLP_ENDPOINT);
        let previous_protocol = env::var_os(ENV_OTEL_EXPORTER_OTLP_PROTOCOL);
        let previous_timeout = env::var_os(ENV_OTEL_EXPORTER_OTLP_TIMEOUT);
        let previous_headers = env::var_os(ENV_OTEL_EXPORTER_OTLP_HEADERS);

        unsafe {
            env::set_var(ENV_OTEL_EXPORTER_OTLP_ENDPOINT, "http://collector:4317");
            env::set_var(ENV_OTEL_EXPORTER_OTLP_PROTOCOL, "grpc");
            env::set_var(ENV_OTEL_EXPORTER_OTLP_TIMEOUT, "2500");
            env::set_var(ENV_OTEL_EXPORTER_OTLP_HEADERS, "api_key=test");
        }

        let config = TelemetryConfig::from_env().unwrap();
        assert!(config.enabled);
        assert_eq!(config.protocol, OtlpProtocol::Grpc);
        assert_eq!(config.timeout_ms, Some(2500));
        assert!(config.headers_configured);
        assert_eq!(
            config.resolved_export_endpoint().as_deref(),
            Some("http://collector:4317")
        );

        unsafe {
            restore_var(ENV_OTEL_EXPORTER_OTLP_ENDPOINT, previous_endpoint);
            restore_var(ENV_OTEL_EXPORTER_OTLP_PROTOCOL, previous_protocol);
            restore_var(ENV_OTEL_EXPORTER_OTLP_TIMEOUT, previous_timeout);
            restore_var(ENV_OTEL_EXPORTER_OTLP_HEADERS, previous_headers);
        }
    }

    #[test]
    fn test_http_endpoint_appends_traces_path() {
        let config = TelemetryConfig {
            enabled: true,
            service_name: DEFAULT_SERVICE_NAME.to_string(),
            endpoint: Some("http://localhost:6006".to_string()),
            protocol: OtlpProtocol::HttpProtobuf,
            timeout_ms: None,
            headers_configured: false,
        };

        assert_eq!(
            config.resolved_export_endpoint().as_deref(),
            Some("http://localhost:6006/v1/traces")
        );
    }

    #[test]
    fn test_invalid_protocol_is_rejected() {
        let _guard = test_env_lock().lock().unwrap();
        let previous_endpoint = env::var_os(ENV_OTEL_EXPORTER_OTLP_ENDPOINT);
        let previous_protocol = env::var_os(ENV_OTEL_EXPORTER_OTLP_PROTOCOL);

        unsafe {
            env::set_var(ENV_OTEL_EXPORTER_OTLP_ENDPOINT, "http://collector:4318");
            env::set_var(ENV_OTEL_EXPORTER_OTLP_PROTOCOL, "http/json");
        }

        let err = TelemetryConfig::from_env().unwrap_err();
        assert!(err
            .to_string()
            .contains("unsupported OTEL_EXPORTER_OTLP_PROTOCOL value"));

        unsafe {
            restore_var(ENV_OTEL_EXPORTER_OTLP_ENDPOINT, previous_endpoint);
            restore_var(ENV_OTEL_EXPORTER_OTLP_PROTOCOL, previous_protocol);
        }
    }

    unsafe fn restore_var(name: &str, value: Option<std::ffi::OsString>) {
        match value {
            Some(value) => env::set_var(name, value),
            None => env::remove_var(name),
        }
    }
}
