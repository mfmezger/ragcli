# OpenTelemetry / OTLP implementation checklist

This checklist tracks the planned work to add optional OpenTelemetry trace export to `ragcli`, including support for Phoenix and other OTLP-compatible backends.

## Related GitHub issues

- [ ] #46 — `epic(telemetry): track OTLP export integration`
  - https://github.com/mfmezger/ragcli/issues/46
- [ ] #43 — `feat(telemetry): add OTLP bootstrap and exporter config`
  - https://github.com/mfmezger/ragcli/issues/43
- [ ] #44 — `feat(tracing): instrument query, index, and Ollama spans`
  - https://github.com/mfmezger/ragcli/issues/44
- [ ] #45 — `docs(telemetry): add doctor output, docs, and validation`
  - https://github.com/mfmezger/ragcli/issues/45

## Milestone 1: OTLP bootstrap and exporter configuration

- [x] Add `src/telemetry.rs`
- [x] Replace `init_tracing()` in `src/main.rs` with telemetry bootstrap logic
- [x] Keep existing stderr `tracing_subscriber::fmt` output by default
- [x] Add optional OTLP trace export via OpenTelemetry
- [x] Support standard env vars:
  - [x] `OTEL_SERVICE_NAME`
  - [x] `OTEL_EXPORTER_OTLP_ENDPOINT`
  - [x] `OTEL_EXPORTER_OTLP_HEADERS`
  - [x] `OTEL_EXPORTER_OTLP_PROTOCOL`
  - [x] `OTEL_EXPORTER_OTLP_TIMEOUT`
- [x] Decide and document enablement behavior when OTLP env vars are absent
- [x] Configure resource attributes:
  - [x] `service.name`
  - [x] `service.version`
  - [ ] optional deployment/resource attributes
- [x] Use batch exporting, not per-span simple export
- [x] Add graceful tracer shutdown / flush on process exit
- [x] Confirm both `http/protobuf` and `grpc` behavior or explicitly document supported protocol(s)

## Milestone 2: Instrumentation

- [x] Keep or improve root command span in `src/app.rs`
- [x] Add attributes for command/store/query mode at the command level
- [x] Expand indexing spans in `src/commands/index.rs`
- [x] Add query lifecycle spans in `src/query/execute.rs`
- [x] Add retrieval/rerank spans in `src/query/retrieve.rs` and `src/query/rerank.rs`
- [x] Add Ollama spans in `src/models.rs` for:
  - [x] tags/model discovery
  - [x] embeddings
  - [x] chat/generation
  - [x] vision captioning
- [x] Record safe metadata only by default:
  - [x] model name
  - [x] endpoint/host
  - [x] duration
  - [x] counts and sizes
  - [x] success/failure
- [x] Avoid exporting full prompts, contexts, image bytes, or source content by default

## Milestone 3: Doctor, docs, and validation

- [ ] Extend `doctor` to report telemetry configuration
- [ ] Redact secrets when reporting OTLP headers/auth presence
- [ ] Add README documentation for OTLP export
- [ ] Add a Phoenix example
- [ ] Add a generic Collector example
- [ ] Document privacy considerations and what is/is not exported by default
- [ ] Add unit tests for telemetry config parsing and disabled/enabled paths
- [ ] Add smoke/integration coverage for OTLP-enabled startup
- [ ] Optionally add a collector-backed integration test later

## Open questions

- [ ] Should OTEL be enabled automatically when `OTEL_EXPORTER_OTLP_ENDPOINT` is set, or require an explicit `RAGCLI_OTEL_ENABLED` flag?
- [ ] Should exporter init failure warn-and-continue by default, or fail fast?
- [ ] Should `doctor` attempt any connectivity validation for OTLP endpoints, or only print resolved config?
- [ ] Should we add `RAGCLI_OTEL_STRICT` and/or `RAGCLI_OTEL_CONSOLE` toggles?
- [ ] Do we want any LLM semantic-convention attributes now, or keep v1 generic?
