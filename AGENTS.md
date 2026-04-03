# Repository Guidelines

## Project Structure & Module Organization
`ragcli` is a small Rust CLI with all application code under `src/`. [`src/main.rs`](/Users/mfm/Projects/ragcli/src/main.rs) wires the CLI commands together, [`src/cli.rs`](/Users/mfm/Projects/ragcli/src/cli.rs) defines the Clap interface, [`src/ingest.rs`](/Users/mfm/Projects/ragcli/src/ingest.rs) handles file discovery and chunking, [`src/store.rs`](/Users/mfm/Projects/ragcli/src/store.rs) manages LanceDB persistence, [`src/models.rs`](/Users/mfm/Projects/ragcli/src/models.rs) wraps Ollama calls, and [`src/config.rs`](/Users/mfm/Projects/ragcli/src/config.rs) owns store configuration. GitHub Actions workflows live under [`.github/workflows/`](/Users/mfm/Projects/ragcli/.github/workflows/). Sample assets such as `example.png` and `My_Neighbor_Totoro.pdf` are local fixtures, not library code. Recorded HTTP fixtures for replay tests live under [`tests/cassettes/`](/Users/mfm/Projects/ragcli/tests/cassettes/).

## Build, Test, and Development Commands
Use standard Cargo commands from the repo root:

- `cargo build` builds the CLI.
- `cargo run -- doctor` checks local store layout and Ollama availability.
- `cargo run -- index ./docs` indexes a file or directory into the default store.
- `cargo run -- query "What is this project about?"` runs retrieval and generation.
- `cargo check` validates the code quickly without producing a release binary.
- `cargo test` runs the unit tests.
- `cargo fmt -- --check` verifies formatting before review.

## Coding Style & Naming Conventions
Follow idiomatic Rust and let `rustfmt` own formatting. Use 4-space indentation, `snake_case` for functions, variables, and modules, and `CamelCase` for structs and enums. Keep modules focused on one responsibility and prefer small helpers over deeply nested logic. This project already uses `anyhow::Result` for fallible flows and `clap` derives for CLI arguments; stay consistent with those patterns.

## Testing Guidelines
Tests are primarily inline `#[cfg(test)]` unit tests inside the module they cover; see [`src/config.rs`](/Users/mfm/Projects/ragcli/src/config.rs), [`src/ingest.rs`](/Users/mfm/Projects/ragcli/src/ingest.rs), [`src/store.rs`](/Users/mfm/Projects/ragcli/src/store.rs), and [`src/models.rs`](/Users/mfm/Projects/ragcli/src/models.rs). Name tests after the behavior they verify, for example `test_replace_source_rows_removes_duplicates`. For HTTP integrations, prefer replayable cassette tests over live network dependencies when practical, and keep cassette fixtures in [`tests/cassettes/`](/Users/mfm/Projects/ragcli/tests/cassettes/). Add tests for new chunking, storage, config, or model-client behavior, and run `cargo test` before opening a PR.

## Commit & Pull Request Guidelines
Recent history follows concise Conventional Commit-style subjects such as `feat(cli): add hybrid lancedb retrieval` and `fix(store): rebuild fts index after delete-only updates`. Prefer `feat`, `fix`, `test`, and `ci` prefixes with a narrow scope. PRs should describe the behavior change, mention any Ollama or store-layout impact, link the relevant issue when applicable, and include CLI output snippets when user-visible behavior changes. Changes that affect CI should mention the relevant workflow, currently [`.github/workflows/cargo-test.yml`](/Users/mfm/Projects/ragcli/.github/workflows/cargo-test.yml).
