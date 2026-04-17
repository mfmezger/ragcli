# Repository Guidelines

## Project Structure & Module Organization
`ragcli` is a Rust CLI with the entrypoint in [`src/main.rs`](src/main.rs), which initializes tracing and hands execution to [`src/app.rs`](src/app.rs) for command dispatch. The Clap interface lives in [`src/cli.rs`](src/cli.rs). Subcommand handlers are split under [`src/commands/`](src/commands/): [`config.rs`](src/commands/config.rs), [`doctor.rs`](src/commands/doctor.rs), [`index.rs`](src/commands/index.rs), [`query.rs`](src/commands/query.rs), [`sources.rs`](src/commands/sources.rs), [`stat.rs`](src/commands/stat.rs), and [`maintenance.rs`](src/commands/maintenance.rs) for `delete`, `clear`, and `prune`.

Core library-style modules stay in `src/`: [`ingest.rs`](src/ingest.rs) handles discovery, format parsing, and chunking; [`store.rs`](src/store.rs) owns LanceDB persistence, retrieval helpers, and indexed-source summaries; [`models.rs`](src/models.rs) wraps Ollama embedding, chat, and vision calls; [`config.rs`](src/config.rs) manages store layout, config loading, and model resolution; and [`query/`](src/query/) contains retrieval pipeline pieces such as execution, rendering, reranking, retrieval, runtime, and shared query types. Supporting modules such as [`agent.rs`](src/agent.rs), [`citation.rs`](src/citation.rs), [`graph.rs`](src/graph.rs), [`retrieval.rs`](src/retrieval.rs), [`rewrite.rs`](src/rewrite.rs), [`source_kind.rs`](src/source_kind.rs), [`fsutil.rs`](src/fsutil.rs), and [`jsonutil.rs`](src/jsonutil.rs) provide focused helpers. Test-only helpers live in [`src/test_support.rs`](src/test_support.rs). Integration-style CLI tests live in [`tests/json_output.rs`](tests/json_output.rs), and recorded HTTP fixtures for replayable model tests live under [`tests/cassettes/`](tests/cassettes/). GitHub Actions workflows live under [`.github/workflows/`](.github/workflows/), and local automation lives in [`Taskfile.yml`](Taskfile.yml).

## Build, Test, and Development Commands
Use standard Cargo commands from the repo root:

- `cargo build` builds the CLI.
- `cargo run -- --help` shows the current command surface.
- `cargo run -- doctor` checks store layout, Ollama reachability, and model availability.
- `cargo run -- doctor --json` prints the same checks as machine-readable JSON.
- `cargo run -- stat` shows indexed-content and disk-usage summaries for the selected store.
- `cargo run -- sources` lists indexed source paths and per-source metadata.
- `cargo run -- delete ./notes/today.md` removes one indexed source.
- `cargo run -- prune` previews stale indexed sources whose backing files are gone.
- `cargo run -- prune --apply` removes stale indexed sources.
- `cargo run -- clear --yes` clears all indexed content from the selected store.
- `cargo run -- index ./docs` indexes a file or directory into the default store.
- `cargo run -- index ./docs --pdf-parser liteparse` indexes PDFs with the alternate parser.
- `cargo run -- index . --exclude '**/target/**' --exclude '**/.git/**' --include-hidden --force` demonstrates common indexing flags.
- `cargo run -- query "What is this project about?"` runs retrieval and generation.
- `cargo run -- query "What changed?" --mode hybrid --show-plan --show-scores --show-citations` inspects retrieval decisions.
- `cargo run -- query "Summarize the release notes" --mode agentic --rewrite --rerank --show-trace` exercises the iterative retrieval path.
- `cargo run -- config show` prints the effective config for the selected store.
- `cargo run -- config show --json` prints config and override provenance as JSON.
- `cargo run -- config set models.chat qwen3.5:4b` updates a config key in the selected store.
- `cargo check` validates the code quickly without producing a release binary.
- `cargo test --all-targets` matches the main CI test command.
- `cargo fmt -- --check` verifies formatting before review.
- `cargo llvm-cov --all-targets --fail-under-lines 80` runs the same coverage gate enforced in CI.

Equivalent `Task` shortcuts are available in [`Taskfile.yml`](Taskfile.yml):

- `task build`
- `task check`
- `task test`
- `task coverage`
- `task coverage-html`
- `task coverage-lcov`
- `task changelog`
- `task changelog-preview`
- `task release -- patch`
- `task release-execute -- patch`
- `task fmt`
- `task doctor`
- `task stat`
- `task sources`
- `task delete -- ./notes/today.md`
- `task clear -- --yes`
- `task prune`
- `task prune -- --apply`
- `task index -- ./docs`
- `task query -- "What is this project about?"`
- `task config-show`
- `task config-set -- models.chat qwen3.5:4b`

Use the global `--name <store>` flag with Cargo commands when working with a non-default store, for example `cargo run -- --name scratch stat` or `cargo run -- --name work query "Summarize the notes"`.

Releases are managed with `cargo-release` via the `[package.metadata.release]` config in [`Cargo.toml`](Cargo.toml). Changelogs are generated with [`git-cliff`](https://git-cliff.org/) using [`cliff.toml`](cliff.toml) and written to [`CHANGELOG.md`](CHANGELOG.md). Use dry runs by default and only release from `main`.

## Coding Style & Naming Conventions
Follow idiomatic Rust and let `rustfmt` own formatting. Use 4-space indentation, `snake_case` for functions, variables, and modules, and `CamelCase` for structs and enums. Keep modules focused on one responsibility and prefer small helpers over deeply nested logic. This project already uses `anyhow::Result` for fallible flows, `clap` derives for CLI parsing, `serde` for structured output, and async command handlers with Tokio; stay consistent with those patterns. When adding user-facing command output, keep the text mode concise and offer JSON output where the command already supports `--json`.

## Testing Guidelines
Tests are primarily inline `#[cfg(test)]` unit tests inside the modules they cover; see [`src/cli.rs`](src/cli.rs), [`src/config.rs`](src/config.rs), [`src/ingest.rs`](src/ingest.rs), [`src/store.rs`](src/store.rs), [`src/models.rs`](src/models.rs), and [`src/commands/maintenance.rs`](src/commands/maintenance.rs). Use [`src/test_support.rs`](src/test_support.rs) for temporary environment and store-layout helpers instead of rolling custom setup in each test. Name tests after the behavior they verify, for example `test_build_prune_report_marks_missing_sources_without_deleting_by_default`. Keep integration-style CLI assertions in `tests/` when you need to validate actual binary output or JSON formatting, as in [`tests/json_output.rs`](tests/json_output.rs).

For HTTP integrations, prefer replayable cassette tests over live network dependencies when practical, and keep cassette fixtures in [`tests/cassettes/`](tests/cassettes/). Add tests for new indexing, maintenance, retrieval, config, or model-client behavior, and run `cargo test --all-targets` before opening a PR. When changing retrieval logic, JSON reporting, or command output used by automation, consider also running the coverage task to catch untested branches.

## Commit & Pull Request Guidelines
Recent history follows concise Conventional Commit-style subjects such as `feat(store): add source management commands`, `fix(prune): avoid cwd-based stale deletion`, `perf(store): stream source aggregation queries`, and `refactor(query): split pipeline modules`. Prefer `feat`, `fix`, `perf`, `refactor`, `test`, and `ci` prefixes with a narrow scope.

PRs should describe the behavior change, mention any Ollama, store-layout, or retrieval-mode impact, link the relevant issue when applicable, and include CLI output snippets when user-visible behavior changes. If you add or modify JSON output, mention that explicitly because downstream tooling may depend on it. Changes that affect CI should mention the relevant workflow, currently [`.github/workflows/cargo-test.yml`](.github/workflows/cargo-test.yml), which runs both `cargo test --all-targets` and the `Coverage (80% required)` job.