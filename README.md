# ragcli

[![Cargo Test](https://github.com/mfmezger/ragcli/actions/workflows/cargo-test.yml/badge.svg)](https://github.com/mfmezger/ragcli/actions/workflows/cargo-test.yml)
[![Coverage](https://img.shields.io/badge/coverage-80%25%20required-brightgreen)](https://github.com/mfmezger/ragcli/actions/workflows/cargo-test.yml)
[![Rust 2021](https://img.shields.io/badge/rust-2021-orange)](https://www.rust-lang.org/)

`ragcli` is a small local RAG CLI written in Rust.

It indexes local files into a persistent LanceDB store, uses Ollama for embeddings and generation, and stays intentionally simple so the whole flow is easy to inspect and extend.

## Features

- local text, Markdown, HTML, CSV/TSV, source code, PDF, and image indexing
- local embedding and generation through Ollama
- hybrid retrieval with LanceDB vector search + BM25 full-text search
- persistent per-store data under `~/.config/ragcli/<name>`
- idempotent re-indexing by `source_path`
- compatibility checks for embedding model + chunk settings
- `doctor` checks for store state, Ollama reachability, and installed models
- `stat` summarizes indexed content, approximate embedded token volume, and store disk usage
- `sources`/`ls`, `delete`, `clear`, and `prune` help inspect and maintain indexed content
- query modes with inspectable retrieval output via `--mode`, `--show-plan`, `--show-scores`, `--show-citations`, and `--show-trace`

## Quick Start

1. Start Ollama.
2. Pull the embedding and chat models.
3. Index a local file or folder.
4. Query your local corpus.

```bash
ollama pull nomic-embed-text-v2-moe:latest
ollama pull qwen3.5:4b

cargo run -- --help
cargo run -- doctor
cargo run -- stat
cargo run -- index ./docs
cargo run -- query "What is this project about?"
```

## Commands

Index a directory or file:

```bash
cargo run -- index ./docs
cargo run -- index ./My_Neighbor_Totoro.pdf
cargo run -- index ./images
cargo run -- index . --exclude '**/target/**' --exclude '**/.git/**'
```

Use a named store:

```bash
cargo run -- index ./docs --name work
cargo run -- query "Summarize the notes" --name work
```

Inspect retrieved context before generation:

```bash
cargo run -- query "What is My Neighbor Totoro about?" --show-context
cargo run -- query "Summarize this file" --source ./notes/today.md
cargo run -- query "What happens on this page?" --source ./My_Neighbor_Totoro.pdf --page 3
cargo run -- query "What changed in the docs?" --path-prefix ./docs/
cargo run -- query "Summarize the markdown notes" --format markdown
cargo run -- query "What is this project about?" --mode hybrid --show-plan
cargo run -- query "Summarize the release notes" --mode agentic --show-trace
cargo run -- query "Which file mentions Totoro?" --show-citations --show-scores
```

Check local setup:

```bash
cargo run -- doctor
cargo run -- doctor --json
```

Inspect what is already embedded:

```bash
cargo run -- stat
cargo run -- stat --json
cargo run -- sources
cargo run -- ls
```

Remove or clean indexed content:

```bash
cargo run -- delete ./notes/today.md
cargo run -- prune
cargo run -- prune --json
cargo run -- prune --apply
cargo run -- clear --yes
```

## Configuration

Config lives at:

```text
~/.config/ragcli/<name>/config.toml
```

Default config:

```toml
[ollama]
base_url = "http://localhost:11434"

[models]
embed = "nomic-embed-text-v2-moe:latest"
chat = "qwen3.5:4b"
vision = "qwen3.5:4b"

[chunk]
size = 1000
overlap = 200
```

Effective runtime values can be overridden with environment variables:

```bash
export RAGCLI_OLLAMA_URL=http://localhost:11434
export RAGCLI_EMBED_MODEL=nomic-embed-text-v2-moe:latest
export RAGCLI_CHAT_MODEL=qwen3.5:4b
export RAGCLI_VISION_MODEL=qwen3.5:4b
```

You can inspect and update config without editing TOML by hand:

```bash
cargo run -- config show
cargo run -- config show --json
cargo run -- config set models.embed nomic-embed-text-v2-moe:latest
cargo run -- config set ollama.base_url http://localhost:11434
```

Supported indexable formats currently include:

- plain text: `.txt`, `.rst`
- Markdown: `.md`, `.markdown`
- HTML: `.html`, `.htm`
- tabular text: `.csv`, `.tsv`
- source/config text: `.rs`, `.py`, `.js`, `.ts`, `.tsx`, `.jsx`, `.go`, `.java`, `.c`, `.cc`, `.cpp`, `.cxx`, `.h`, `.hpp`, `.sh`, `.bash`, `.toml`, `.yaml`, `.yml`, `.json`
- PDF: `.pdf`
- images: `.png`, `.jpg`, `.jpeg`, `.webp`

## Storage

Each store lives under:

```text
~/.config/ragcli/<name>/
  lancedb/
  meta/
  cache/
  models/
  config.toml
```

`meta/store.toml` records the embedding model, embedding dimension, chunk settings, and store schema version used to build the store.

When upgrading across store schema changes, reindex into a fresh store or remove the old store before indexing again.

## Behavior Notes

- Re-indexing replaces existing rows for the same `source_path`.
- `sources`/`ls` lists indexed paths with per-source format, chunk count, character count, token estimate, and page count when applicable.
- `delete <path>` removes one indexed source path.
- `prune` previews rows whose stored `source_path` no longer exists on disk; add `--apply` to remove them.
- `clear` removes all indexed rows for the selected store and requires `--yes`.
- Text and source files are decoded lossily, so non-UTF-8 files do not abort indexing.
- Hidden files and directories are skipped during directory traversal unless `--include-hidden` is set.
- `index --exclude <glob>` can be repeated to skip unwanted files or directories.
- HTML is converted to readable text before chunking, and CSV/TSV rows are flattened into labeled text.
- Images are captioned with an Ollama vision model at index time and stored as text for retrieval.
- Queries support `--mode naive|hybrid|agentic|local|global|mix`; `agentic` runs the iterative Ralph-style retrieval loop, while `local`, `global`, and `mix` use distinct placeholder graph-mode paths that still fall back to hybrid retrieval until graph indexing lands.
- Queries use LanceDB hybrid search: semantic nearest-neighbor search plus BM25 full-text search on `chunk_text`.
- Query-time retrieval filters support `--source`, `--path-prefix`, `--page`, and `--format`.
- Querying refuses to mix a store with a different embedding model than the one used to build it.
- Ollama chat requests are sent with `think: false` to reduce hangs with reasoning-capable models.

## Development

Build:

```bash
cargo build
```

Verify:

```bash
cargo check
cargo test --all-targets
cargo fmt -- --check
```

Coverage:

```bash
cargo install cargo-llvm-cov
rustup component add llvm-tools-preview
cargo llvm-cov --all-targets --fail-under-lines 80
cargo llvm-cov --all-targets --html
```

If you use [Task](https://taskfile.dev/), the repo also includes [Taskfile.yml](Taskfile.yml) with shortcuts for the common workflows:

```bash
task build
task check
task test
task coverage
task coverage-html
task coverage-lcov
task changelog
task changelog-preview
task release -- patch
task release-execute -- patch
task fmt
task doctor
task stat
task sources
task delete -- ./notes/today.md
task clear -- --yes
task prune
task prune -- --apply
```

Task arguments can be forwarded to CLI tasks with `--`, for example:

```bash
task index -- ./docs
task query -- "What is this project about?"
```

## CI and Coverage

GitHub Actions runs both `cargo test --all-targets` and a separate coverage job using `cargo-llvm-cov` for pushes to `main` and pull requests targeting `main`.

The coverage job fails if line coverage drops below 80%:

```bash
cargo llvm-cov --all-targets --fail-under-lines 80
```

To make this a merge requirement on GitHub, set the `Coverage (80% required)` status check as a required check in the branch protection rule for `main`.

Releases are configured with `cargo-release` through [`Cargo.toml`](Cargo.toml). The repository is set up to:

- only release from `main`
- create tags as `v<version>`
- regenerate [`CHANGELOG.md`](CHANGELOG.md) with [`git-cliff`](https://git-cliff.org/) before the release commit
- default to `cargo release` dry runs unless you pass `--execute`
- skip crates.io publishing for now

Typical usage:

```bash
cargo install cargo-release
cargo install git-cliff
cargo release patch
cargo release patch --execute
```
