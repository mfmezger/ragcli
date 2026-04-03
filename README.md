# ragcli

`ragcli` is a small local RAG CLI written in Rust.

It indexes local files into a persistent LanceDB store, uses Ollama for embeddings and generation, and stays intentionally simple so the whole flow is easy to inspect and extend.

## Features

- local text, Markdown, PDF, and image indexing
- local embedding and generation through Ollama
- hybrid retrieval with LanceDB vector search + BM25 full-text search
- persistent per-store data under `~/.config/ragcli/<name>`
- idempotent re-indexing by `source_path`
- compatibility checks for embedding model + chunk settings
- `doctor` checks for store state, Ollama reachability, and installed models
- `stat` summarizes indexed content, approximate embedded token volume, and store disk usage
- optional retrieval inspection with `query --show-context`

## Quick Start

1. Start Ollama.
2. Pull the embedding and chat models.
3. Index a local file or folder.
4. Query your local corpus.

```bash
ollama pull nomic-embed-text-v2-moe:latest
ollama pull qwen3.5:4b

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
```

Use a named store:

```bash
cargo run -- index ./docs --name work
cargo run -- query "Summarize the notes" --name work
```

Inspect retrieved context before generation:

```bash
cargo run -- query "What is My Neighbor Totoro about?" --show-context
```

Check local setup:

```bash
cargo run -- doctor
```

Inspect what is already embedded:

```bash
cargo run -- stat
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
cargo run -- config set models.embed nomic-embed-text-v2-moe:latest
cargo run -- config set ollama.base_url http://localhost:11434
```

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

`meta/store.toml` records the embedding model, embedding dimension, and chunk settings used to build the store.

## Behavior Notes

- Re-indexing replaces existing rows for the same `source_path`.
- Text files are decoded lossily, so non-UTF-8 files do not abort indexing.
- Images are captioned with an Ollama vision model at index time and stored as text for retrieval.
- Queries use LanceDB hybrid search: semantic nearest-neighbor search plus BM25 full-text search on `chunk_text`.
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
cargo test
```

If you use [Task](https://taskfile.dev/), the repo also includes [Taskfile.yml](Taskfile.yml) with shortcuts for the common workflows:

```bash
task build
task check
task test
task release -- patch
task fmt
task doctor
task stat
```

Task arguments can be forwarded to CLI tasks with `--`, for example:

```bash
task index -- ./docs
task query -- "What is this project about?"
```

Releases are configured with `cargo-release` through [`Cargo.toml`](Cargo.toml). The repository is set up to:

- only release from `main`
- create tags as `v<version>`
- default to `cargo release` dry runs unless you pass `--execute`
- skip crates.io publishing for now

Typical usage:

```bash
cargo install cargo-release
cargo release patch
cargo release patch --execute
```
