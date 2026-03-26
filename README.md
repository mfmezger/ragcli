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
