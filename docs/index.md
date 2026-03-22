---
title: Home
description: quarry - Concurrent site crawler + semantic search engine for AI agents
---

# 🪨 quarry

**Turn any website into a queryable knowledge base.**

quarry is a two-component system designed for AI agents that need accurate, offline-accessible site knowledge. The Go crawler systematically traverses a target site, extracts structured content, and produces a JSON index. The Rust semantic search engine embeds these pages with GPU-accelerated ONNX inference and serves natural language queries through HTTP or CLI interfaces.

## Why quarry?

| Feature | Benefit |
|---------|---------|
| **Offline First** | No live web requests at query time — all data is pre-indexed |
| **GPU Accelerated** | ONNX Runtime with CUDA for fast embedding inference |
| **Memory Safe** | Zero leaks verified, bounded memory for any site size |
| **Agent Friendly** | JSON output, REST API, structured error responses |
| **Resource Conscious** | Runs on 2GB VRAM (MX250-class hardware) |
| **Easy Deployment** | Single binary per component, no runtime dependencies |

## Quick Start

### 1. Install Dependencies

```bash
# Go 1.22+
go install

# Rust 1.75+
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# CUDA 13.0 (for GPU inference)
# See NVIDIA docs for your platform
```

### 2. Export the ONNX Model

```bash
pip install transformers optimum[onnx] torch
python scripts/export_onnx.py --output-dir models
```

### 3. Build

```bash
# Go crawler
cd crawler && go build -o bin/crawler ./cmd/crawler

# Rust search engine
cd semantic-search && cargo build --release
```

### 4. Crawl a Site

```bash
./crawler/bin/crawler \
  --entry "https://minecraft-linux.github.io/" \
  --workers 10 \
  --output-dir ./output
```

### 5. Index and Search

```bash
# Build the vector index
./semantic-search/target/release/indexer

# Query via CLI
./semantic-search/target/release/search "how to install minecraft" --top-k 5

# Or start the HTTP API
./semantic-search/target/release/search-api
curl -X POST http://localhost:8080/search \
  -H "Content-Type: application/json" \
  -d '{"query": "java runtime requirements", "top_k": 5}'
```

!!! tip "Full Installation Guide"
    See [Installation](installation.md) for detailed setup instructions including CUDA, ONNX Runtime, and troubleshooting.

## Architecture Overview

```mermaid
flowchart LR
    subgraph Go["Go Crawler"]
        A[Entry URL] --> B[Worker Pool]
        B --> C[HTML Extractor]
        C --> D[JSON Writer]
    end

    subgraph Rust["Rust Semantic Search"]
        E[Document Loader] --> F[ONNX Inference]
        F --> G[HNSW Index]
        G --> H[Query Engine]
    end

    D -->|pages/*.json| E
    H -->|top-K results| I[AI Agent]
```

quarry separates crawling from querying. The Go crawler runs periodically to refresh content. The Rust search engine runs continuously, serving low-latency queries from an in-memory HNSW index. Both components communicate through filesystem JSON, eliminating the need for databases or message queues.

!!! info "Deep Dive"
    See [Architecture](architecture.md) for the complete system design, data flow, and design decisions.

## Component Guides

### Go Crawler

The crawler implements a concurrent worker pool with:

- **BFS traversal** with configurable depth
- **robots.txt compliance** (RFC 9309)
- **URL deduplication** via sync.Map (O(1) lookup)
- **Streaming JSON output** (bounded memory)
- **Inverted index** for keyword search

[CLI Reference →](crawler.md#cli-reference){ .md-button }

### Rust Semantic Search

The search engine provides:

- **ONNX inference** with CUDA execution provider
- **HNSW vector index** for O(log n) queries
- **VRAM management** with 1.5GB ceiling
- **HTTP API** and CLI interfaces
- **Persistent storage** with checksum verification

[API Reference →](semantic-search.md#http-api-reference){ .md-button }

## Supported Models

| Model | Params | VRAM (batch=8) | Quality | Recommended |
|-------|--------|----------------|---------|-------------|
| all-MiniLM-L6-v2 | 22M | ~620 MB | 58.5 MTEB | ✅ Default |
| bge-small-en-v1.5 | 33M | ~780 MB | 62.2 MTEB | Higher quality |
| e5-small-v2 | 33M | ~780 MB | 61.5 MTEB | Strong retrieval |

!!! note "Model Export"
    See [Model Export Guide](model-export.md) for exporting custom models to ONNX format.

## Performance

Benchmarks on NVIDIA GeForce MX250 (2GB VRAM):

| Operation | Items | Time | VRAM |
|-----------|-------|------|------|
| Crawl | 47 pages | 12.3s | ~15 MB |
| Index | 47 pages | 3.8s | 623 MB |
| Query (p50) | 1 | 23 ms | 625 MB |
| Query (p99) | 1 | 45 ms | 625 MB |

[Full Benchmarks →](performance.md){ .md-button }

## Documentation Index

### Getting Started

- [Installation](installation.md) — Prerequisites and setup
- [Architecture](architecture.md) — System design overview

### Components

- [Go Crawler](crawler.md) — CLI, configuration, internals
- [Semantic Search](semantic-search.md) — API, inference, storage

### Guides

- [Model Export](model-export.md) — Export HuggingFace models to ONNX
- [Testing](testing.md) — Run and write tests
- [Performance](performance.md) — Benchmarks and tuning

### Project

- [Contributing](contributing.md) — Development guidelines
- [Roadmap](roadmap.md) — Planned features and limitations

## Get Started

Ready to build your knowledge base?

[Install quarry :material-arrow-right:](installation.md){ .md-button .md-button--primary }
