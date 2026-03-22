# Project Roadmap

Development roadmap and future plans for the quarry search system.

## Current Status

Quarry is a functional two-component search system with the following implemented features:

### Go Crawler (Complete)

- Concurrent worker pool with configurable parallelism
- robots.txt compliance with RFC 9309 parsing
- URL normalization and deduplication via sync.Map
- Streaming JSON output with bounded memory
- Inverted keyword index generation
- Graceful shutdown with goroutine leak prevention
- Configurable rate limiting and retries

### Rust Semantic Search (Complete)

- ONNX model loading with CUDA execution provider
- HNSW vector indexing via instant-distance
- Batch embedding with dynamic VRAM management
- CLI query tool with JSON output
- HTTP API with /search and /health endpoints
- Persistence with checksum verification
- Automatic GPU/CPU fallback

### Testing (Complete)

- Unit tests for both components
- Integration tests with mock HTTP server
- End-to-end pipeline test
- Goroutine leak detection (goleak)
- Memory leak verification (valgrind/heaptrack)
- CI workflow via GitHub Actions

## Planned Features

### Near-Term (Next Release)

**1. Incremental Indexing**

Currently, the indexer rebuilds the entire vector store on each run. Incremental indexing would:
- Detect new, modified, and deleted documents
- Add new embeddings without rebuilding
- Remove stale entries from HNSW index
- Reduce indexing time for small updates from minutes to seconds

*Rationale*: Production deployments run periodic crawls. Full reindexing wastes GPU time when only a few pages changed.

**2. Query Caching**

Implement an LRU cache for frequent queries:
- Cache embedding vectors for repeated queries
- Cache search results with TTL (5 minutes default)
- Configurable cache size (100 queries default)
- Cache hit ratio metrics in /health response

*Rationale*: Documentation sites often have common queries ("how to install", "troubleshooting"). Caching reduces GPU load and latency.

**3. Structured Filters**

Add metadata filtering to queries:
- Filter by URL path prefix (e.g., only search `/docs/`)
- Filter by date range (pages crawled after date)
- Filter by keyword presence in title
- Combine filters with semantic search

*Rationale*: Large documentation sites have distinct sections. Users often want to search within a specific area.

**4. API Authentication**

Add optional API key authentication:
- Bearer token authentication for /search
- Rate limiting per API key
- Key management CLI commands
- Anonymous access toggle

*Rationale*: Public deployments may need access control or rate limiting to prevent abuse.

### Medium-Term

**5. Distributed Indexing**

Support for multi-machine deployment:
- Shard documents across multiple indexers
- Coordinate via Redis or NATS
- Merge partial indexes on query
- Handle node failures gracefully

*Rationale*: Sites with 100K+ pages exceed single-machine capacity. Distributed indexing enables horizontal scaling.

**6. Real-time Index Updates**

WebSocket-based document ingestion:
- Accept new documents via WebSocket
- Embed and index in real-time
- Immediate availability in search
- Batch commits for durability

*Rationale*: Some use cases require near-instant search availability for new content (e.g., live documentation updates).

**7. Advanced Ranking Signals**

Combine semantic similarity with other signals:
- PageRank-style authority scoring
- Recency boost for recently updated pages
- Click-through rate from search logs
- Custom boost functions

*Rationale*: Pure semantic similarity may not always surface the most useful page. Multiple signals improve relevance.

**8. Multi-model Support**

Allow different embedding models for different corpora:
- Code search with CodeBERT
- Multilingual support with multilingual models
- Domain-specific fine-tuned models
- Model selection per query

*Rationale*: Different content types benefit from specialized embeddings. A single model cannot excel at all domains.

### Long-Term / Stretch

**9. Hybrid Search**

Combine keyword and semantic search:
- BM25 for exact term matching
- Semantic for conceptual similarity
- Reciprocal rank fusion for combined results
- Configurable weight between modes

*Rationale*: Keyword search is better for exact matches (error codes, function names). Hybrid provides best of both.

**10. Relevance Feedback**

Learn from user interactions:
- Click tracking for result quality
- Implicit feedback from dwell time
- Query reformulation suggestions
- Personalized ranking per user

*Rationale*: User behavior signals can significantly improve search quality over time.

## Known Limitations

Current limitations that may or may not be addressed:

| Limitation | Impact | Workaround |
|------------|--------|------------|
| No JavaScript rendering | Cannot crawl SPAs | Use server-side rendered sites or pre-render |
| Same-domain only | Cannot follow external links | Run separate crawlers per domain |
| Single GPU support | Cannot use multiple GPUs | Use a single powerful GPU |
| No incremental crawl | Must crawl entire site each time | Schedule off-peak full crawls |
| Fixed embedding model | Cannot change model after indexing | Delete index and re-index to change model |
| No access control | API is open by default | Deploy behind reverse proxy with auth |
| No query suggestions | No "did you mean" functionality | Future feature (medium-term) |
| No analytics | No search analytics dashboard | Parse server logs externally |
| Max 10K documents | HNSW parameters tuned for <10K docs | Adjust M/ef_construction for larger corpora |
| No parallel queries | GPU mutex serializes embedding | Use multiple API instances with load balancer |

## Not Planned

Features explicitly out of scope:

**Web UI**: Quarry provides API and CLI interfaces. Building a web frontend is left to downstream projects. The HTTP API is designed for easy integration with any frontend framework.

**Crawler for General Web**: The crawler is optimized for single-site documentation crawling. General-purpose web crawling (politeness policies, distributed frontier, schema extraction) is out of scope. Use dedicated crawlers like Scrapy or Apache Nutch for that use case.

**Fine-tuning Pipeline**: Model fine-tuning requires significant ML infrastructure and expertise. Quarry focuses on inference with pre-trained models. Use the HuggingFace ecosystem for fine-tuning.

**Enterprise Features**: SSO integration, audit logs, SLA guarantees, and enterprise support are not planned. Commercial search solutions (Elastic, Algolia) are better suited for enterprise deployments.

**Windows Native Support**: Development targets Linux and macOS. Windows users should use WSL2. The code may work on native Windows but is not tested or supported.

**Mobile Deployment**: The MX250 target suggests laptop/desktop deployment. Mobile inference (iOS/Android) would require significant architectural changes and is not planned.

## Version History

| Version | Date | Highlights |
|---------|------|------------|
| 1.0.0 | 2025-03-22 | Initial release with crawler and semantic search |
| | | - Worker pool crawler with robots.txt compliance |
| | | - ONNX embedding with CUDA execution provider |
| | | - HNSW vector search with HTTP API |
| | | - Complete test suite with E2E pipeline test |

## Contributing to the Roadmap

To propose a new feature or prioritize existing items:

1. Check existing issues and roadmap items
2. Open a discussion describing the use case
3. Provide concrete examples of how the feature would be used
4. Indicate if you're willing to implement the feature

Priority is given to features that:
- Benefit multiple users (not single-site customizations)
- Align with the core mission (documentation search on constrained hardware)
- Have clear success criteria
- Can be implemented without major architectural changes
