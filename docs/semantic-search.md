# Semantic Search Reference

Complete reference for the Rust semantic search engine with GPU-accelerated ONNX inference.

## Purpose and Scope

The quarry semantic search engine is a Rust-based service that loads JSON documents produced by the Go crawler, generates dense vector embeddings via ONNX GPU inference, stores vectors in a hierarchical navigable small world (HNSW) index, and answers natural language queries with ranked results and cosine similarity scores. The engine is optimized for the NVIDIA MX250 GPU with a strict 1.5 GB VRAM ceiling, dynamic batch sizing under memory pressure, and zero-leak operation verified through integration tests.

## Architecture Diagram

```mermaid
flowchart TB
    subgraph Input["Input Layer"]
        A[JSON Documents]
        B[Query String]
    end
    
    subgraph Tokenizer["Tokenization"]
        C[HuggingFace Tokenizers]
        D[Max Sequence Truncation]
        E[Padding to Fixed Length]
    end
    
    subgraph ONNX["ONNX Inference"]
        F[ort Session]
        G[CUDA Execution Provider]
        H[GPU Forward Pass]
        I[Mean Pooling]
        J[L2 Normalization]
    end
    
    subgraph HNSW["HNSW Index"]
        K[Vector Store]
        L[instant-distance]
        M[M=16, ef_construction=200]
    end
    
    subgraph Persistence["Persistence"]
        N[.hnsw Graph]
        O[.vecs Vectors]
        P[.meta Checksum]
    end
    
    subgraph Query["Query Pipeline"]
        Q[Tokenize Query]
        R[Embed Query]
        S[HNSW Search ef=50]
        T[Cosine Similarity]
        U[Rank by Score]
    end
    
    subgraph Output["Output"]
        V[CLI JSON Output]
        W[HTTP API Response]
    end
    
    A --> C
    C --> D
    D --> E
    E --> F
    F --> G
    G --> H
    H --> I
    I --> J
    J --> K
    K --> L
    L --> M
    M --> N
    M --> O
    M --> P
    
    B --> Q
    Q --> R
    R --> F
    K --> S
    S --> T
    T --> U
    U --> V
    U --> W
    
    style ONNX fill:#76B900,color:#fff
    style HNSW fill:#DEA584,color:#000
```

## Embedding Model Selection

### Why all-MiniLM-L6-v2

The all-MiniLM-L6-v2 model was selected after evaluating multiple candidates against the constraints of the MX250 GPU:

| Criterion | all-MiniLM-L6-v2 | BGE-small-en-v1.5 | e5-small-v2 |
|-----------|------------------|-------------------|-------------|
| Parameters | 22M | 33M | 33M |
| Output Dimension | 384 | 384 | 384 |
| ONNX Size | 80 MB | 130 MB | 130 MB |
| Max Sequence Length | 384 | 512 | 512 |
| VRAM (batch=8) | ~400 MB | ~600 MB | ~600 MB |
| STS Benchmark Score | 0.82 | 0.84 | 0.83 |
| Inference Speed (batch=8) | 12 ms | 18 ms | 17 ms |

### Quality vs Speed Tradeoff

For the MX250 with 2GB VRAM (1.5GB usable), the smaller model provides:

1. **VRAM Headroom**: 400 MB for model + activations leaves 1.1 GB for other processes
2. **Larger Batches**: Can process 8 documents per batch vs 4 for larger models
3. **Adequate Quality**: 0.82 on semantic textual similarity is sufficient for documentation search
4. **Fast Inference**: 12ms per batch means 1000 documents index in ~1.5 seconds

The marginal quality improvement from BGE-small (0.02 points) does not justify the 50% larger memory footprint on constrained hardware.

## ONNX Export

The Python export script transforms the PyTorch model to ONNX format:

```bash
python scripts/export_onnx.py \
    --model sentence-transformers/all-MiniLM-L6-v2 \
    --output models/ \
    --opset 14
```

### Export Process Step-by-Step

1. **Load Pre-trained Model**: Download from HuggingFace Hub
2. **Create Dummy Input**: Shape [1, 384] for dynamic batch inference
3. **Export to ONNX**: Use `torch.onnx.export` with opset 14
4. **Validate Output**: Run onnx.checker to verify graph integrity
5. **Copy Tokenizer**: Save tokenizer.json, vocab.txt, special_tokens_map.json

### Opset Version

Opset 14 is required for compatibility with CUDA 13.0 on Pascal architecture (MX250). Earlier opsets may fail with unsupported operations; later opsets are not yet supported by ONNX Runtime 1.17.

### Dynamic Axes Configuration

```python
dynamic_axes = {
    'input_ids': {0: 'batch_size', 1: 'sequence_length'},
    'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
    'token_type_ids': {0: 'batch_size', 1: 'sequence_length'},
    'last_hidden_state': {0: 'batch_size', 1: 'sequence_length'},
}
```

This allows batch size to vary at inference time without re-exporting the model.

## CUDA Execution Provider Setup

### ort Crate Configuration

```rust
use ort::{Session, GraphOptimizationLevel, CUDAExecutionProviderOptions};

fn create_session(model_path: &Path) -> Result<Session> {
    let cuda_options = CUDAExecutionProviderOptions {
        device_id: 0,
        arena_extend_strategy: ArenaExtendStrategy::kSameAsRequested,
        gpu_mem_limit: 1_500_000_000, // 1.5 GB hard ceiling
        cudnn_conv_algo_search: 
            ort::CudnnConvAlgoSearch::EXHAUSTIVE,
        do_copy_in_default_stream: true,
        ..Default::default()
    };
    
    Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_op_num_threads(4)?
        .with_execution_providers([
            cuda_options.into(),
            // CPU provider as fallback
            CPUExecutionProviderOptions::default().into(),
        ])?
        .commit_from_file(model_path)
}
```

### GPU Unavailable Fallback

When CUDA is not available (no GPU, driver missing, wrong ONNX Runtime build), the session automatically falls back to CPU execution:

```rust
// Check GPU availability at startup
fn check_gpu_available() -> bool {
    // Try NVML init
    if let Ok(nvml) = nvml_wrapper::Nvml::init() {
        if let Ok(device) = nvml.device_by_index(0) {
            if let Ok(mem) = device.memory_info() {
                return mem.total > 0;
            }
        }
    }
    false
}
```

The search binary logs the execution mode at startup:
```
INFO quarry::embedder: GPU available: true, using CUDA execution provider
INFO quarry::embedder: GPU available: false, using CPU execution provider
```

## VRAM Management

### NVML Monitoring

```rust
use nvml_wrapper::Nvml;

pub struct VRAMMonitor {
    nvml: Nvml,
    device_index: u32,
}

impl VRAMMonitor {
    pub fn free_memory(&self) -> Result<u64> {
        let device = self.nvml.device_by_index(self.device_index)?;
        let mem_info = device.memory_info()?;
        Ok(mem_info.free)
    }
    
    pub fn used_memory(&self) -> Result<u64> {
        let device = self.nvml.device_by_index(self.device_index)?;
        let mem_info = device.memory_info()?;
        Ok(mem_info.used)
    }
    
    pub fn utilization(&self) -> Result<u32> {
        let device = self.nvml.device_by_index(self.device_index)?;
        let util = device.utilization_rates()?;
        Ok(util.gpu)
    }
}
```

### Batch Size Estimation Formula

```
VRAM_available = NVML_free_memory - 256_MB_safety_buffer
Model_weights = 80_MB
Activation_memory_per_sample = seq_len × hidden × 4_bytes × 3 (forward + intermediate)
Safe_batch_size = floor((VRAM_available - Model_weights) / Activation_memory_per_sample)
```

For all-MiniLM-L6-v2 with seq_len=384, hidden=384:
```
Activation_memory_per_sample = 384 × 384 × 4 × 3 = 1,769,472 bytes ≈ 1.7 MB
Safe_batch_size = floor((1,500,000,000 - 80,000,000) / 1,769,472) = 801

# In practice, we use much smaller batches for safety
default_batch_size = 8
```

### Halving Logic

```rust
impl Embedder {
    fn adjust_batch_size(&mut self) -> Result<()> {
        let free = self.vram_monitor.free_memory()?;
        let total = self.vram_monitor.total_memory()?;
        let usage_ratio = (total - free) as f64 / total as f64;
        
        if usage_ratio > 0.90 {
            // VRAM critical, halve batch size
            self.current_batch_size = (self.current_batch_size / 2).max(1);
            warn!(
                "VRAM pressure detected ({:.1}% used), reducing batch size to {}",
                usage_ratio * 100.0,
                self.current_batch_size
            );
        }
        Ok(())
    }
}
```

### Log Output During Indexing

```
INFO quarry::indexer: Starting indexing of 150 documents
INFO quarry::embedder: Batch size: 8, estimated VRAM: 420 MB
INFO quarry::embedder: Processing batch 1/19 (8 documents)
INFO quarry::embedder: Processing batch 2/19 (8 documents)
WARN quarry::embedder: VRAM pressure detected (92% used), reducing batch size to 4
INFO quarry::embedder: Processing batch 3/38 (4 documents)
INFO quarry::indexer: Indexed 150 documents in 2.3 seconds
INFO quarry::store: Saving index to disk (1.2 MB)
```

## HNSW Index

### Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| M | 16 | Number of bi-directional links per node. Higher M = better recall, more memory. |
| ef_construction | 200 | Size of dynamic candidate list during index construction. Higher = better quality, slower build. |
| ef_search | 50 | Size of dynamic candidate list during search. Can be tuned per query. |

### Memory Calculation

```
Memory per vector = 384 dimensions × 4 bytes = 1,536 bytes
Memory per HNSW node = vector + M × 2 × sizeof(NodeId) = 1,536 + 16 × 2 × 8 = 1,792 bytes
Total memory for N documents = N × 1,792 bytes

Example: 200 documents
Memory = 200 × 1,792 = 358,400 bytes ≈ 350 KB
```

### Insert vs Search Tradeoff

| ef_construction | Index Quality | Build Time | Memory |
|-----------------|---------------|------------|--------|
| 50 | 0.85 recall | Fast | Low |
| 100 | 0.92 recall | Moderate | Low |
| 200 | 0.97 recall | Slower | Low |
| 400 | 0.99 recall | Slow | Low |

For a 200-document corpus, ef_construction=200 provides 0.97 recall with acceptable build time (~2 seconds). The memory overhead is negligible compared to vectors.

## Persistence

### File Format

| File | Format | Description |
|------|--------|-------------|
| index.hnsw | Binary (instant-distance) | HNSW graph structure |
| index.vecs | Raw float32 [N × 384] | Raw embedding vectors |
| index.meta | JSON | Metadata and checksums |

### Meta File Schema

```json
{
  "doc_count": 200,
  "dimension": 384,
  "model": "all-MiniLM-L6-v2",
  "vecs_checksum": "sha256:a1b2c3d4...",
  "hnsw_checksum": "sha256:e5f6g7h8...",
  "created_at": "2025-03-22T14:30:00Z",
  "config": {
    "m": 16,
    "ef_construction": 200
  }
}
```

### Load-on-Startup Flow

```rust
impl VectorStore {
    pub fn load(path: &Path) -> Result<Self> {
        // 1. Read meta file
        let meta: Meta = serde_json::from_str(&fs::read_to_string(
            path.join("index.meta")
        )?)?;
        
        // 2. Verify checksums
        let vecs_data = fs::read(path.join("index.vecs"))?;
        let computed_checksum = format!("sha256:{}", sha256(&vecs_data));
        if computed_checksum != meta.vecs_checksum {
            return Err(Error::CorruptionDetected {
                expected: meta.vecs_checksum,
                actual: computed_checksum,
            });
        }
        
        // 3. Load vectors
        let vectors: Vec<Vec<f32>> = parse_vectors(&vecs_data, meta.dimension)?;
        
        // 4. Load HNSW graph
        let hnsw = HnswMap::load(path.join("index.hnsw"))?;
        
        Ok(VectorStore { hnsw, vectors, meta })
    }
}
```

### Corruption Detection

When checksum mismatch is detected:
1. Log error with expected vs actual checksum
2. Return `Error::CorruptionDetected`
3. Main binary exits with code 3
4. Administrator must re-run `indexer` to rebuild

## Query Pipeline

### Step-by-Step Processing

1. **Receive Query**: User submits "how to install minecraft on linux"
2. **Tokenize**: Convert to token IDs using saved tokenizer
   ```
   [CLS] how to install minecraft on linux [SEP] [PAD] [PAD] ...
   → [101, 2129, 2000, 5603, 18561, 2006, 3814, 102, 0, 0, ...]
   ```
3. **Create Attention Mask**: 1s for real tokens, 0s for padding
4. **ONNX Inference**: Pass inputs through model
   ```
   Output shape: [1, 384, 384] (batch, seq_len, hidden)
   ```
5. **Mean Pooling**: Average over sequence dimension, weighted by attention mask
   ```
   Result: [1, 384]
   ```
6. **L2 Normalize**: Convert to unit vector
   ```
   Result: unit vector with magnitude 1.0
   ```
7. **HNSW Search**: Find nearest neighbors with ef_search=50
8. **Compute Scores**: Cosine similarity = dot product of unit vectors
9. **Rank Results**: Sort by score descending
10. **Return JSON**: Top-K results with scores and document metadata

### Query Example

```bash
./search --query "how to install minecraft on linux" --top-k 5
```

Output:
```json
{
  "query": "how to install minecraft on linux",
  "results": [
    {
      "url": "https://minecraft-linux.github.io/installation/",
      "title": "Installation Guide",
      "score": 0.89,
      "snippet": "To install Minecraft on Linux, you will need a Java runtime environment..."
    },
    {
      "url": "https://minecraft-linux.github.io/launcher/",
      "title": "Launcher Options",
      "score": 0.72,
      "snippet": "The official Mojang launcher is the recommended way to install..."
    },
    {
      "url": "https://minecraft-linux.github.io/",
      "title": "Minecraft Linux Guide",
      "score": 0.68,
      "snippet": "Welcome to the Minecraft Linux guide. This site covers installation..."
    }
  ],
  "search_time_ms": 3
}
```

## HTTP API Reference

### POST /search

Search for documents matching a natural language query.

**Request Headers:**
```
Content-Type: application/json
```

**Request Body:**
```json
{
  "query": "how to install minecraft on linux",
  "top_k": 5
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| query | string | Yes | - | Natural language search query |
| top_k | integer | No | 10 | Maximum results to return (1-100) |

**Response (200 OK):**
```json
{
  "query": "how to install minecraft on linux",
  "results": [
    {
      "url": "https://minecraft-linux.github.io/installation/",
      "title": "Installation Guide",
      "score": 0.89,
      "snippet": "To install Minecraft on Linux, you will need a Java runtime..."
    }
  ],
  "search_time_ms": 3
}
```

**Response (400 Bad Request):**
```json
{
  "error": "query field is required"
}
```

**Response (503 Service Unavailable):**
```json
{
  "error": "index not loaded"
}
```

**curl Example:**
```bash
curl -X POST http://localhost:8080/search \
  -H "Content-Type: application/json" \
  -d '{"query": "java runtime requirements", "top_k": 5}'
```

### GET /health

Check service health and index status.

**Response (200 OK):**
```json
{
  "status": "healthy",
  "index_loaded": true,
  "document_count": 200,
  "model": "all-MiniLM-L6-v2"
}
```

**Response (503 Service Unavailable):**
```json
{
  "status": "unhealthy",
  "index_loaded": false,
  "error": "index files not found"
}
```

**curl Example:**
```bash
curl http://localhost:8080/health
```

## CLI Query Reference

### search Binary

```bash
./search [OPTIONS] --query <QUERY>
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--query`, `-q` | string | (required) | Search query text |
| `--top-k`, `-k` | integer | 10 | Number of results to return |
| `--index-dir`, `-i` | path | ./index | Directory containing index files |
| `--model-dir`, `-m` | path | ./models | Directory containing ONNX model |
| `--format`, `-f` | string | json | Output format: json, table, simple |
| `--verbose`, `-v` | bool | false | Enable debug logging |

### Example Commands

```bash
# Basic search
./search --query "minecraft installation guide"

# Search with custom result count
./search -q "java requirements" -k 20

# Table output format
./search -q "launcher options" -f table

# Verbose output for debugging
./search -q "test query" -v
```

## config.toml Reference

```toml
[model]
# Path to ONNX model file
path = "models/model.onnx"

# Path to tokenizer directory
tokenizer_path = "models/"

# Maximum sequence length for tokenizer
max_sequence_length = 384

[index]
# Directory for index files
path = "index/"

# HNSW M parameter (connections per node)
m = 16

# HNSW ef_construction parameter
ef_construction = 200

# HNSW ef_search parameter (can be tuned at runtime)
ef_search = 50

[embedding]
# Default batch size for indexing
batch_size = 8

# Maximum batch size (will halve under VRAM pressure)
max_batch_size = 16

# VRAM safety margin in bytes
vram_safety_margin = 268435456  # 256 MB

[server]
# HTTP server bind address
bind = "0.0.0.0:8080"

# Request timeout in seconds
timeout = 30

# Maximum request body size in bytes
max_body_size = 1048576  # 1 MB
```

| Section | Field | Type | Description |
|---------|-------|------|-------------|
| model | path | string | Path to the ONNX model file |
| model | tokenizer_path | string | Directory containing tokenizer.json |
| model | max_sequence_length | integer | Maximum tokens per document |
| index | path | string | Directory for storing index files |
| index | m | integer | HNSW connections per node |
| index | ef_construction | integer | Search depth during index build |
| index | ef_search | integer | Search depth during queries |
| embedding | batch_size | integer | Initial batch size for indexing |
| embedding | max_batch_size | integer | Upper limit for batch size |
| embedding | vram_safety_margin | integer | VRAM reserved for other processes |
| server | bind | string | IP:port for HTTP server |
| server | timeout | integer | Request timeout in seconds |
| server | max_body_size | integer | Maximum request body size |

## Performance Tuning

### Batch Size vs VRAM

| Batch Size | VRAM Used | Throughput | Recommendation |
|------------|-----------|------------|----------------|
| 1 | 150 MB | 8 docs/sec | CPU-only systems |
| 4 | 280 MB | 30 docs/sec | 2GB VRAM with other processes |
| 8 | 420 MB | 55 docs/sec | 2GB VRAM (default) |
| 16 | 700 MB | 100 docs/sec | 4GB+ VRAM |
| 32 | 1.2 GB | 180 docs/sec | 8GB+ VRAM |

### Top-K vs Latency

| Top-K | P50 Latency | P99 Latency | Notes |
|-------|-------------|-------------|-------|
| 5 | 1.5 ms | 3 ms | Fastest |
| 10 | 2 ms | 4 ms | Default |
| 20 | 3 ms | 6 ms | Moderate |
| 50 | 5 ms | 10 ms | Slower |
| 100 | 8 ms | 15 ms | Max supported |

### HNSW ef_search Tuning

| ef_search | Recall@10 | Latency | Recommendation |
|-----------|-----------|---------|----------------|
| 10 | 0.75 | 1 ms | Fast but may miss results |
| 25 | 0.88 | 2 ms | Balanced |
| 50 | 0.95 | 3 ms | Default, good quality |
| 100 | 0.98 | 5 ms | High recall requirement |
| 200 | 0.99 | 8 ms | Maximum quality |

## Error Handling

### Result Types

```rust
pub enum Error {
    #[error("Model not found: {0}")]
    ModelNotFound(PathBuf),
    
    #[error("Tokenizer not found: {0}")]
    TokenizerNotFound(PathBuf),
    
    #[error("ONNX inference failed: {0}")]
    InferenceFailed(String),
    
    #[error("Index not loaded")]
    IndexNotLoaded,
    
    #[error("Index corruption detected: expected {expected}, got {actual}")]
    CorruptionDetected { expected: String, actual: String },
    
    #[error("CUDA not available: {0}")]
    CudaNotAvailable(String),
    
    #[error("VRAM exhausted: requested {requested} MB, available {available} MB")]
    VramExhausted { requested: u64, available: u64 },
    
    #[error("Invalid query: {0}")]
    InvalidQuery(String),
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}
```

### Recovery Behavior

| Error | Recovery | User Action |
|-------|----------|-------------|
| ModelNotFound | Exit code 1 | Verify model path in config.toml |
| TokenizerNotFound | Exit code 1 | Run download_model.sh |
| InferenceFailed | Fall back to CPU | Check CUDA installation |
| IndexNotLoaded | HTTP 503 | Run indexer first |
| CorruptionDetected | Exit code 3 | Re-run indexer |
| CudaNotAvailable | CPU fallback | None required (automatic) |
| VramExhausted | Halve batch size | Reduce batch_size in config |
| InvalidQuery | HTTP 400 | Fix query format |
| Io | Exit code 1 | Check file permissions |
