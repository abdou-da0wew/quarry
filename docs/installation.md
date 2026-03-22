# Installation

Complete installation guide for the quarry search system on Linux, macOS, and Windows WSL2.

## Prerequisites

| Tool | Version | Required/Optional | Purpose |
|------|---------|-------------------|---------|
| Go | 1.22+ | Required | Build and run the crawler |
| Rust | 1.75+ | Required | Build the semantic search engine |
| CUDA Toolkit | 13.0 | Required | GPU inference for ONNX Runtime |
| NVIDIA Driver | 580.95.05+ | Required | CUDA compatibility |
| Python | 3.10+ | Optional | ONNX model export only |
| Git | 2.40+ | Required | Clone the repository |
| pkg-config | 0.29+ | Required (Linux) | Library discovery for ONNX |

## Hardware Requirements

| Component | Minimum | Recommended | Notes |
|-----------|---------|-------------|-------|
| GPU | MX250 (2GB) | GTX 1650 (4GB) | MX250 supports CUDA Compute 6.1 |
| VRAM | 1.5 GB usable | 3 GB usable | Hard ceiling for ONNX arena |
| System RAM | 4 GB | 8 GB | Crawler buffers, HNSW index |
| Disk Space | 500 MB | 1 GB | Model, vectors, JSON documents |
| CPU | 2 cores | 4 cores | Worker parallelism |

### Verify GPU Compatibility

```bash
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
```

Expected output for MX250:
```
name, memory.total [MiB], compute_cap
NVIDIA GeForce MX250, 2048 MiB, 6.1
```

## Section A: Install Go

### Linux (Ubuntu/Debian)

```bash
# Remove any existing Go installation
sudo rm -rf /usr/local/go

# Download Go 1.22
wget https://go.dev/dl/go1.22.0.linux-amd64.tar.gz

# Extract to /usr/local
sudo tar -C /usr/local -xzf go1.22.0.linux-amd64.tar.gz

# Add to PATH
echo 'export PATH=$PATH:/usr/local/go/bin' >> ~/.bashrc
echo 'export GOPATH=$HOME/go' >> ~/.bashrc
echo 'export PATH=$PATH:$GOPATH/bin' >> ~/.bashrc
source ~/.bashrc

# Verify
go version
```

### macOS

```bash
# Using Homebrew
brew install go

# Verify
go version
```

### Windows WSL2

```bash
# Inside WSL2 Ubuntu
wget https://go.dev/dl/go1.22.0.linux-amd64.tar.gz
sudo tar -C /usr/local -xzf go1.22.0.linux-amd64.tar.gz
echo 'export PATH=$PATH:/usr/local/go/bin' >> ~/.bashrc
source ~/.bashrc
go version
```

## Section B: Install Rust

```bash
# Install rustup
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Select option 1 (default installation)
# After installation completes:
source $HOME/.cargo/env

# Set stable toolchain as default
rustup default stable

# Verify
rustc --version
cargo --version
```

### Additional Rust Components

```bash
# Install clippy for linting
rustup component add clippy

# Install rustfmt for formatting
rustup component add rustfmt
```

## Section C: Install CUDA Toolkit

### Linux (Ubuntu/Debian)

```bash
# Add NVIDIA package repository
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update

# Install CUDA Toolkit 13.0
sudo apt install cuda-toolkit-13-0

# Add to PATH
echo 'export PATH=/usr/local/cuda-13.0/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify
nvcc --version
```

Expected output:
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Cuda compilation tools, release 13.0, V13.0.xxx
```

### Verify CUDA with NVIDIA Driver

```bash
nvidia-smi
```

The driver version must support CUDA 13.0. Check NVIDIA's CUDA compatibility matrix if the driver is too old.

### macOS

macOS does not support CUDA. The semantic search engine will fall back to CPU inference automatically. No CUDA installation is required.

### Windows WSL2

```bash
# Inside WSL2, install CUDA Toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install cuda-toolkit-13-0

# Add to PATH
echo 'export PATH=/usr/local/cuda-13.0/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify (requires NVIDIA driver installed on Windows host)
nvidia-smi
```

## Section D: Install ONNX Runtime

The Rust semantic search uses the `ort` crate which links against ONNX Runtime. You must install the ONNX Runtime shared library with CUDA support.

### Linux (Ubuntu/Debian)

```bash
# Download ONNX Runtime with CUDA
cd /tmp
wget https://github.com/microsoft/onnxruntime/releases/download/v1.17.0/onnxruntime-linux-x64-gpu-1.17.0.tgz
tar -xzf onnxruntime-linux-x64-gpu-1.17.0.tgz

# Install to system location
sudo cp -r onnxruntime-linux-x64-gpu-1.17.0/include/* /usr/local/include/
sudo cp -r onnxruntime-linux-x64-gpu-1.17.0/lib/* /usr/local/lib/

# Update library cache
sudo ldconfig

# Set environment variables for the ort crate
echo 'export ORT_LIB_LOCATION=/usr/local/lib/libonnxruntime.so' >> ~/.bashrc
source ~/.bashrc
```

### Verify ONNX Runtime

```bash
# Check that the library is loadable
ldconfig -p | grep onnxruntime
```

Expected output:
```
libonnxruntime.so.1.17.0 (libc6,x86-64) => /usr/local/lib/libonnxruntime.so.1.17.0
```

### CPU-Only Fallback

If GPU is unavailable, use the CPU-only build:

```bash
cd /tmp
wget https://github.com/microsoft/onnxruntime/releases/download/v1.17.0/onnxruntime-linux-x64-1.17.0.tgz
tar -xzf onnxruntime-linux-x64-1.17.0.tgz
sudo cp -r onnxruntime-linux-x64-1.17.0/lib/* /usr/local/lib/
sudo ldconfig
```

The Rust code will automatically detect GPU availability and fall back to CPU.

### macOS

```bash
# CPU-only (no CUDA on macOS)
cd /tmp
wget https://github.com/microsoft/onnxruntime/releases/download/v1.17.0/onnxruntime-osx-arm64-1.17.0.tgz
tar -xzf onnxruntime-osx-arm64-1.17.0.tgz
sudo cp -r onnxruntime-osx-arm64-1.17.0/lib/* /usr/local/lib/
```

## Section E: Install Python Dependencies

Python is only required for the ONNX model export step. The runtime uses no Python.

```bash
# Create virtual environment (recommended)
python3 -m venv ~/.venv/quarry
source ~/.venv/quarry/bin/activate

# Install dependencies
pip install torch transformers optimum onnx onnxruntime-gpu

# For CPU-only export
pip install torch transformers optimum onnx onnxruntime
```

### Verify Python Installation

```bash
python -c "from transformers import AutoModel; print('OK')"
python -c "import onnxruntime as ort; print(f'ONNX Runtime: {ort.__version__}')"
```

## Section F: Clone and Verify Repository

```bash
# Clone the repository
git clone https://github.com/your-org/quarry.git
cd quarry

# Verify directory structure
ls -la
```

Expected output:
```
crawler/           # Go crawler source
semantic-search/   # Rust semantic search source
tests/             # Integration tests
docs/              # Documentation
spec.md            # Architecture specification
```

### Verify Go Module

```bash
cd crawler
go mod download
go mod verify
cd ..
```

### Verify Rust Workspace

```bash
cd semantic-search
cargo check
cd ..
```

## Section G: Export ONNX Model

The embedding model must be exported to ONNX format before the semantic search engine can use it.

```bash
cd semantic-search

# Download the model and tokenizer
./scripts/download_model.sh

# Export to ONNX
python scripts/export_onnx.py --model sentence-transformers/all-MiniLM-L6-v2 --output models/

# Verify the export
ls -la models/
```

Expected output:
```
model.onnx         # ONNX model (~80 MB)
tokenizer.json     # HuggingFace tokenizer
vocab.txt          # Vocabulary file
special_tokens_map.json
```

### Verify ONNX Model

```bash
python -c "
import onnx
model = onnx.load('models/model.onnx')
onnx.checker.check_model(model)
print('ONNX model is valid')
print(f'Inputs: {[i.name for i in model.graph.input]}')
print(f'Outputs: {[o.name for o in model.graph.output]}')
"
```

## Section H: Build Go Crawler

```bash
cd crawler

# Build the binary
go build -o bin/crawler ./cmd/crawler

# Verify
./bin/crawler --help
```

Expected output:
```
Usage: crawler [options]

Options:
  --entry URL         Entry point URL (required)
  --workers N         Number of concurrent workers (default: 10)
  --timeout MS        HTTP timeout in milliseconds (default: 30000)
  --output-dir DIR    Output directory for JSON files (default: ./output)
  --max-pages N       Maximum pages to crawl (default: 1000)
  --delay-ms MS       Delay between requests to same host (default: 100)
  --help              Show this help message
```

### Run Tests

```bash
# Unit tests with race detector
go test ./... -race -v

# With goroutine leak detection
go test ./... -race -tags=goleak
```

## Section I: Build Rust Workspace

```bash
cd semantic-search

# Build all binaries in release mode
cargo build --release

# Verify binaries
ls -la target/release/
```

Expected binaries:
```
indexer      # Build vector index from JSON documents
search       # CLI query tool
search-api   # HTTP API server
```

### Run Tests

```bash
# Unit tests
cargo test --workspace

# With clippy linting
cargo clippy -- -D warnings

# Integration tests (requires model and GPU)
cargo test --workspace --test integration_tests
```

## Verification Checklist

Complete all 8 checks to confirm a working installation:

1. **Go Version**
   ```bash
   go version
   # Expected: go version go1.22.x linux/amd64
   ```

2. **Rust Version**
   ```bash
   rustc --version
   # Expected: rustc 1.75.x (or later)
   ```

3. **CUDA Available**
   ```bash
   nvidia-smi
   # Expected: GPU listing with driver and CUDA version
   ```

4. **ONNX Runtime Linked**
   ```bash
   cd semantic-search && cargo test --lib -- --test-threads=1 2>&1 | grep -i "onnx\|ort"
   # Expected: No linking errors
   ```

5. **Model Exported**
   ```bash
   ls -la semantic-search/models/model.onnx
   # Expected: File exists, ~80 MB
   ```

6. **Go Crawler Builds**
   ```bash
   cd crawler && go build ./cmd/crawler
   # Expected: No errors
   ```

7. **Rust Workspace Builds**
   ```bash
   cd semantic-search && cargo build --release
   # Expected: Compiles without errors
   ```

8. **End-to-End Test**
   ```bash
   ./tests/e2e/pipeline_test.sh
   # Expected: All assertions pass, exit code 0
   ```

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| `nvidia-smi: command not found` | NVIDIA driver not installed | Install the NVIDIA driver for your GPU from NVIDIA's website |
| `nvcc: command not found` | CUDA Toolkit not in PATH | Add `/usr/local/cuda/bin` to PATH in your shell profile |
| `CUDA out of memory` during indexing | Batch size too large for VRAM | Reduce `batch_size` in config.toml to 4 or 2 |
| `libonnxruntime.so: cannot open shared object file` | ONNX Runtime not installed or not in library path | Install ONNX Runtime and run `sudo ldconfig` |
| `cannot find -lcuda` | CUDA stubs not found during Rust build | Set `LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:$LD_LIBRARY_PATH` |
| GPU not detected by ONNX Runtime | ONNX Runtime built without CUDA | Re-install the GPU variant of ONNX Runtime (`onnxruntime-linux-x64-gpu`) |
| `robots.txt blocking all requests` | User-agent not allowed | The crawler respects robots.txt by design; check the site's robots.txt |
| Goroutine leak detected in tests | Incomplete shutdown | Check that all channels are closed and contexts are cancelled in the test |
| `tokenizer.json not found` | Model files not downloaded | Run `./scripts/download_model.sh` or check models/ directory |
| `HNSW store corruption detected` | Incomplete write during shutdown | Delete `.hnsw`, `.vecs`, `.meta` files and re-run indexer |
| Go build fails on Windows WSL2 | Case-sensitive filesystem issues | Ensure repository is cloned within WSL2 filesystem, not mounted from Windows |
| `ort::Error: Failed to load model` | ONNX model incompatible with opset version | Re-export model with opset 14: `python scripts/export_onnx.py --opset 14` |
| `CUDA driver version is insufficient` | Driver too old for CUDA 13.0 | Update NVIDIA driver to version supporting CUDA 13.0 |
| Embedding dimension mismatch | Model changed after index built | Re-run indexer after changing the embedding model |
| HTTP 429 during crawl | Rate limited by target site | Increase `--delay-ms` to 500 or higher |
| `goleak: found unexpected goroutine` | Background goroutine not cleaned up | Ensure worker pool shutdown completes before test ends |
| `ort::Error: Got invalid dimensions for input` | Token sequence exceeds max length | Reduce batch size or truncate long documents before embedding |
| Slow indexing on CPU | No GPU available | CPU fallback is 10-50x slower; consider using a machine with GPU |
| `Address already in use` for API port | Another process on :8080 | Kill the existing process or change port in config.toml |
