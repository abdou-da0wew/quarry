# Model Export Guide

Complete guide for exporting embedding models to ONNX format for use with the quarry semantic search engine.

## Why ONNX

ONNX (Open Neural Network Exchange) provides a portable, optimized format for neural model inference. The quarry semantic search engine uses ONNX for three key reasons:

First, ONNX Runtime eliminates the need for Python at inference time. The Rust embedder loads the model directly via the `ort` crate without any Python dependencies, resulting in faster startup, lower memory overhead, and simpler deployment.

Second, ONNX Runtime with CUDA execution provider delivers inference speeds comparable to native PyTorch while using significantly less VRAM. The optimized graph execution and memory pooling reduce overhead by 30-50% compared to eager PyTorch execution.

Third, the ONNX format is framework-agnostic. Models exported from PyTorch can be consumed by any ONNX-compatible runtime, enabling future migration to different inference engines or hardware accelerators without retraining.

## Supported Base Models

| Model Name | HuggingFace Hub ID | Params | Max Seq Len | VRAM (batch=8) | Tested | Notes |
|------------|-------------------|--------|-------------|----------------|--------|-------|
| all-MiniLM-L6-v2 | sentence-transformers/all-MiniLM-L6-v2 | 22M | 384 | 420 MB | Yes | Default, best for MX250 |
| BGE-small-en-v1.5 | BAAI/bge-small-en-v1.5 | 33M | 512 | 600 MB | Yes | Higher quality, more VRAM |
| e5-small-v2 | intfloat/e5-small-v2 | 33M | 512 | 600 MB | Yes | Similar to BGE |
| all-MiniLM-L12-v2 | sentence-transformers/all-MiniLM-L12-v2 | 33M | 384 | 580 MB | No | Larger MiniLM variant |
| bge-base-en-v1.5 | BAAI/bge-base-en-v1.5 | 109M | 512 | 1.2 GB | No | Requires 4GB+ VRAM |

For the MX250 with 2GB VRAM, all-MiniLM-L6-v2 is the recommended choice. BGE-small and e5-small can work with reduced batch sizes (batch=4) if higher quality is required.

## Export Walkthrough

### Step 1: Install Dependencies

```bash
# Create and activate virtual environment
python3 -m venv ~/.venv/quarry-export
source ~/.venv/quarry-export/bin/activate

# Install required packages
pip install torch transformers optimum onnx onnxruntime
```

### Step 2: Run Export Script

```bash
cd semantic-search
python scripts/export_onnx.py \
    --model sentence-transformers/all-MiniLM-L6-v2 \
    --output models/ \
    --opset 14
```

### Step 3: Verify Output

```bash
ls -la models/
```

Expected files:
```
model.onnx           # ONNX model (~80 MB)
tokenizer.json       # HuggingFace tokenizer config
vocab.txt            # Vocabulary file
special_tokens_map.json
tokenizer_config.json
config.json          # Model configuration
```

### Step 4: Validate Model

```bash
python -c "
import onnx
import onnxruntime as ort
import numpy as np

# Load and check model
model = onnx.load('models/model.onnx')
onnx.checker.check_model(model)
print('ONNX model is valid')

# Test inference
session = ort.InferenceSession('models/model.onnx')
input_ids = np.array([[101, 2023, 2003, 1037, 3231, 102]], dtype=np.int64)
attention_mask = np.array([[1, 1, 1, 1, 1, 1]], dtype=np.int64)

outputs = session.run(None, {
    'input_ids': input_ids,
    'attention_mask': attention_mask,
    'token_type_ids': np.zeros_like(input_ids)
})

print(f'Output shape: {outputs[0].shape}')
print('Inference test passed')
"
```

## Annotated Script Walkthrough

The `scripts/export_onnx.py` script performs the following operations:

### Section 1: Argument Parsing

```python
def parse_args():
    parser = argparse.ArgumentParser(
        description='Export HuggingFace model to ONNX format'
    )
    parser.add_argument(
        '--model', '-m',
        default='sentence-transformers/all-MiniLM-L6-v2',
        help='HuggingFace model name or path'
    )
    parser.add_argument(
        '--output', '-o',
        default='models/',
        help='Output directory for ONNX model and tokenizer'
    )
    parser.add_argument(
        '--opset',
        type=int,
        default=14,
        help='ONNX opset version (default: 14 for CUDA 13.0 compatibility)'
    )
    parser.add_argument(
        '--max-seq-length',
        type=int,
        default=384,
        help='Maximum sequence length for tokenizer'
    )
    return parser.parse_args()
```

This section defines command-line arguments for model name, output directory, opset version, and sequence length. The default opset 14 is required for CUDA 13.0 compatibility on Pascal architecture.

### Section 2: Model Loading

```python
def load_model(model_name: str):
    """Load model and tokenizer from HuggingFace Hub."""
    print(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()  # Set to evaluation mode
    
    return model, tokenizer
```

The model is loaded using HuggingFace's `AutoModel` and `AutoTokenizer` classes. Setting `eval()` mode disables dropout layers that are not needed for inference.

### Section 3: Dummy Input Creation

```python
def create_dummy_input(tokenizer, max_length: int = 384):
    """Create dummy inputs for ONNX export tracing."""
    # Use a representative sentence
    dummy_text = "This is a sample sentence for model tracing."
    
    inputs = tokenizer(
        dummy_text,
        padding='max_length',
        max_length=max_length,
        truncation=True,
        return_tensors='pt'
    )
    
    return {
        'input_ids': inputs['input_ids'],
        'attention_mask': inputs['attention_mask'],
        'token_type_ids': inputs.get('token_type_ids', torch.zeros_like(inputs['input_ids']))
    }
```

The dummy input is used by PyTorch's JIT tracer to record the computational graph. Using `max_length` padding ensures consistent input shapes for the ONNX graph, while dynamic axes allow variation at runtime.

### Section 4: Dynamic Axes Definition

```python
def get_dynamic_axes():
    """Define dynamic axes for flexible batch and sequence dimensions."""
    return {
        'input_ids': {0: 'batch_size', 1: 'sequence_length'},
        'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
        'token_type_ids': {0: 'batch_size', 1: 'sequence_length'},
        'last_hidden_state': {0: 'batch_size', 1: 'sequence_length'},
        'pooler_output': {0: 'batch_size'}
    }
```

Dynamic axes tell ONNX that the batch size and sequence length can vary at inference time. Without this, the exported model would only accept inputs of exactly the shape used during tracing.

### Section 5: ONNX Export

```python
def export_to_onnx(model, inputs, output_path: str, opset: int, dynamic_axes: dict):
    """Export PyTorch model to ONNX format."""
    print(f"Exporting to ONNX (opset {opset})...")
    
    torch.onnx.export(
        model,
        (
            inputs['input_ids'],
            inputs['attention_mask'],
            inputs['token_type_ids']
        ),
        output_path,
        input_names=['input_ids', 'attention_mask', 'token_type_ids'],
        output_names=['last_hidden_state', 'pooler_output'],
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
        export_params=True
    )
    
    print(f"Model exported to: {output_path}")
```

The export function uses PyTorch's built-in `torch.onnx.export`. Key parameters:
- `do_constant_folding=True`: Optimizes the graph by folding constant operations
- `export_params=True`: Embeds model weights in the ONNX file
- `opset_version`: Determines which ONNX operators are available

### Section 6: Model Validation

```python
def validate_onnx_model(model_path: str, inputs: dict):
    """Validate the exported ONNX model."""
    print("Validating ONNX model...")
    
    # Check model structure
    model = onnx.load(model_path)
    onnx.checker.check_model(model)
    print("  ✓ ONNX structure is valid")
    
    # Compare PyTorch and ONNX outputs
    session = ort.InferenceSession(model_path)
    
    onnx_outputs = session.run(
        None,
        {
            'input_ids': inputs['input_ids'].numpy(),
            'attention_mask': inputs['attention_mask'].numpy(),
            'token_type_ids': inputs['token_type_ids'].numpy()
        }
    )
    
    print("  ✓ Inference successful")
    print(f"  ✓ Output shape: {onnx_outputs[0].shape}")
```

The validation step verifies that:
1. The ONNX graph is structurally valid
2. The model can be loaded by ONNX Runtime
3. Inference produces outputs of the expected shape

### Section 7: Tokenizer Export

```python
def save_tokenizer(tokenizer, output_dir: str):
    """Save tokenizer files alongside the ONNX model."""
    print("Saving tokenizer...")
    tokenizer.save_pretrained(output_dir)
    
    # List saved files
    files = list(Path(output_dir).glob('*'))
    for f in files:
        print(f"  ✓ {f.name}")
```

The tokenizer must be saved in HuggingFace format so that the Rust `tokenizers` crate can load it. The saved files include vocabulary, special tokens, and tokenizer configuration.

## Dynamic Axes Explained

### What Dynamic Axes Are

In ONNX, tensors have fixed shapes by default. A model exported with input shape `[1, 384]` can only process single samples of exactly 384 tokens. Dynamic axes specify which dimensions can vary at runtime.

### Why Quarry Uses Dynamic Axes

The semantic search engine processes documents in batches for efficiency. During indexing, documents are grouped into batches of 8 (default) to maximize GPU utilization. During querying, only a single text is processed at a time. Dynamic axes allow the same ONNX model to handle both cases without modification.

### How Batch Size Stays Flexible

With dynamic axes defined as `{0: 'batch_size'}`, ONNX Runtime accepts any batch size at inference time:

```rust
// Batch of 8 for indexing
let input_ids = Array2::from_shape_vec((8, 384), ids)?;

// Single query
let input_ids = Array2::from_shape_vec((1, 384), ids)?;
```

The model automatically handles both shapes without recompilation or re-export.

## Output Verification

### Verify Model Properties

```bash
python -c "
import onnx
model = onnx.load('models/model.onnx')

print('=== Model Inputs ===')
for inp in model.graph.input:
    print(f'  {inp.name}: {[d.dim_value if d.dim_value else \"dynamic\" for d in inp.type.tensor_type.shape.dim]}')

print('=== Model Outputs ===')
for out in model.graph.output:
    print(f'  {out.name}: {[d.dim_value if d.dim_value else \"dynamic\" for d in out.type.tensor_type.shape.dim]}')

print(f'=== Opset Version: {model.opset_import[0].version} ===')
"
```

Expected output:
```
=== Model Inputs ===
  input_ids: ['dynamic', 'dynamic']
  attention_mask: ['dynamic', 'dynamic']
  token_type_ids: ['dynamic', 'dynamic']
=== Model Outputs ===
  last_hidden_state: ['dynamic', 'dynamic', 384]
  pooler_output: ['dynamic', 384]
=== Opset Version: 14 ===
```

### Verify Embedding Quality

```bash
python -c "
from transformers import AutoTokenizer, AutoModel
import onnxruntime as ort
import numpy as np

# Load PyTorch model
tokenizer = AutoTokenizer.from_pretrained('models/')
pt_model = AutoModel.from_pretrained('models/')
pt_model.eval()

# Load ONNX model
ort_session = ort.InferenceSession('models/model.onnx')

# Test sentence
text = 'How to install Minecraft on Linux'

# PyTorch embedding
pt_inputs = tokenizer(text, return_tensors='pt', padding='max_length', max_length=384)
pt_output = pt_model(**pt_inputs).last_hidden_state.mean(dim=1).detach().numpy()

# ONNX embedding
ort_inputs = {
    'input_ids': pt_inputs['input_ids'].numpy(),
    'attention_mask': pt_inputs['attention_mask'].numpy(),
    'token_type_ids': pt_inputs.get('token_type_ids', np.zeros_like(pt_inputs['input_ids'])).numpy()
}
ort_output = ort_session.run(None, ort_inputs)[0].mean(axis=1)

# Compare
cosine_sim = np.dot(pt_output[0], ort_output[0]) / (np.linalg.norm(pt_output[0]) * np.linalg.norm(ort_output[0]))
print(f'Cosine similarity between PyTorch and ONNX embeddings: {cosine_sim:.6f}')
print('✓ Embeddings match!' if cosine_sim > 0.999 else '✗ Embeddings differ significantly')
"
```

Expected output:
```
Cosine similarity between PyTorch and ONNX embeddings: 0.999987
✓ Embeddings match!
```

## Adding a New Model

### Step 1: Choose Model

Select a model from the HuggingFace Hub that meets these criteria:
- Transformer-based encoder (BERT, RoBERTa, DistilBERT, etc.)
- Outputs dense embeddings (not sequence-to-sequence)
- Has a sentence-transformers variant or produces good sentence embeddings

### Step 2: Export the Model

```bash
python scripts/export_onnx.py \
    --model BAAI/bge-small-en-v1.5 \
    --output models/bge-small/ \
    --opset 14 \
    --max-seq-length 512
```

### Step 3: Update Configuration

Edit `config.toml`:

```toml
[model]
path = "models/bge-small/model.onnx"
tokenizer_path = "models/bge-small/"
max_sequence_length = 512
```

### Step 4: Rebuild Index

```bash
./indexer --config config.toml --input ../crawler/output/pages/
```

The index must be rebuilt when changing models because embeddings from different models are not comparable.

### Step 5: Verify Search Quality

Run test queries and compare result quality:

```bash
./search -q "test query" -v
```

## Troubleshooting

### Opset Version Errors

**Symptom:**
```
RuntimeError: Exporting the operator 'aten::layer_norm' to ONNX opset version 11 is not supported
```

**Cause:** The model uses operations not available in the specified opset version.

**Fix:** Use a higher opset version:
```bash
python scripts/export_onnx.py --opset 14 --model sentence-transformers/all-MiniLM-L6-v2
```

For newer models (2024+), opset 17 or 18 may be required.

### Shape Mismatch

**Symptom:**
```
onnxruntime.capi.onnxruntime_pybind11_state.InvalidArgument: Got invalid dimensions for input
```

**Cause:** The tokenizer and model have different maximum sequence lengths.

**Fix:** Ensure `--max-seq-length` matches the model's expected input:
```bash
# all-MiniLM-L6-v2: 384
python scripts/export_onnx.py --max-seq-length 384

# BGE-small: 512
python scripts/export_onnx.py --max-seq-length 512
```

### CUDA OOM During Export

**Symptom:**
```
RuntimeError: CUDA out of memory
```

**Cause:** The GPU doesn't have enough memory for export (model + gradients).

**Fix:** Use CPU for export:
```bash
CUDA_VISIBLE_DEVICES="" python scripts/export_onnx.py --model ...
```

Export only needs CPU; GPU is required only at inference time.

### Tokenizer Vocab Mismatch

**Symptom:**
```
Error: Tokenizer vocabulary size (30522) does not match model (30524)
```

**Cause:** The tokenizer files were saved from a different model.

**Fix:** Ensure tokenizer and model are loaded from the same source:
```bash
python scripts/export_onnx.py --model sentence-transformers/all-MiniLM-L6-v2 --output models/
```

Do not mix tokenizer files between different models.

### Output Dimension Mismatch

**Symptom:**
```
Error: Vector dimension mismatch: expected 384, got 768
```

**Cause:** The new model has a different embedding dimension than the existing index.

**Fix:** Delete the old index and rebuild:
```bash
rm -rf index/
./indexer --config config.toml --input ../crawler/output/pages/
```

### Model Not Found

**Symptom:**
```
Error: Model not found: models/model.onnx
```

**Cause:** The export script didn't run or the output path is wrong.

**Fix:** Verify the export completed and check paths:
```bash
ls models/model.onnx
# If missing, re-run export
python scripts/export_onnx.py --output models/
```
