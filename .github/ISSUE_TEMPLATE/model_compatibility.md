---
name: Model Compatibility Report
about: Report compatibility with a new embedding model
title: '[MODEL] '
labels: model-compat
assignees: ''
---

## Model Information

- **Model Name**: [e.g., all-MiniLM-L6-v2]
- **HuggingFace Hub ID**: [e.g., sentence-transformers/all-MiniLM-L6-v2]
- **Parameter Count**: [e.g., 22M]
- **Max Sequence Length**: [e.g., 384]
- **Embedding Dimension**: [e.g., 384]

## Export Information

- **ONNX Opset Used**: [e.g., 14]
- **ONNX Model Size**: [e.g., 80 MB]
- **Export Script Version**: [e.g., quarry 1.0.0]

## Hardware Tested

- **GPU Model**: [e.g., NVIDIA GeForce MX250]
- **VRAM**: [e.g., 2048 MiB]
- **CUDA Version**: [e.g., 13.0]

## VRAM Usage

Tested with `batch_size=8`:

- **Idle VRAM**: [e.g., 200 MB]
- **Model Load VRAM**: [e.g., 280 MB]
- **Peak VRAM during indexing**: [e.g., 420 MB]
- **Largest successful batch size**: [e.g., 16]

## Export Steps Used

```bash
# Commands used to export the model
python scripts/export_onnx.py \
    --model [hub-id] \
    --output models/[model-name]/ \
    --opset [version]
```

## Search Quality Notes

### Test Queries and Results

| Query | Expected Top Result | Actual Top Result | Score |
|-------|---------------------|-------------------|-------|
| "how to install minecraft" | installation.html | [result] | [score] |
| "java requirements" | installation.html | [result] | [score] |

### Quality Assessment

- [ ] Results match or exceed all-MiniLM-L6-v2 baseline
- [ ] Results are acceptable but not better
- [ ] Results are noticeably worse

**Additional Notes**: [Any observations about result quality]

## Issues Encountered

Describe any problems during export, indexing, or querying:

- [ ] None - model works perfectly
- [ ] Export script required modifications
- [ ] CUDA compatibility issues
- [ ] OOM errors during indexing
- [ ] Tokenizer mismatch errors
- [ ] Other: [describe]

**Details**: [Explain any issues and how you resolved them]

## Configuration Changes

If you needed to modify `config.toml` for this model:

```toml
[model]
path = "models/[model-name]/model.onnx"
tokenizer_path = "models/[model-name]/"
max_sequence_length = [value]

[embedding]
batch_size = [value]
```

## Verification

- [ ] ONNX model passes `onnx.checker.check_model()`
- [ ] Embedding outputs match PyTorch reference (cosine sim > 0.99)
- [ ] Successfully indexed 100+ documents
- [ ] Search queries return relevant results

## Additional Context

Any other information that might be helpful:

- Links to model paper or documentation
- Special tokenizer requirements
- Known limitations of the model
