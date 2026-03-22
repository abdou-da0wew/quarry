#!/bin/bash
# Download pre-exported ONNX model files for all-MiniLM-L6-v2
#
# Usage: ./scripts/download_model.sh [--output-dir OUTPUT_DIR]
#
# This script downloads the ONNX model from HuggingFace Hub.
# For custom export, use export_onnx.py instead.

set -e

OUTPUT_DIR="${1:-models}"
MODEL_ID="sentence-transformers/all-MiniLM-L6-v2"

echo "Downloading ONNX model files..."
echo "Model: $MODEL_ID"
echo "Output: $OUTPUT_DIR"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check for required tools
if command -v huggingface-cli &> /dev/null; then
    echo "Using huggingface-cli..."
    huggingface-cli download "$MODEL_ID" \
        --local-dir "$OUTPUT_DIR" \
        --include "model.onnx" "tokenizer.json" "tokenizer_config.json" "vocab.txt" "special_tokens_map.json"
elif command -v wget &> /dev/null; then
    echo "Using wget..."
    BASE_URL="https://huggingface.co/$MODEL_ID/resolve/main"
    
    wget -q --show-progress -O "$OUTPUT_DIR/tokenizer.json" "$BASE_URL/tokenizer.json" || true
    wget -q --show-progress -O "$OUTPUT_DIR/tokenizer_config.json" "$BASE_URL/tokenizer_config.json" || true
    wget -q --show-progress -O "$OUTPUT_DIR/vocab.txt" "$BASE_URL/vocab.txt" || true
    wget -q --show-progress -O "$OUTPUT_DIR/special_tokens_map.json" "$BASE_URL/special_tokens_map.json" || true
    
    # Check if pre-converted ONNX exists
    if wget --spider -q "$BASE_URL/model.onnx" 2>/dev/null; then
        wget -q --show-progress -O "$OUTPUT_DIR/model.onnx" "$BASE_URL/model.onnx"
    else
        echo "Pre-converted ONNX not available, running export..."
        python3 scripts/export_onnx.py --output-dir "$OUTPUT_DIR"
    fi
elif command -v curl &> /dev/null; then
    echo "Using curl..."
    BASE_URL="https://huggingface.co/$MODEL_ID/resolve/main"
    
    curl -L -o "$OUTPUT_DIR/tokenizer.json" "$BASE_URL/tokenizer.json" || true
    curl -L -o "$OUTPUT_DIR/tokenizer_config.json" "$BASE_URL/tokenizer_config.json" || true
    curl -L -o "$OUTPUT_DIR/vocab.txt" "$BASE_URL/vocab.txt" || true
    curl -L -o "$OUTPUT_DIR/special_tokens_map.json" "$BASE_URL/special_tokens_map.json" || true
    
    # Check if pre-converted ONNX exists
    if curl --output /dev/null --silent --head --fail "$BASE_URL/model.onnx"; then
        curl -L -o "$OUTPUT_DIR/model.onnx" "$BASE_URL/model.onnx"
    else
        echo "Pre-converted ONNX not available, running export..."
        python3 scripts/export_onnx.py --output-dir "$OUTPUT_DIR"
    fi
else
    echo "ERROR: Neither huggingface-cli, wget, nor curl found."
    echo "Please install one of these tools or run export_onnx.py manually."
    exit 1
fi

# Verify files exist
echo ""
echo "Verifying downloaded files..."

REQUIRED_FILES=("tokenizer.json")
MISSING=0

for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$OUTPUT_DIR/$file" ]; then
        SIZE=$(stat -f%z "$OUTPUT_DIR/$file" 2>/dev/null || stat -c%s "$OUTPUT_DIR/$file" 2>/dev/null || echo "unknown")
        echo "  ✓ $file ($SIZE bytes)"
    else
        echo "  ✗ $file (missing)"
        MISSING=1
    fi
done

if [ -f "$OUTPUT_DIR/model.onnx" ]; then
    SIZE=$(stat -f%z "$OUTPUT_DIR/model.onnx" 2>/dev/null || stat -c%s "$OUTPUT_DIR/model.onnx" 2>/dev/null || echo "unknown")
    echo "  ✓ model.onnx ($SIZE bytes)"
else
    echo "  ✗ model.onnx (missing - run export_onnx.py)"
    MISSING=1
fi

if [ $MISSING -eq 1 ]; then
    echo ""
    echo "Some files are missing. Try running export_onnx.py:"
    echo "  python3 scripts/export_onnx.py --output-dir $OUTPUT_DIR"
    exit 1
fi

echo ""
echo "Model files downloaded successfully to $OUTPUT_DIR"
echo ""
echo "Files:"
ls -lh "$OUTPUT_DIR"
