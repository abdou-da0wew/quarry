#!/usr/bin/env python3
"""
Export all-MiniLM-L6-v2 to ONNX format for GPU inference.

This script exports the sentence-transformers model to ONNX format
with opset 14 for Pascal GPU compatibility (MX250).

Usage:
    python export_onnx.py [--output-dir OUTPUT_DIR]

Requirements:
    pip install transformers optimum[onnx] torch onnx
"""

import argparse
import os
import shutil
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Export model to ONNX format")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Output directory for ONNX model files",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="HuggingFace model ID to export",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=14,
        help="ONNX opset version (14 for Pascal GPU compatibility)",
    )
    parser.add_argument(
        "--optimize",
        type=str,
        default="O2",
        choices=["O1", "O2", "O3"],
        help="Optimization level",
    )
    args = parser.parse_args()

    print(f"Exporting model: {args.model}")
    print(f"Output directory: {args.output_dir}")
    print(f"ONNX opset version: {args.opset}")
    print(f"Optimization level: {args.optimize}")

    # Check dependencies
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
    except ImportError:
        print("ERROR: PyTorch not installed. Run: pip install torch")
        sys.exit(1)

    try:
        import transformers
        print(f"Transformers version: {transformers.__version__}")
    except ImportError:
        print("ERROR: Transformers not installed. Run: pip install transformers")
        sys.exit(1)

    try:
        import onnx
        print(f"ONNX version: {onnx.__version__}")
    except ImportError:
        print("ERROR: ONNX not installed. Run: pip install onnx")
        sys.exit(1)

    try:
        from optimum.exporters.onnx import main_export
        print("Optimum ONNX exporter available")
    except ImportError:
        print("ERROR: Optimum not installed. Run: pip install optimum[onnx]")
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export model
    print("\nExporting model to ONNX...")
    try:
        main_export(
            model_name_or_path=args.model,
            output=output_dir,
            opset=args.opset,
            task="feature-extraction",
            optimize=args.optimize,
        )
        print("Model export successful!")
    except Exception as e:
        print(f"ERROR: Failed to export model: {e}")
        sys.exit(1)

    # Rename model file if needed
    onnx_files = list(output_dir.glob("*.onnx"))
    if onnx_files:
        model_file = onnx_files[0]
        if model_file.name != "model.onnx":
            target = output_dir / "model.onnx"
            if target.exists():
                target.unlink()
            shutil.move(str(model_file), str(target))
            print(f"Renamed {model_file.name} to model.onnx")

    # Export tokenizer
    print("\nExporting tokenizer...")
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        tokenizer.save_pretrained(output_dir)
        print("Tokenizer export successful!")
    except Exception as e:
        print(f"ERROR: Failed to export tokenizer: {e}")
        sys.exit(1)

    # Verify export
    print("\nVerifying ONNX model...")
    try:
        import onnx
        model_path = output_dir / "model.onnx"
        if not model_path.exists():
            # Try to find any .onnx file
            onnx_files = list(output_dir.glob("*.onnx"))
            if not onnx_files:
                print("ERROR: No ONNX model file found")
                sys.exit(1)
            model_path = onnx_files[0]

        model = onnx.load(str(model_path))
        onnx.checker.check_model(model)

        print(f"\nModel Info:")
        print(f"  IR Version: {model.ir_version}")
        print(f"  Producer: {model.producer_name} {model.producer_version}")

        print(f"\nInputs:")
        for inp in model.graph.input:
            shape = [d.dim_value if d.dim_value else d.dim_param
                     for d in inp.type.tensor_type.shape.dim]
            print(f"  {inp.name}: {shape}")

        print(f"\nOutputs:")
        for out in model.graph.output:
            shape = [d.dim_value if d.dim_value else d.dim_param
                     for d in out.type.tensor_type.shape.dim]
            print(f"  {out.name}: {shape}")

        print("\nONNX model verified successfully!")

    except Exception as e:
        print(f"ERROR: ONNX verification failed: {e}")
        sys.exit(1)

    # List output files
    print(f"\nOutput files in {output_dir}:")
    for f in sorted(output_dir.iterdir()):
        size = f.stat().st_size
        if size > 1024 * 1024:
            size_str = f"{size / (1024 * 1024):.1f} MB"
        elif size > 1024:
            size_str = f"{size / 1024:.1f} KB"
        else:
            size_str = f"{size} B"
        print(f"  {f.name}: {size_str}")

    # Print usage instructions
    print("\n" + "=" * 60)
    print("Export complete!")
    print("=" * 60)
    print(f"\nModel files are in: {output_dir.absolute()}")
    print("\nUsage with Rust:")
    print(f"  config.model_path = \"{output_dir}/model.onnx\"")
    print(f"  config.tokenizer_path = \"{output_dir}/tokenizer.json\"")


if __name__ == "__main__":
    main()
