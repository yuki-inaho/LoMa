"""Prune exported LoMa graphs and make their IR header ORT 1.17 compatible."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import onnx
from onnx import checker
from onnx.utils import extract_model


def finalize_model(source: Path, destination: Path, max_ir_version: int = 9) -> Path:
    """Keep graph inputs/outputs, prune unused data, and cap the ONNX IR header."""
    source = source.expanduser().resolve()
    destination = destination.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if max_ir_version <= 0:
        raise ValueError("max_ir_version must be positive")

    original = onnx.load(source)
    checker.check_model(original)
    inputs = [value.name for value in original.graph.input]
    outputs = [value.name for value in original.graph.output]
    destination.parent.mkdir(parents=True, exist_ok=True)

    temporary_fd, temporary_name = tempfile.mkstemp(
        prefix=destination.stem + ".", suffix=".onnx", dir=destination.parent
    )
    os.close(temporary_fd)
    temporary = Path(temporary_name)
    temporary.unlink()
    try:
        extract_model(str(source), str(temporary), inputs, outputs, check_model=True)
        model = onnx.load(temporary)
        model.ir_version = min(model.ir_version, max_ir_version)
        checker.check_model(model)
        onnx.save_model(model, temporary, save_as_external_data=False)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+", type=Path)
    parser.add_argument("--in-place", action="store_true")
    parser.add_argument("--suffix", default="_ort117")
    parser.add_argument("--max-ir-version", type=int, default=9)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for source in args.models:
        destination = (
            source
            if args.in_place
            else source.with_name(source.stem + args.suffix + source.suffix)
        )
        before = source.stat().st_size
        output = finalize_model(source, destination, args.max_ir_version)
        print(
            f"{output}: {before} -> {output.stat().st_size} bytes, "
            f"IR <= {args.max_ir_version}"
        )


if __name__ == "__main__":
    main()
