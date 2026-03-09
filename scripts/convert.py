#!/usr/bin/env python3
"""
Convert a YOLO .pt model to OpenVINO IR (.xml/.bin).

Notebook section used:
https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/yolov12-optimization/yolov12-object-detection.ipynb
"Convert model to OpenVINO IR"
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert YOLO .pt model to OpenVINO IR.")
    parser.add_argument("--model", required=True, help="Path to YOLO PyTorch model (.pt).")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run conversion even if target XML already exists.",
    )
    return parser.parse_args()


def expected_xml_path(pt_path: Path) -> Path:
    model_name = pt_path.stem
    return pt_path.parent / f"{model_name}_openvino_model" / f"{model_name}.xml"


def main() -> None:
    args = parse_args()
    pt_path = Path(args.model).expanduser().resolve()

    if not pt_path.exists():
        raise FileNotFoundError(f"Model file does not exist: {pt_path}")
    if pt_path.suffix.lower() != ".pt":
        raise ValueError(f"Expected a .pt file, got: {pt_path}")

    det_model_path = expected_xml_path(pt_path)
    print(f"Input model: {pt_path}")
    print(f"Target IR: {det_model_path}")

    if det_model_path.exists() and not args.force:
        print("IR already exists, skipping conversion. Use --force to convert again.")
        return

    # Same conversion call as in the notebook:
    # det_model.export(format="openvino", dynamic=True, half=True)
    det_model = YOLO(str(pt_path))
    det_model.export(format="openvino", dynamic=True, half=True)

    if not det_model_path.exists():
        raise RuntimeError(f"Conversion finished but XML not found at expected path: {det_model_path}")

    print(f"Conversion complete: {det_model_path}")


if __name__ == "__main__":
    main()
