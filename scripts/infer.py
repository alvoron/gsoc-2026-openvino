#!/usr/bin/env python3
"""
Run object detection on a single image with YOLO model and save
an annotated output image with bounding boxes and labels.

Based on the OpenVINO notebook flow:
https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/yolov12-optimization/yolov12-object-detection.ipynb
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO inference on one image (.pt or OpenVINO IR).")
    parser.add_argument(
        "--model",
        required=True,
        help="Path to a YOLO model: .pt, OpenVINO .xml, or model directory with one .xml.",
    )
    parser.add_argument("--image", required=True, help="Path to input image.")
    parser.add_argument("--output", default="yolo26_result.jpg", help="Path to save annotated image.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold. Default: 0.25.")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU NMS threshold. Default: 0.45.")
    return parser.parse_args()


def resolve_model_path(model_arg: str) -> tuple[str, Path]:
    model_path = Path(model_arg)
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    if model_path.is_file():
        suffix = model_path.suffix.lower()
        if suffix == ".xml":
            return "openvino", model_path
        if suffix == ".pt":
            return "pytorch", model_path
        raise ValueError("Model file must be .pt or OpenVINO .xml.")
    if model_path.is_dir():
        raise ValueError(f"Model path is a directory. Pass an explicit .pt or .xml file path: {model_path}")
    raise ValueError("Model must be a .pt file or OpenVINO .xml file.")


def validate_openvino_model(model_xml: Path) -> Path:
    if "_openvino_model" not in model_xml.parent.name:
        raise ValueError(
            f"OpenVINO model XML must be inside a '*_openvino_model' directory: {model_xml.parent}"
        )
    model_bin = model_xml.with_suffix(".bin")
    if not model_bin.exists():
        raise FileNotFoundError(f"OpenVINO weights file is missing: {model_bin}")
    metadata = model_xml.parent / "metadata.yaml"
    if not metadata.exists():
        raise FileNotFoundError(f"OpenVINO metadata file is missing: {metadata}")
    return model_xml.parent


def main() -> None:
    args = parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image path does not exist: {image_path}")

    backend, model_path = resolve_model_path(args.model)
    if backend == "openvino":
        model_for_yolo = validate_openvino_model(model_path)
        predict_device = "intel:cpu"
    else:
        model_for_yolo = model_path
        predict_device = "cpu"

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    detector = YOLO(model_for_yolo, task="detect")
    results = detector(str(image_path), device=predict_device, conf=args.conf, iou=args.iou)

    annotated = results[0].plot()  # BGR image with boxes and labels
    if not cv2.imwrite(str(output_path), annotated):
        raise RuntimeError(f"Failed to write output image to {output_path}")

    names = results[0].names
    boxes = results[0].boxes
    print(f"Inference backend: {backend}")
    print(f"Inference device: {predict_device}")
    print(f"Saved result: {output_path}")
    print(f"Detections: {len(boxes)}")
    for idx, box in enumerate(boxes, start=1):
        cls_id = int(box.cls.item())
        conf = float(box.conf.item())
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
        if isinstance(names, dict):
            label = names.get(cls_id, str(cls_id))
        elif isinstance(names, list) and cls_id < len(names):
            label = names[cls_id]
        else:
            label = str(cls_id)
        print(f"{idx:02d}. {label:20s} conf={conf:.3f} box=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")


if __name__ == "__main__":
    main()
