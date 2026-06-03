from __future__ import annotations

import argparse

from ultralytics import YOLO

from .app import load_config


def run_inference(config):
    model = YOLO(str(config.paths.model))
    predict_kwargs = {
        "source": str(config.paths.input_video),
        "save": config.yolo_inference.save,
        "exist_ok": config.yolo_inference.exist_ok,
    }
    if config.yolo_inference.project is not None:
        predict_kwargs["project"] = str(config.yolo_inference.project)
    if config.yolo_inference.name:
        predict_kwargs["name"] = config.yolo_inference.name

    results = model.predict(**predict_kwargs)
    print(results[0])

    print("--------------------------------")
    for box in results[0].boxes:
        print(box)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run simple YOLO inference using config.toml paths.")
    parser.add_argument("--config", default="config.toml", help="Path to the TOML configuration file.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    run_inference(config)


if __name__ == "__main__":
    main()
