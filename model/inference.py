"""Command-line inference for the saved Hugging Face model artifacts."""

from __future__ import annotations

import argparse
import json

from model.model import TicketClassifier


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Classify one synthetic public-sector ticket.")
    parser.add_argument("--model-dir", default="model")
    parser.add_argument("--text", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    classifier = TicketClassifier(args.model_dir)
    result = classifier.predict(args.text)
    print(json.dumps({"label": result.label, "confidence": result.confidence, "scores": result.scores}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
