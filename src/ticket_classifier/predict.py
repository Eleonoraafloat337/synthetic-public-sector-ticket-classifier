"""Run single-text prediction with a saved Hugging Face classifier."""

from __future__ import annotations

import argparse
from typing import Any

from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


def predict_text(model_dir: str, text: str, top_k: int = 3) -> dict[str, Any]:
    """Return the predicted label, confidence, and top labels."""
    if not text.strip():
        raise ValueError("--text must not be empty")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, top_k=top_k, truncation=True)
    results = classifier(text)[0]
    top_labels = [{"label": item["label"], "score": float(item["score"])} for item in results]
    winner = top_labels[0]
    return {"predicted_label": winner["label"], "confidence": winner["score"], "top_labels": top_labels}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict a synthetic ticket label.")
    parser.add_argument("--model-dir", required=True, help="Directory containing a saved model and tokenizer.")
    parser.add_argument("--text", required=True, help="Ticket text to classify.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = predict_text(args.model_dir, args.text, top_k=3)
    print(f"predicted_label: {result['predicted_label']}")
    print(f"confidence: {result['confidence']:.4f}")
    print("top_3_labels:")
    for item in result["top_labels"]:
        print(f"- {item['label']}: {item['score']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
