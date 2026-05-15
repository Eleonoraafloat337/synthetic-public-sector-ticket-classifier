"""Evaluate a saved classifier on a synthetic test JSONL file."""

from __future__ import annotations

import argparse

import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from ticket_classifier.data_validate import validate_file
from ticket_classifier.labels import LABELS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a saved ticket classifier.")
    parser.add_argument("--model-dir", required=True, help="Directory containing a saved model and tokenizer.")
    parser.add_argument("--test", required=True, help="Synthetic test JSONL path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    errors = validate_file(args.test)
    if errors:
        details = "\n".join(f"- {error.format()}" for error in errors[:20])
        raise ValueError(f"Dataset validation failed for {args.test}:\n{details}")

    dataset = load_dataset("json", data_files={"test": args.test})["test"]
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, top_k=None, truncation=True)

    true_labels = list(dataset["label"])
    predicted_labels: list[str] = []
    for row in dataset:
        scores = classifier(row["text"])[0]
        predicted_labels.append(max(scores, key=lambda item: item["score"])["label"])

    print(f"accuracy: {accuracy_score(true_labels, predicted_labels):.4f}")
    print("\nper-label precision/recall/F1:")
    print(classification_report(true_labels, predicted_labels, labels=list(LABELS), zero_division=0))
    print("confusion matrix:")
    matrix = confusion_matrix(true_labels, predicted_labels, labels=list(LABELS))
    print("labels:", ", ".join(LABELS))
    print(np.array2string(matrix))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
