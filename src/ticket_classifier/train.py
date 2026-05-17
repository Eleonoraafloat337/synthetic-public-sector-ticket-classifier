"""Train a small Hugging Face sequence classifier on synthetic JSONL data."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from ticket_classifier.config import DEFAULT_BASE_MODEL, DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS, DEFAULT_MAX_LENGTH, DEFAULT_SEED
from ticket_classifier.data_validate import validate_file
from ticket_classifier.labels import ID_TO_LABEL, LABEL_TO_ID


def compute_metrics(eval_prediction: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    logits, labels = eval_prediction
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="macro", zero_division=0)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
    }


def fail_if_invalid(paths: list[str]) -> None:
    for path in paths:
        errors = validate_file(path)
        if errors:
            details = "\n".join(f"- {error.format()}" for error in errors[:20])
            raise ValueError(f"Dataset validation failed for {path}:\n{details}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a synthetic public-sector ticket classifier.")
    parser.add_argument("--train", required=True, help="Training JSONL path.")
    parser.add_argument("--validation", required=True, help="Validation JSONL path.")
    parser.add_argument("--output-dir", required=True, help="Directory for the saved model and tokenizer.")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL, help="Hugging Face base model name.")
    parser.add_argument("--epochs", type=float, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    fail_if_invalid([args.train, args.validation])
    set_seed(args.seed)

    dataset = load_dataset("json", data_files={"train": args.train, "validation": args.validation})
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    def tokenize(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    encoded = dataset.map(tokenize, batched=True)
    encoded = encoded.map(lambda row: {"labels": LABEL_TO_ID[row["label"]]})
    encoded = encoded.remove_columns(["text", "label"])

    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=len(LABEL_TO_ID),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )

    output_dir = Path(args.output_dir)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        seed=args.seed,
        report_to=[],
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded["train"],
        eval_dataset=encoded["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"Saved model and tokenizer to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
