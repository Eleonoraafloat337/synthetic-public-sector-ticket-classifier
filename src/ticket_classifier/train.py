"""Train and evaluate a governed Hugging Face sequence classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from transformers.trainer_utils import EvalPrediction

from ticket_classifier.config import DEFAULT_BASE_MODEL, DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS, DEFAULT_MAX_LENGTH, DEFAULT_SEED
from ticket_classifier.data_validate import validate_file
from ticket_classifier.labels import ID_TO_LABEL, LABEL_TO_ID, LABELS


def compute_metrics(eval_prediction: EvalPrediction | tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    """Return macro and per-label metrics for Trainer evaluation."""
    logits, labels = (
        (eval_prediction.predictions, eval_prediction.label_ids)
        if isinstance(eval_prediction, EvalPrediction)
        else eval_prediction
    )
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="macro", zero_division=0)
    _, _, per_label_f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        labels=list(ID_TO_LABEL),
        zero_division=0,
    )
    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
    }
    for index, label in ID_TO_LABEL.items():
        metrics[f"f1_{label}"] = float(per_label_f1[index])
    return metrics


def fail_if_invalid(paths: list[str]) -> None:
    """Validate all input JSONL files before training starts."""
    for path in paths:
        errors = validate_file(path)
        if errors:
            details = "\n".join(f"- {error.format()}" for error in errors[:20])
            raise ValueError(f"Dataset validation failed for {path}:\n{details}")


def write_label_map(output_dir: Path) -> None:
    """Persist stable label mappings for inference."""
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"labels": list(LABELS), "label_to_id": LABEL_TO_ID, "id_to_label": {str(k): v for k, v in ID_TO_LABEL.items()}}
    (output_dir / "label_map.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def export_evaluation_artifacts(
    logits: np.ndarray,
    labels: np.ndarray,
    output_dir: Path,
) -> dict[str, Any]:
    """Write confusion matrix, classification report, and calibration chart."""
    import matplotlib.pyplot as plt

    eval_dir = output_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    predictions = np.argmax(logits, axis=-1)
    probabilities = _softmax(logits)
    confidences = probabilities.max(axis=1)
    correctness = (predictions == labels).astype(float)

    report = classification_report(
        labels,
        predictions,
        labels=list(ID_TO_LABEL),
        target_names=list(LABELS),
        zero_division=0,
        output_dict=True,
    )
    (eval_dir / "classification_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    matrix = confusion_matrix(labels, predictions, labels=list(ID_TO_LABEL))
    fig, ax = plt.subplots(figsize=(12, 10))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(LABELS)), LABELS, rotation=45, ha="right")
    ax.set_yticks(range(len(LABELS)), LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(eval_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    bins = np.linspace(0.0, 1.0, 11)
    bin_ids = np.digitize(confidences, bins, right=True)
    accuracies: list[float] = []
    mean_confidences: list[float] = []
    ece = 0.0
    for bin_id in range(1, len(bins)):
        mask = bin_ids == bin_id
        if not np.any(mask):
            continue
        bin_accuracy = float(correctness[mask].mean())
        bin_confidence = float(confidences[mask].mean())
        accuracies.append(bin_accuracy)
        mean_confidences.append(bin_confidence)
        ece += float(mask.mean()) * abs(bin_accuracy - bin_confidence)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    ax.plot(mean_confidences, accuracies, marker="o", label=f"Model (ECE={ece:.3f})")
    ax.set_xlabel("Mean confidence")
    ax.set_ylabel("Accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(eval_dir / "calibration.png", dpi=160)
    plt.close(fig)
    return {"classification_report": report, "ece": ece}


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a synthetic public-sector ticket classifier.")
    parser.add_argument("--train", default="data/train.jsonl", help="Training JSONL path.")
    parser.add_argument("--validation", default="data/validation.jsonl", help="Validation JSONL path.")
    parser.add_argument("--test", default="data/test.jsonl", help="Held-out test JSONL path.")
    parser.add_argument("--output-dir", default="output/model", help="Directory for the saved model and tokenizer.")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL, help="Hugging Face base model name.")
    parser.add_argument("--epochs", type=float, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--mlflow-experiment", default="public-sector-ticket-classifier")
    parser.add_argument("--mlflow-run-name", default="deberta-v3-small-ticket-classifier")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    fail_if_invalid([args.train, args.validation, args.test])

    from datasets import load_dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(args.seed)
    dataset = load_dataset("json", data_files={"train": args.train, "validation": args.validation, "test": args.test})
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    def tokenize(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    encoded = dataset.map(tokenize, batched=True)
    encoded = encoded.map(lambda row: {"labels": LABEL_TO_ID[row["label"]]})
    encoded = encoded.remove_columns([column for column in ("text", "label", "notes") if column in encoded["train"].column_names])

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
        greater_is_better=True,
        label_smoothing_factor=0.1,
        seed=args.seed,
        report_to=["mlflow"],
        logging_steps=10,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded["train"],
        eval_dataset=encoded["validation"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    import mlflow

    mlflow.set_experiment(args.mlflow_experiment)
    with mlflow.start_run(run_name=args.mlflow_run_name):
        mlflow.log_params(
            {
                "base_model": args.base_model,
                "label_smoothing_factor": 0.1,
                "metric_for_best_model": "f1_macro",
                "labels": ",".join(LABELS),
            },
        )
        trainer.train()
        test_output = trainer.predict(encoded["test"])
        test_metrics = {f"test/{key}": float(value) for key, value in compute_metrics((test_output.predictions, test_output.label_ids)).items()}
        mlflow.log_metrics(test_metrics)
        artifacts = export_evaluation_artifacts(test_output.predictions, test_output.label_ids, output_dir.parent)
        mlflow.log_metric("test/ece", float(artifacts["ece"]))
        mlflow.log_artifacts(str(output_dir.parent / "eval"), artifact_path="eval")

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    write_label_map(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
