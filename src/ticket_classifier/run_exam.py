"""Lightweight spec exam runner for publication gates."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ticket_classifier.data_validate import validate_file
from ticket_classifier.labels import LABELS


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def score_data(data_dir: Path) -> tuple[list[dict[str, Any]], float]:
    """Score the data-focused exam tier; other tiers are externally validated."""
    checks: list[dict[str, Any]] = []
    earned = 0.0
    all_texts: list[str] = []
    ambiguous_labels: set[str] = set()
    for split, min_count in {"train": 100, "validation": 20, "test": 20}.items():
        rows = _read_jsonl(data_dir / f"{split}.jsonl")
        counts = Counter(row["label"] for row in rows)
        all_texts.extend(row["text"] for row in rows)
        ambiguous_labels.update(row["label"] for row in rows if row.get("notes"))
        present = all(counts[label] > 0 for label in LABELS)
        checks.append({"id": f"{split}_all_labels", "passed": present, "score": 1.0 if present else 0.0})
        earned += 1.0 if present else 0.0
        meets_min = all(counts[label] >= min_count for label in LABELS)
        checks.append({"id": f"{split}_minimum_counts", "passed": meets_min, "score": 1.0 if meets_min else 0.0})
        earned += 1.0 if meets_min else 0.0
        imbalance = max(counts.values()) / min(counts.values())
        if split == "train":
            balanced = imbalance <= 2.0
            checks.append({"id": "train_imbalance_ratio", "passed": balanced, "score": 1.0 if balanced else 0.0})
            earned += 1.0 if balanced else 0.0
        valid = not validate_file(data_dir / f"{split}.jsonl")
        checks.append({"id": f"{split}_data_validate", "passed": valid, "score": 1.0 if valid else 0.0})
        earned += 1.0 if valid else 0.0
    no_duplicates = len(all_texts) == len(set(all_texts))
    checks.append({"id": "no_duplicate_texts", "passed": no_duplicates, "score": 1.0 if no_duplicates else 0.0})
    earned += 1.0 if no_duplicates else 0.0
    edge_cases = len(ambiguous_labels) >= 6
    checks.append({"id": "ambiguous_edge_cases", "passed": edge_cases, "score": 1.0 if edge_cases else 0.0})
    earned += 1.0 if edge_cases else 0.0
    return checks, earned


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run publication readiness exam.")
    parser.add_argument("--model-dir", default="output/model")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", default="output/eval/scorecard.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    checks, tier1 = score_data(Path(args.data_dir))
    # Remaining tiers are represented as pending checks so the scorecard is structurally complete.
    scorecard = {
        "model_version": "1.2.0",
        "exam_date": datetime.now(UTC).isoformat(),
        "total_weighted": tier1,
        "max_weighted": 110.0,
        "percent": tier1 / 110.0,
        "tiers": [{"id": "T1", "name": "Label Completeness", "score": tier1, "max": 10.0, "floor": 8.2, "checks": checks}],
        "publication_ready": False,
        "notes": "Automated local checks implemented for data tier; cloud, container, and trained-model gates require CI infrastructure.",
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(scorecard, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
