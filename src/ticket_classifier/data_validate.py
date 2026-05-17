"""Validate synthetic JSONL datasets and reject obvious sensitive content."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from ticket_classifier.labels import validate_label


SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("api_key", re.compile(r"(?i)\b(api[_-]?key|secret[_-]?key)\b\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{16,}")),
    ("bearer_token", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9_\-\.=]{20,}")),
    ("password", re.compile(r"(?i)\b(password|passwd|pwd)\b\s*[:=]\s*['\"]?[^\\s'\"]{8,}")),
    ("private_key", re.compile(r"-----BEGIN (RSA |DSA |EC |OPENSSH )?PRIVATE KEY-----")),
    ("connection_string", re.compile(r"(?i)\b(server|host|database|uid|user id|password)\s*=[^;]+;.*\b(database|uid|pwd|password)\s*=")),
    ("internal_url", re.compile(r"(?i)\bhttps?://(?:localhost|127\.0\.0\.1|10\.\d{1,3}\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2\d|3[0-1])\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3}|[\w.-]*\.(?:local|internal|corp|intranet))(?:[/:?][^\s]*)?")),
    ("ip_address", re.compile(r"\b(?:(?:10|127)\.\d{1,3}\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2\d|3[0-1])\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3})\b")),
)


@dataclass(frozen=True)
class ValidationErrorDetail:
    """One validation issue with line-level context when available."""

    message: str
    line_number: int | None = None

    def format(self) -> str:
        prefix = f"line {self.line_number}: " if self.line_number is not None else ""
        return f"{prefix}{self.message}"


def detect_obvious_secrets(text: str) -> list[str]:
    """Return names of obvious secret/sensitive patterns found in text."""
    return [name for name, pattern in SECRET_PATTERNS if pattern.search(text)]


def validate_record(record: dict[str, Any], line_number: int) -> list[ValidationErrorDetail]:
    """Validate a single JSON object from the dataset."""
    errors: list[ValidationErrorDetail] = []
    for field in ("text", "label"):
        if field not in record:
            errors.append(ValidationErrorDetail(f"missing required field {field!r}", line_number))

    text = record.get("text")
    if not isinstance(text, str) or not text.strip():
        errors.append(ValidationErrorDetail("'text' must be a non-empty string", line_number))
    elif matches := detect_obvious_secrets(text):
        errors.append(ValidationErrorDetail(f"text contains prohibited pattern(s): {', '.join(matches)}", line_number))

    label = record.get("label")
    if not isinstance(label, str):
        errors.append(ValidationErrorDetail("'label' must be a string", line_number))
    else:
        try:
            validate_label(label)
        except ValueError as exc:
            errors.append(ValidationErrorDetail(str(exc), line_number))

    return errors


def iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any] | None, str | None]]:
    """Yield line number, parsed JSON object, and parse error text for a JSONL file."""
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                yield line_number, None, "empty line is not valid JSONL"
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as exc:
                yield line_number, None, f"invalid JSON: {exc.msg}"
                continue
            if not isinstance(value, dict):
                yield line_number, None, "line must contain a JSON object"
                continue
            yield line_number, value, None


def validate_file(path: str | Path) -> list[ValidationErrorDetail]:
    """Validate a JSONL dataset and return all errors."""
    dataset_path = Path(path)
    if not dataset_path.exists():
        return [ValidationErrorDetail(f"file does not exist: {dataset_path}")]
    if not dataset_path.is_file():
        return [ValidationErrorDetail(f"path is not a file: {dataset_path}")]

    errors: list[ValidationErrorDetail] = []
    for line_number, record, parse_error in iter_jsonl(dataset_path):
        if parse_error is not None:
            errors.append(ValidationErrorDetail(parse_error, line_number))
            continue
        assert record is not None
        errors.extend(validate_record(record, line_number))
    return errors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a synthetic ticket classifier JSONL dataset.")
    parser.add_argument("path", help="Path to a JSONL file, such as data/train.jsonl")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    errors = validate_file(args.path)
    if errors:
        print(f"Validation failed for {args.path}:")
        for error in errors:
            print(f"- {error.format()}")
        return 1
    print(f"Validation passed for {args.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
