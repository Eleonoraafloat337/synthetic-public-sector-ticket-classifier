import json
from pathlib import Path

from ticket_classifier.data_validate import detect_obvious_secrets, validate_file, validate_record
from ticket_classifier.labels import LABELS


def test_valid_dataset_row_passes() -> None:
    row = {"text": "Please assign the annual records training to the finance team.", "label": "training_request"}
    assert validate_record(row, line_number=1) == []


def test_invalid_label_rejection(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"text": "Please help me log in.", "label": "unknown"}) + "\n", encoding="utf-8")
    errors = validate_file(path)
    assert errors
    assert "Invalid label" in errors[0].message


def test_secret_detection() -> None:
    text = "Authorization: Bearer abcdefghijklmnopqrstuvwxyz123456"
    assert "bearer_token" in detect_obvious_secrets(text)


def test_internal_url_detection() -> None:
    text = "The sample URL http://portal.internal/request should never appear in data."
    assert "internal_url" in detect_obvious_secrets(text)


def test_all_splits_have_all_twelve_labels() -> None:
    for split in ("train", "validation", "test"):
        labels = {
            json.loads(line)["label"]
            for line in Path(f"data/{split}.jsonl").read_text(encoding="utf-8").splitlines()
        }
        assert labels == set(LABELS)


def test_pii_patterns_are_rejected() -> None:
    assert "email" in detect_obvious_secrets("Please contact person@example.test")
    assert "phone" in detect_obvious_secrets("Call 555-010-1234 for the case")
    assert "ssn" in detect_obvious_secrets("The sample identifier is 123-45-6789")
