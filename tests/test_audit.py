import json
from uuid import UUID

from inference.audit import build_audit_record


def test_audit_record_fields_and_no_raw_text() -> None:
    raw_text = "Please reset my password"
    record = build_audit_record(
        input_length=len(raw_text),
        label="login_issue",
        confidence=0.94,
        below_threshold=False,
        model_version="1.2.0",
        request_id="00000000-0000-4000-8000-000000000000",
    )
    payload = json.dumps(record)
    assert raw_text not in payload
    assert set(record) == {
        "event",
        "timestamp",
        "input_length",
        "label",
        "confidence",
        "below_threshold",
        "model_version",
        "request_id",
    }
    assert UUID(record["request_id"]).version == 4
