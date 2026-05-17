"""Allowed labels and stable id mappings for ticket classification."""

from __future__ import annotations

LABELS: tuple[str, ...] = (
    "policy_question",
    "login_issue",
    "incident_report",
    "training_request",
    "data_privacy",
    "document_access",
    "procurement_request",
    "compliance_audit",
    "identity_management",
    "infrastructure_fault",
    "grievance_hr",
    "change_management",
)

LABEL_TO_ID: dict[str, int] = {label: index for index, label in enumerate(LABELS)}
ID_TO_LABEL: dict[int, str] = {index: label for label, index in LABEL_TO_ID.items()}


def validate_label(label: str) -> str:
    """Return a valid label or raise a clear ValueError."""
    if label not in LABEL_TO_ID:
        allowed = ", ".join(LABELS)
        raise ValueError(f"Invalid label {label!r}. Allowed labels: {allowed}")
    return label
