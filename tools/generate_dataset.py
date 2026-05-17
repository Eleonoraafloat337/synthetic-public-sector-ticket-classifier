"""Generate synthetic public-sector ticket data with no real PII."""

from __future__ import annotations

import json
import random
from pathlib import Path

LABEL_TOPICS = {
    "policy_question": ["records retention", "remote work", "travel approval", "procurement policy", "leave rules"],
    "login_issue": ["password reset", "locked account", "single sign-on", "portal login", "authentication error"],
    "incident_report": ["minor safety incident", "lost access card", "near miss", "equipment damage", "visitor concern"],
    "training_request": ["privacy refresher", "records course", "induction", "security awareness", "manager workshop"],
    "data_privacy": ["personal information", "consent question", "data deletion", "breach concern", "subject access"],
    "document_access": ["case folder", "briefing paper", "policy library", "archived memo", "shared register"],
    "procurement_request": ["software license", "laptop purchase", "vendor onboarding", "hardware quote", "subscription renewal"],
    "compliance_audit": ["audit evidence", "regulatory report", "control attestation", "Essential 8", "FISMA checklist"],
    "identity_management": ["MFA setup", "service account", "role provisioning", "offboarding", "access group"],
    "infrastructure_fault": ["VPN outage", "network latency", "server degradation", "cloud resource", "printing queue"],
    "grievance_hr": ["workplace complaint", "misconduct concern", "HR dispute", "manager conduct", "bullying report"],
    "change_management": ["CAB submission", "release approval", "rollback plan", "maintenance window", "change freeze"],
}

TEMPLATES = [
    "Please help with {topic}; our team cannot complete the routine public service task today.",
    "I need guidance about {topic}. The request is not urgent but it is blocking my next step.",
    "Could someone review this {topic} ticket and advise the correct path?",
    "We have a question on {topic} after a staff member followed the usual process and got stuck.",
    "Kindly assist, I am new in the office and {topic} is confusing for me.",
    "For tomorrow's work queue, can the service desk check {topic} and tell us what is needed?",
    "The branch has raised {topic}; please route this to the responsible support group.",
    "I may not explain perfect English, but need help for {topic} so our form can move forward.",
    "There is an issue involving {topic}. It affects a low-risk administrative workflow.",
    "Can this ticket be assigned for {topic}? We only need standard support, not an exception.",
]

AMBIGUOUS = {
    "policy_question": "Mentions document access, but primary intent is asking which policy applies.",
    "login_issue": "Mentions MFA and identity, but the user is blocked at login.",
    "data_privacy": "Mentions an incident, but primary concern is privacy handling.",
    "procurement_request": "Mentions policy, but primary intent is purchasing approval.",
    "identity_management": "Mentions login, but primary intent is role provisioning.",
    "change_management": "Mentions infrastructure, but primary intent is CAB approval.",
}


def build_row(label: str, index: int, split: str) -> dict[str, str]:
    topic = LABEL_TOPICS[label][index % len(LABEL_TOPICS[label])]
    template = TEMPLATES[(index + len(split)) % len(TEMPLATES)]
    text = f"{template.format(topic=topic)} Ref {split}-{label.replace('_', '-')}-{index:03d}."
    row = {"text": text, "label": label}
    if label in AMBIGUOUS and index < 8:
        row["notes"] = AMBIGUOUS[label]
        row["text"] = f"{text} This may sound related to another queue, but the main need is {topic}."
    return row


def write_split(path: Path, split: str, per_label: int) -> None:
    rows = [build_row(label, index, split) for label in LABEL_TOPICS for index in range(per_label)]
    random.Random(42 + len(split)).shuffle(rows)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def main() -> int:
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    write_split(data_dir / "train.jsonl", "train", 300)
    write_split(data_dir / "validation.jsonl", "validation", 25)
    write_split(data_dir / "test.jsonl", "test", 25)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
