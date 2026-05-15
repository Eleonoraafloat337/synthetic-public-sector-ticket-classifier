"""Generate a Hugging Face-compatible model card."""

from __future__ import annotations

from pathlib import Path

from ticket_classifier.labels import LABELS


def generate_model_card(repo_id: str, metrics_summary: str | None = None) -> str:
    """Return Markdown for the model card README."""
    metrics = metrics_summary or "Evaluation results should be added after running the held-out synthetic test set."
    labels = ", ".join(f"`{label}`" for label in LABELS)
    return f"""---
license: mit
language:
- en
library_name: transformers
tags:
- text-classification
- synthetic-data
- public-sector
- helpdesk
- educational
---

# Synthetic Public Sector Ticket Classifier

This model is a small educational text classifier for synthetic public-sector IT and helpdesk-style tickets. It predicts one of the following labels: {labels}.

## Intended Use

- Demonstrating supervised fine-tuning with Hugging Face Transformers.
- Practicing CLI-first model training, evaluation, and prediction workflows.
- Teaching safe handling of synthetic examples before adapting a workflow to governed data.

## Out-of-Scope Use

- Do not use this model for production routing, legal decisions, benefits decisions, security incident response, or privacy determinations without governance review.
- Do not assume the model has seen any real agency, client, resident, or employee data.
- Do not use outputs as authoritative advice.

## Training Data

The training, validation, and test files contain only synthetic examples created for this repository. They intentionally avoid real client data, case data, secrets, internal URLs, credentials, access tokens, logs, IP addresses, and regulated information.

## Evaluation

{metrics}

## Limitations

- The dataset is intentionally small and synthetic.
- Label boundaries are simplified for education.
- The model may be overconfident on unfamiliar wording.
- Performance on real tickets is unknown and should not be inferred from synthetic results.

## Safety Notes

Public publishing requires a fresh review of data, license, model behavior, generated model card, and repository visibility. Publishing from this project requires an explicit `--confirm-publish` CLI flag.

## License

MIT

## Example Usage

```python
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="{repo_id}"
)

print(classifier("I cannot access the policy document library."))
```
"""


def write_model_card(output_path: str | Path, repo_id: str, metrics_summary: str | None = None) -> Path:
    """Write a model card to output_path and return the resulting Path."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(generate_model_card(repo_id=repo_id, metrics_summary=metrics_summary), encoding="utf-8")
    return path
