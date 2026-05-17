---
license: mit
language:
- en
library_name: transformers
pipeline_tag: text-classification
tags:
- text-classification
- synthetic-data
- public-sector
- helpdesk
- educational
datasets:
- synthetic
model-index:
- name: synthetic-public-sector-ticket-classifier
  results:
  - task:
      type: text-classification
      name: Text Classification
    dataset:
      type: synthetic
      name: Repository synthetic validation split
    metrics:
    - type: accuracy
      value: 0.6111111111111112
      name: Accuracy
    - type: f1
      value: 0.5962962962962962
      name: Macro F1
---

# Synthetic Public Sector Ticket Classifier

This is a compact Hugging Face Transformers text-classification model for synthetic public-sector helpdesk tickets. It predicts one of six labels: `policy_question`, `login_issue`, `incident_report`, `training_request`, `data_privacy`, or `document_access`.

GitHub repository: https://github.com/nprasann/synthetic-public-sector-ticket-classifier

## Intended Use

- Educational demonstrations of safe text-classification workflows.
- Local experiments with synthetic helpdesk routing examples.
- Testing Hugging Face model save/load and publishing flows.

## Out-of-Scope Use

Do not use this model for production routing, legal decisions, benefits decisions, HR decisions, privacy determinations, security incident response, or authoritative public-sector advice.

## Dataset Description

The repository contains synthetic JSONL data only:

- `data/train.jsonl`: 60 examples.
- `data/validation.jsonl`: 18 examples.
- `data/test.jsonl`: 18 examples.

The data is written to avoid real client data, employee data, agency case details, credentials, internal URLs, logs, IP addresses, and regulated information.

## Training Procedure

The model artifacts in this repository were produced with `python -m model.train --epochs 200 --batch-size 8 --learning-rate 0.001 --output-dir model`.

The training script builds a small BERT-style sequence classifier from configuration and trains a word-level tokenizer from the synthetic repository data. No external pretrained checkpoint is downloaded by `model.train`.

## Evaluation Metrics

Validation split metrics from the packaged training run:

| Metric | Value |
| --- | ---: |
| Accuracy | 0.6111111111111112 |
| Macro F1 | 0.5962962962962962 |

These metrics are from a very small synthetic validation split and should not be interpreted as real-world performance.

## Limitations and Ethical Considerations

- The dataset is small and synthetic.
- The model may be brittle on wording outside the examples.
- The model may be overconfident.
- It has not been evaluated for demographic fairness.
- Human review is required for any consequential workflow.

## Example Inference

```python
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="nprasann/synthetic-public-sector-ticket-classifier",
    tokenizer="nprasann/synthetic-public-sector-ticket-classifier",
)

print(classifier("I cannot access the policy document library."))
```
