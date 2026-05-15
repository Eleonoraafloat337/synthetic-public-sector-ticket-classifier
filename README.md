# synthetic-public-sector-ticket-classifier

> A safe, synthetic, public-sector IT/helpdesk text-classification project using Hugging Face Transformers.

![Python](https://img.shields.io/badge/python-3.11%2B-3776AB)
![Transformers](https://img.shields.io/badge/Hugging%20Face-Transformers-FFD21E)
![CLI first](https://img.shields.io/badge/workflow-CLI%20first-2E7D32)
![Synthetic data only](https://img.shields.io/badge/data-synthetic%20only-6A1B9A)
![Tests](https://img.shields.io/badge/tests-pytest-0A7EA4)

`synthetic-public-sector-ticket-classifier` is an educational, CLI-first machine learning repository that trains a small sequence classifier for synthetic public-sector IT/helpdesk-style tickets.

All examples are synthetic. This repository must not contain real client data, case data, credentials, internal URLs, secrets, access tokens, logs, IP addresses, or regulated information.

## Repository Metadata

Suggested GitHub description:

> Synthetic public-sector helpdesk ticket classifier using Hugging Face Transformers and safe JSONL training data.

Suggested GitHub topics:

`huggingface`, `transformers`, `text-classification`, `synthetic-data`, `public-sector`, `helpdesk`, `machine-learning`, `python`, `pytorch`, `datasets`, `pytest`, `model-card`, `responsible-ai`

## Software Versions

This repo is designed for Python 3.11+ and pins direct runtime/test dependencies for repeatable local demos.

| Component | Version |
| --- | --- |
| Python | `>=3.11` |
| `torch` | `2.7.0` |
| `transformers` | `4.51.3` |
| `accelerate` | `1.6.0` |
| `datasets` | `3.6.0` |
| `evaluate` | `0.4.3` |
| `scikit-learn` | `1.6.1` |
| `numpy` | `2.2.5` |
| `huggingface-hub` | `0.31.1` |
| `pytest` | `9.0.3` |

## Concepts

A model is a learned function that maps inputs to outputs. In this project, the input is ticket text and the output is one of six labels: `policy_question`, `login_issue`, `incident_report`, `training_request`, `data_privacy`, or `document_access`.

Training from scratch means learning a model from the beginning, usually with very large datasets and significant compute. Fine-tuning starts with an existing model and updates it on a smaller task-specific dataset. This project fine-tunes `distilbert-base-uncased` for educational text classification.

Hugging Face Hub is a model-sharing platform. It can host trained models, tokenizers, model cards, and datasets. Publishing from this repo is deliberately opt-in and requires `--confirm-publish`.

## Quickstart

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Validate the synthetic data:

```bash
python -m ticket_classifier.data_validate data/train.jsonl
python -m ticket_classifier.data_validate data/validation.jsonl
python -m ticket_classifier.data_validate data/test.jsonl
```

Train locally:

```bash
python -m ticket_classifier.train \
  --train data/train.jsonl \
  --validation data/validation.jsonl \
  --output-dir output/model
```

Evaluate:

```bash
python -m ticket_classifier.evaluate_model \
  --model-dir output/model \
  --test data/test.jsonl
```

Predict:

```bash
python -m ticket_classifier.predict \
  --model-dir output/model \
  --text "User cannot access the policy document library."
```

Run tests:

```bash
pytest
```

## Publishing Safely

Publishing is never the default. The command refuses to publish unless `--confirm-publish` is present.

```bash
python -m ticket_classifier.publish_to_hub \
  --model-dir output/model \
  --repo-id YOUR_USERNAME/synthetic-public-sector-ticket-classifier \
  --private \
  --confirm-publish
```

Before public publishing, review the data, license, generated model card, model behavior, and repository visibility. Authentication requires either `HUGGINGFACE_TOKEN` or an existing `huggingface-cli login`.

## Download And Use A Published Model

```python
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="YOUR_USERNAME/synthetic-public-sector-ticket-classifier"
)

print(classifier("I cannot access the policy document library."))
```

## Repository Contents

- `data/*.jsonl`: synthetic train, validation, and test splits.
- `src/ticket_classifier/`: CLI modules for validation, training, evaluation, prediction, publishing, and model-card generation.
- `docs/`: educational notes and safety guidance.
- `tests/`: pytest coverage for validation, labels, and model-card generation.

## Data Safety Checklist

- Use only synthetic examples.
- Do not include real client, resident, employee, agency, case, ticket, or incident data.
- Do not include credentials, tokens, passwords, private keys, connection strings, internal URLs, logs, or IP addresses.
- Run `python -m ticket_classifier.data_validate data/train.jsonl` before training.
- Re-review data and model behavior before any public publishing.

## License

MIT
