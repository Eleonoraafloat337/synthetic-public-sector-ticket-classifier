# synthetic-public-sector-ticket-classifier

Safe, synthetic public-sector helpdesk ticket classification with Hugging Face Transformers.

All examples in this repository are synthetic. Do not add real client data, employee data, credentials, internal URLs, access tokens, logs, IP addresses, or regulated information.

## Model Access

The Hugging Face model repository is:

[nprasann/synthetic-public-sector-ticket-classifier](https://huggingface.co/nprasann/synthetic-public-sector-ticket-classifier)

## Labels

- `policy_question`
- `login_issue`
- `incident_report`
- `training_request`
- `data_privacy`
- `document_access`

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

Train the compact repository model into `./model`:

```bash
python -m model.train --output-dir model
```

Run local inference:

```bash
python -m model.inference \
  --model-dir model \
  --text "I cannot access the policy document library."
```

Run tests:

```bash
pytest
```

## Load From Hugging Face

```python
from transformers import AutoModel, AutoTokenizer

hf_model = "nprasann/synthetic-public-sector-ticket-classifier"
model = AutoModel.from_pretrained(hf_model)
tokenizer = AutoTokenizer.from_pretrained(hf_model)
```

For classification:

```python
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="nprasann/synthetic-public-sector-ticket-classifier",
    tokenizer="nprasann/synthetic-public-sector-ticket-classifier",
)

print(classifier("I cannot access the policy document library."))
```

## Repository Layout

- `data/`: synthetic JSONL train, validation, and test splits.
- `model/`: compact Hugging Face model artifacts plus training and inference entry points.
- `src/ticket_classifier/`: original CLI package for data validation, training, evaluation, prediction, publishing, and model-card generation.
- `tests/`: pytest coverage for validation, labels, and model-card generation.
- `docs/`: educational notes and safety guidance.

## Publishing

Publishing is deliberately opt-in. The publishing command refuses to upload unless `--confirm-publish` is provided.

```bash
python -m ticket_classifier.publish_to_hub \
  --model-dir model \
  --repo-id nprasann/synthetic-public-sector-ticket-classifier \
  --confirm-publish
```

## License

MIT
