# Public Sector Ticket Classifier

Production-ready FastAPI and training scaffold for routing synthetic government helpdesk tickets across 12 labels. The model path, thresholds, and secrets are environment driven so the same image can run locally or in Azure Container Apps.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python tools/generate_dataset.py
PYTHONPATH=src python -m ticket_classifier.data_validate data/train.jsonl
PYTHONPATH=src uvicorn inference.app:app --app-dir src --host 0.0.0.0 --port 8080
```

Train locally when GPU resources are available:

```bash
PYTHONPATH=src python -m ticket_classifier.train \
  --train data/train.jsonl \
  --validation data/validation.jsonl \
  --test data/test.jsonl \
  --output-dir output/model
```

## Environment Variables

| Variable | Purpose | Default |
|---|---|---|
| `MODEL_DIR` | Directory containing model artifacts and `label_map.json` | `output/model` |
| `MODEL_VERSION` | Version returned by inference responses | `1.2.0` |
| `BASE_MODEL` | Hugging Face base model for training | `microsoft/deberta-v3-small` |
| `CONFIDENCE_THRESHOLD` | Below this score, return `uncertain` | `0.5` |
| `LOW_CONFIDENCE_THRESHOLD` | Metrics threshold for low-confidence rate | `0.6` |
| `DRIFT_WINDOW` | Number of recent predictions used for drift checks | `1000` |
| `DRIFT_THRESHOLD` | Jensen-Shannon warning threshold | `0.1` |
| `HUGGINGFACE_TOKEN` | Optional Hub token, sourced from Key Vault in Azure | unset |
| `MLFLOW_TRACKING_URI` | MLflow tracking backend | local or workflow-provided |

## Architecture

```text
Azure Container Apps <-> FastAPI inference service
        |
Azure Machine Learning <-> MLflow training and registry
        |
Azure Container Registry <-> Docker image
        |
Azure Monitor / App Insights <-> JSON logs and Prometheus metrics
        |
Azure Key Vault <-> HF token and runtime secrets
```

## Production Checklist

- [ ] Data validation passes for all JSONL splits.
- [ ] `python -m ticket_classifier.train` produces `output/model/label_map.json`.
- [ ] Macro F1 is at least 0.80 and no label F1 is below 0.65.
- [ ] `python -m ticket_classifier.run_exam --model-dir output/model --output output/eval/scorecard.json` completes.
- [ ] `ruff`, `mypy --strict`, and `pytest --cov=src --cov-fail-under=85` pass.
- [ ] Docker image runs as non-root and responds at `/health` within 30 seconds.
- [ ] Container App references Key Vault for secrets.
- [ ] No raw ticket text appears in application logs.
- [ ] Blue/green deployment health check passes before traffic promotion.

## Publishing

Hugging Face publication is guarded. The command refuses to publish unless `--confirm-publish` is present:

```bash
PYTHONPATH=src python -m ticket_classifier.publish_to_hub \
  --model-dir output/model \
  --repo-id nprasann/synthetic-public-sector-ticket-classifier \
  --confirm-publish
```
