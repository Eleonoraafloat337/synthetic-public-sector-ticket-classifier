# Hugging Face Publishing Guide

Hugging Face Hub hosts models, tokenizers, datasets, and model cards. This project can publish a trained model, but only when the explicit confirmation flag is provided.

Publishing command:

```bash
python -m ticket_classifier.publish_to_hub \
  --model-dir output/model \
  --repo-id YOUR_USERNAME/synthetic-public-sector-ticket-classifier \
  --private \
  --confirm-publish
```

The command requires `HUGGINGFACE_TOKEN` or an existing `huggingface-cli login`.

Before public publishing:

- Confirm the dataset is synthetic.
- Confirm no secrets, internal URLs, logs, IP addresses, or regulated information are present.
- Review the generated model card.
- Review the license.
- Test model behavior on representative safe examples.
- Prefer private publishing until review is complete.
