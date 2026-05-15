# Model Training vs Fine-Tuning

A machine learning model is a learned function. For this repository, it reads ticket text and predicts a label.

Training from scratch means learning model parameters from the beginning. That usually requires very large datasets, specialized infrastructure, extensive evaluation, and a governance program.

Fine-tuning starts from an existing base model and adapts it to a smaller task. This project fine-tunes `distilbert-base-uncased` for a six-label educational classifier. Fine-tuning is useful when examples show a stable pattern, such as routing helpdesk tickets into known categories.

This repository is intentionally small and synthetic. It is useful for learning workflows, not for production routing.
