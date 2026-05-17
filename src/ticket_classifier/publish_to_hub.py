"""Explicit, opt-in publishing to Hugging Face Hub."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, HfFolder, whoami
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ticket_classifier.model_card import write_model_card


def has_huggingface_auth() -> bool:
    """Return True when a token is available from env or huggingface-cli login."""
    if os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN"):
        return True
    return HfFolder.get_token() is not None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Publish a trained model to Hugging Face Hub.")
    parser.add_argument("--model-dir", required=True, help="Directory containing a saved model and tokenizer.")
    parser.add_argument("--repo-id", required=True, help="Hugging Face repo id, for example USERNAME/synthetic-public-sector-ticket-classifier.")
    parser.add_argument("--private", action="store_true", help="Create or update the Hub repo as private.")
    parser.add_argument("--confirm-publish", action="store_true", help="Required. Without this flag, no publishing happens.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.confirm_publish:
        print("Refusing to publish. Re-run with --confirm-publish after reviewing data, license, model card, and behavior.")
        return 2
    if not has_huggingface_auth():
        raise RuntimeError("No Hugging Face token found. Set HUGGINGFACE_TOKEN or run `huggingface-cli login`.")

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"model directory does not exist: {model_dir}")

    print("Public publishing requires review of data, license, model behavior, and generated README.md model card.")
    write_model_card(model_dir / "README.md", repo_id=args.repo_id)

    api = HfApi()
    whoami()
    api.create_repo(repo_id=args.repo_id, private=args.private, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer.push_to_hub(args.repo_id, private=args.private)
    model.push_to_hub(args.repo_id, private=args.private)
    api.upload_file(path_or_fileobj=str(model_dir / "README.md"), path_in_repo="README.md", repo_id=args.repo_id)
    print(f"Published model, tokenizer, and model card to https://huggingface.co/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
