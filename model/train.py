"""Train a compact Hugging Face-compatible ticket classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, f1_score
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import WordLevelTrainer
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, BertConfig, PreTrainedTokenizerFast


LABELS = (
    "policy_question",
    "login_issue",
    "incident_report",
    "training_request",
    "data_privacy",
    "document_access",
)
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}


class TicketDataset(Dataset[dict[str, torch.Tensor]]):
    """PyTorch dataset backed by synthetic JSONL records."""

    def __init__(self, rows: list[dict[str, str]], tokenizer: PreTrainedTokenizerFast, max_length: int) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        encoded = self.tokenizer(
            row["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in encoded.items()}
        item["labels"] = torch.tensor(LABEL_TO_ID[row["label"]], dtype=torch.long)
        return item


def read_jsonl(path: Path) -> list[dict[str, str]]:
    """Read a JSONL data split."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def train_tokenizer(rows: list[dict[str, str]], output_dir: Path) -> PreTrainedTokenizerFast:
    """Train and save a lightweight word-level tokenizer."""
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"], min_frequency=1)
    tokenizer.train_from_iterator((row["text"] for row in rows), trainer=trainer)
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", tokenizer.token_to_id("[CLS]")),
            ("[SEP]", tokenizer.token_to_id("[SEP]")),
        ],
    )
    wrapped = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )
    wrapped.model_input_names = ["input_ids", "attention_mask"]
    wrapped.model_max_length = 160
    wrapped.save_pretrained(output_dir)
    tokenizer_config_path = output_dir / "tokenizer_config.json"
    tokenizer_config = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
    tokenizer_config["model_input_names"] = ["input_ids", "attention_mask"]
    tokenizer_config["model_max_length"] = 160
    tokenizer_config_path.write_text(json.dumps(tokenizer_config, indent=2) + "\n", encoding="utf-8")
    return wrapped


def build_model(vocab_size: int) -> AutoModelForSequenceClassification:
    """Create a compact BERT classifier from configuration."""
    config = BertConfig(
        vocab_size=vocab_size,
        num_hidden_layers=2,
        hidden_size=96,
        intermediate_size=192,
        num_attention_heads=4,
        max_position_embeddings=160,
        dropout=0.1,
        attention_probs_dropout_prob=0.1,
        hidden_dropout_prob=0.1,
        num_labels=len(LABELS),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )
    return AutoModelForSequenceClassification.from_config(config)


def evaluate(model: torch.nn.Module, loader: DataLoader[dict[str, torch.Tensor]]) -> dict[str, float]:
    """Evaluate accuracy and macro F1."""
    model.eval()
    predictions: list[int] = []
    labels: list[int] = []
    with torch.no_grad():
        for batch in loader:
            expected = batch.pop("labels")
            logits = model(**batch).logits
            predictions.extend(torch.argmax(logits, dim=-1).tolist())
            labels.extend(expected.tolist())
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1_macro": float(f1_score(labels, predictions, average="macro", zero_division=0)),
    }


def train(args: argparse.Namespace) -> dict[str, float]:
    """Train the model and save Hugging Face artifacts."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = read_jsonl(Path(args.train))
    validation_rows = read_jsonl(Path(args.validation))
    tokenizer = train_tokenizer(train_rows + validation_rows, output_dir)
    model = build_model(vocab_size=len(tokenizer))

    train_loader = DataLoader(TicketDataset(train_rows, tokenizer, args.max_length), batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(TicketDataset(validation_rows, tokenizer, args.max_length), batch_size=args.batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    for _ in range(args.epochs):
        model.train()
        for batch in train_loader:
            labels = batch.pop("labels")
            loss = model(**batch, labels=labels).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    metrics = evaluate(model, validation_loader)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    (output_dir / "training_args.json").write_text(json.dumps(vars(args), indent=2) + "\n", encoding="utf-8")
    (output_dir / "eval_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the compact public-sector ticket classifier.")
    parser.add_argument("--train", default="data/train.jsonl")
    parser.add_argument("--validation", default="data/validation.jsonl")
    parser.add_argument("--output-dir", default="model")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--max-length", type=int, default=96)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    metrics = train(args)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
