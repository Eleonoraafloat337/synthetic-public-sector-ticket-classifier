import numpy as np

from ticket_classifier.labels import LABELS
from ticket_classifier.train import compute_metrics, write_label_map


def test_compute_metrics_includes_per_label_f1() -> None:
    logits = np.eye(len(LABELS))
    labels = np.arange(len(LABELS))
    metrics = compute_metrics((logits, labels))
    assert metrics["f1_macro"] == 1.0
    assert all(f"f1_{label}" in metrics for label in LABELS)


def test_write_label_map(tmp_path) -> None:
    write_label_map(tmp_path)
    assert (tmp_path / "label_map.json").exists()
