from pathlib import Path

from ticket_classifier.model_card import generate_model_card, write_model_card


def test_model_card_generation_contains_required_sections() -> None:
    card = generate_model_card("example/synthetic-public-sector-ticket-classifier")
    assert "## Intended Use" in card
    assert "## Safety Notes" in card
    assert "synthetic examples" in card


def test_model_card_write(tmp_path: Path) -> None:
    path = write_model_card(tmp_path / "README.md", "example/synthetic-public-sector-ticket-classifier")
    assert path.exists()
    assert "pipeline" in path.read_text(encoding="utf-8")
