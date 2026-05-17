from inference.model import TicketClassifier


def test_keyword_smoke_known_labels(tmp_path) -> None:
    classifier = TicketClassifier(tmp_path, "1.2.0", confidence_threshold=0.0)
    classifier.load()
    predictions = classifier.classify_batch(
        [
            "I cannot login because my password is locked",
            "Please approve this software license procurement",
            "We need CAB approval for a release change",
        ],
    )
    assert [prediction.label for prediction in predictions] == ["login_issue", "procurement_request", "change_management"]
