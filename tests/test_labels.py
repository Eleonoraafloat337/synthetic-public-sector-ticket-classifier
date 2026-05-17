from ticket_classifier.labels import ID_TO_LABEL, LABELS, LABEL_TO_ID, validate_label


def test_label_mappings_are_stable_and_complete() -> None:
    assert LABELS == (
        "policy_question",
        "login_issue",
        "incident_report",
        "training_request",
        "data_privacy",
        "document_access",
    )
    assert set(LABEL_TO_ID) == set(LABELS)
    assert {ID_TO_LABEL[index] for index in ID_TO_LABEL} == set(LABELS)


def test_validate_label_rejects_unknown_label() -> None:
    try:
        validate_label("billing_question")
    except ValueError as exc:
        assert "Invalid label" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid label")
