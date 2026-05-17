import pytest

from monitoring.drift import DriftDetector, jensen_shannon_divergence


def test_js_score_for_known_distributions() -> None:
    assert jensen_shannon_divergence([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0.0)
    assert jensen_shannon_divergence([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.6931471805599453)


def test_drift_snapshot_contains_shift() -> None:
    detector = DriftDetector(baseline_counts={"login_issue": 10, "policy_question": 10}, window_size=10)
    for _ in range(10):
        detector.observe("login_issue")
    snapshot = detector.snapshot()
    assert snapshot["js_divergence"] > 0
    assert "per_label_shift" in snapshot
