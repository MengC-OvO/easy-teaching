from collections import Counter, defaultdict

import pytest

from scripts.run_local_model_final_eval import class_metrics, prf, redact_from_predictions


def test_prf_and_injection_macro_metrics() -> None:
    assert prf(8, 1, 2)["f1"] == pytest.approx(16 / 19)
    confusion = defaultdict(Counter)
    confusion["normal"].update({"normal": 8, "suspicious": 2})
    confusion["suspicious"].update({"suspicious": 7, "normal": 3})
    confusion["block"].update({"block": 9, "normal": 1})
    result = class_metrics(confusion)
    assert 0 < result["macro_f1"] < 1
    assert result["per_class"]["block"]["recall"] == 0.9


def test_direct_deidentification_uses_only_model_predictions() -> None:
    source = "Ava Lee can be reached at ava@example.test."
    predicted = Counter({("PERSON_NAME", "Ava Lee"): 1, ("EMAIL", "ava@example.test"): 1})
    result = redact_from_predictions(source, predicted)
    assert "Ava Lee" not in result
    assert "ava@example.test" not in result
    assert "<PERSON_NAME_1>" in result
    assert "<EMAIL_1>" in result
