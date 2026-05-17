import os

import pytest

from central_server.scorer import FEATURES, Scorer

MODEL_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "masina_invata",
    "isolation_forest",
    "beth_iforest_model_host2tier",
)


@pytest.fixture(scope="module")
def scorer():
    return Scorer(MODEL_DIR)


def _model_features_match_scorer(scorer):
    meta_feats = scorer._meta.get("features") or []
    return list(meta_feats) == FEATURES


def test_scorer_loads_model(scorer):
    assert scorer._meta, "meta.json should be loaded and non-empty"
    for key in ("low_percentile", "high_percentile", "features"):
        assert key in scorer._meta, f"missing key '{key}' in meta.json"


def test_score_batch_empty_returns_empty(scorer):
    assert scorer.score_batch([], "test-host") == []


def test_score_batch_schema(scorer):
    if not _model_features_match_scorer(scorer):
        pytest.skip(
            f"on-disk model is legacy ({len(scorer._meta.get('features') or [])} features); "
            f"current scorer expects {len(FEATURES)} — retrain to enable this test"
        )

    event = {
        "timestamp": 1.0,
        "processId": 1234,
        "threadId": 1234,
        "userId": 1500,
        "mountNamespace": 4026531840,
        "processName": "bash",
        "eventId": 59,
        "eventName": "execve",
        "argsNum": 2,
        "returnValue": 0,
        "args": "[]",
    }

    results = scorer.score_batch([event], "test-host")

    assert len(results) == 1
    result = results[0]
    for key in ("userId", "severity", "raw_score", "eventName", "processName", "hostname"):
        assert key in result, f"missing key '{key}' in result"
    assert result["severity"] in {0, 1, 2}
    assert result["userId"] == 1500
    assert result["hostname"] == "test-host"
