import json

from evals.production_online import _percentile, parse_sse_events


def test_parse_sse_events_keeps_valid_data_frames_only() -> None:
    payload = {"event": "completed", "data": {"status": "completed"}}
    text = (
        "event: completed\n"
        f"data: {json.dumps(payload)}\n\n"
        ": heartbeat\n\n"
        "data: not-json\n\n"
    )

    assert parse_sse_events(text) == [payload]


def test_production_percentile_is_deterministic() -> None:
    assert _percentile([10, 20, 30, 40, 50], 0.50) == 30
    assert _percentile([10, 20, 30, 40, 50], 0.95) == 50
