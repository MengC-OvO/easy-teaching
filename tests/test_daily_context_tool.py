import asyncio
import json
from datetime import date

from app.tools import ToolDomain, ToolExecutionContext, ToolRegistry
from app.tools.controlled_tools.daily_context import build_get_daily_context_tool


class StubResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class StubClient:
    def __init__(self, responses):
        self.responses = iter(responses)

    def get(self, url, *, params, timeout):
        return StubResponse(next(self.responses))


class StubStore:
    def get_centre_location(self, *, teacher_id, class_id):
        assert teacher_id == "teacher-1"
        assert class_id == "kangaroo-room"
        return {"centre_id": "demo-centre", "suburb": "Sydney", "state": "NSW"}


def test_daily_context_combines_trusted_location_weather_and_holiday(tmp_path) -> None:
    calendar = tmp_path / "holidays.json"
    calendar.write_text(
        json.dumps(
            {
                "state": "NSW",
                "source": "https://example.test/nsw-holidays",
                "holidays": {"2026-08-08": "Synthetic holiday"},
            }
        ),
        encoding="utf-8",
    )
    registry = ToolRegistry()
    tool = build_get_daily_context_tool(
        StubStore(),
        calendar_path=calendar,
        client=StubClient(
            [
                {
                    "results": [
                        {"name": "Sydney", "latitude": -33.86, "longitude": 151.2}
                    ]
                },
                {
                    "daily": {
                        "weather_code": [3],
                        "temperature_2m_max": [31.5],
                        "temperature_2m_min": [14.0],
                        "precipitation_probability_max": [70],
                        "uv_index_max": [8.0],
                    }
                },
            ]
        ),
    )
    registry.register(tool)

    result = asyncio.run(
        registry.execute_async(
            "get_daily_context",
            {"target_date": date(2026, 8, 8)},
            execution_context=ToolExecutionContext(
                teacher_id="teacher-1", class_id="kangaroo-room"
            ),
        )
    )

    assert result.success is True
    assert tool.domain is ToolDomain.EXTERNAL
    assert result.data["suburb"] == "Sydney"
    assert result.data["public_holiday"] == "Synthetic holiday"
    assert result.data["temperature_max_c"] == 31.5
    assert len(result.data["alerts"]) == 4
