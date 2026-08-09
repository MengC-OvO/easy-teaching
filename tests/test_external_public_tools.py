from datetime import date

from app.tools import ToolDomain, ToolRegistry
from app.tools.controlled_tools.external_public import (
    build_get_public_weather_tool,
    build_search_public_resources_tool,
)


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


def test_public_search_returns_source_links() -> None:
    registry = ToolRegistry()
    tool = build_search_public_resources_tool(
        StubClient(
            [
                {
                    "query": {
                        "pages": {
                            "1": {
                                "index": 1,
                                "title": "Outdoor education",
                                "extract": "Public background.",
                                "fullurl": "https://example.test/outdoor",
                            }
                        }
                    }
                }
            ]
        )
    )
    registry.register(tool)

    result = registry.execute(
        "search_public_resources",
        {"query": "outdoor education"},
    )

    assert result.success is True
    assert tool.domain is ToolDomain.EXTERNAL
    assert result.data["items"][0]["url"] == "https://example.test/outdoor"


def test_public_weather_uses_geocoding_then_forecast() -> None:
    registry = ToolRegistry()
    registry.register(
        build_get_public_weather_tool(
            StubClient(
                [
                    {
                        "results": [
                            {"name": "Sydney", "latitude": -33.86, "longitude": 151.2}
                        ]
                    },
                    {
                        "daily": {
                            "weather_code": [3],
                            "temperature_2m_max": [22.5],
                            "temperature_2m_min": [14.0],
                            "precipitation_probability_max": [30],
                        }
                    },
                ]
            )
        )
    )

    result = registry.execute(
        "get_public_weather",
        {"location": "Sydney", "forecast_date": date(2026, 8, 8)},
    )

    assert result.success is True
    assert result.data["temperature_max_c"] == 22.5
    assert result.data["forecast_date"] == "2026-08-08"
