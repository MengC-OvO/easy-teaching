"""不需要密钥的公开信息工具；只接受非身份化参数。"""

from datetime import date
import inspect
from typing import Any, Dict, List, Optional, Protocol

import httpx
from pydantic import BaseModel, Field

from app.schemas import RiskLevel
from app.tools.definition import (
    ToolCategory,
    ToolDefinition,
    ToolDomain,
    ToolPermission,
    ToolResult,
)


class HttpClientProtocol(Protocol):
    def get(self, url: str, *, params: Dict[str, Any], timeout: float):
        ...


class PublicSearchInput(BaseModel):
    query: str = Field(min_length=2, max_length=200)
    limit: int = Field(default=3, ge=1, le=5)


class PublicSearchItem(BaseModel):
    title: str
    summary: str
    url: str


class PublicSearchOutput(BaseModel):
    query: str
    items: List[PublicSearchItem] = Field(default_factory=list)


class PublicWeatherInput(BaseModel):
    location: str = Field(min_length=2, max_length=100)
    forecast_date: Optional[date] = None


class PublicWeatherOutput(BaseModel):
    location: str
    available: bool = True
    message: Optional[str] = None
    forecast_date: Optional[str] = None
    weather_code: Optional[int] = None
    temperature_max_c: Optional[float] = None
    temperature_min_c: Optional[float] = None
    precipitation_probability_max: Optional[int] = None
    source_url: Optional[str] = None


def _get(client: Optional[HttpClientProtocol], url: str, params: Dict[str, Any]):
    if client is not None:
        return client.get(url, params=params, timeout=10.0)
    with httpx.Client() as owned_client:
        return owned_client.get(url, params=params, timeout=10.0)


async def _aget(client: Optional[HttpClientProtocol], url: str, params: Dict[str, Any]):
    if client is not None:
        response = client.get(url, params=params, timeout=10.0)
        return await response if inspect.isawaitable(response) else response
    async with httpx.AsyncClient() as owned_client:
        return await owned_client.get(url, params=params, timeout=10.0)


def build_search_public_resources_tool(
    client: Optional[HttpClientProtocol] = None,
) -> ToolDefinition:
    def handler(args: PublicSearchInput) -> ToolResult:
        response = _get(
            client,
            "https://en.wikipedia.org/w/api.php",
            {
                "action": "query",
                "generator": "search",
                "gsrsearch": args.query,
                "gsrlimit": args.limit,
                "prop": "extracts|info",
                "exintro": 1,
                "explaintext": 1,
                "inprop": "url",
                "format": "json",
                "origin": "*",
            },
        )
        response.raise_for_status()
        pages = response.json().get("query", {}).get("pages", {})
        items = [
            PublicSearchItem(
                title=page.get("title", "Untitled"),
                summary=page.get("extract", "")[:1200],
                url=page.get("fullurl", "https://en.wikipedia.org/"),
            )
            for page in sorted(pages.values(), key=lambda item: item.get("index", 0))
        ]
        output = PublicSearchOutput(query=args.query, items=items)
        return ToolResult.ok(
            data=output.model_dump(mode="json"),
            risk_level=RiskLevel.L0_READ_ONLY,
        )

    async def async_handler(args: PublicSearchInput) -> ToolResult:
        response = await _aget(
            client,
            "https://en.wikipedia.org/w/api.php",
            {
                "action": "query",
                "generator": "search",
                "gsrsearch": args.query,
                "gsrlimit": args.limit,
                "prop": "extracts|info",
                "exintro": 1,
                "explaintext": 1,
                "inprop": "url",
                "format": "json",
                "origin": "*",
            },
        )
        response.raise_for_status()
        pages = response.json().get("query", {}).get("pages", {})
        output = PublicSearchOutput(
            query=args.query,
            items=[
                PublicSearchItem(
                    title=page.get("title", "Untitled"),
                    summary=page.get("extract", "")[:1200],
                    url=page.get("fullurl", "https://en.wikipedia.org/"),
                )
                for page in sorted(pages.values(), key=lambda item: item.get("index", 0))
            ],
        )
        return ToolResult.ok(
            data=output.model_dump(mode="json"),
            risk_level=RiskLevel.L0_READ_ONLY,
        )

    return ToolDefinition(
        name="search_public_resources",
        description=(
            "Search public background resources for non-identifying activity ideas. "
            "Never include child, family, teacher, email, phone, or internal IDs."
        ),
        category=ToolCategory.CURRICULUM,
        input_model=PublicSearchInput,
        output_model=PublicSearchOutput,
        risk_level=RiskLevel.L0_READ_ONLY,
        permission=ToolPermission.AUTO_EXECUTE,
        domain=ToolDomain.EXTERNAL,
        parallel_safe=True,
        timeout_seconds=12.0,
        handler=handler,
        async_handler=async_handler,
    )


def build_get_public_weather_tool(
    client: Optional[HttpClientProtocol] = None,
) -> ToolDefinition:
    def handler(args: PublicWeatherInput) -> ToolResult:
        geocoding = _get(
            client,
            "https://geocoding-api.open-meteo.com/v1/search",
            {"name": args.location, "count": 1, "language": "en", "format": "json"},
        )
        geocoding.raise_for_status()
        locations = geocoding.json().get("results", [])
        if not locations:
            return ToolResult.ok(
                data={
                    "location": args.location,
                    "available": False,
                    "message": "Location was not found.",
                },
                risk_level=RiskLevel.L0_READ_ONLY,
            )

        location = locations[0]
        forecast_day = args.forecast_date or date.today()
        forecast = _get(
            client,
            "https://api.open-meteo.com/v1/forecast",
            {
                "latitude": location["latitude"],
                "longitude": location["longitude"],
                "daily": (
                    "weather_code,temperature_2m_max,temperature_2m_min,"
                    "precipitation_probability_max"
                ),
                "start_date": forecast_day.isoformat(),
                "end_date": forecast_day.isoformat(),
                "timezone": "auto",
            },
        )
        forecast.raise_for_status()
        daily = forecast.json().get("daily", {})
        output = PublicWeatherOutput(
            location=location.get("name", args.location),
            forecast_date=forecast_day.isoformat(),
            weather_code=_first(daily.get("weather_code")),
            temperature_max_c=_first(daily.get("temperature_2m_max")),
            temperature_min_c=_first(daily.get("temperature_2m_min")),
            precipitation_probability_max=_first(
                daily.get("precipitation_probability_max")
            ),
            source_url="https://open-meteo.com/",
        )
        return ToolResult.ok(
            data=output.model_dump(mode="json"),
            risk_level=RiskLevel.L0_READ_ONLY,
        )

    async def async_handler(args: PublicWeatherInput) -> ToolResult:
        geocoding = await _aget(
            client,
            "https://geocoding-api.open-meteo.com/v1/search",
            {"name": args.location, "count": 1, "language": "en", "format": "json"},
        )
        geocoding.raise_for_status()
        locations = geocoding.json().get("results", [])
        if not locations:
            return ToolResult.ok(
                data={
                    "location": args.location,
                    "available": False,
                    "message": "Location was not found.",
                },
                risk_level=RiskLevel.L0_READ_ONLY,
            )
        location = locations[0]
        forecast_day = args.forecast_date or date.today()
        forecast = await _aget(
            client,
            "https://api.open-meteo.com/v1/forecast",
            {
                "latitude": location["latitude"],
                "longitude": location["longitude"],
                "daily": (
                    "weather_code,temperature_2m_max,temperature_2m_min,"
                    "precipitation_probability_max"
                ),
                "start_date": forecast_day.isoformat(),
                "end_date": forecast_day.isoformat(),
                "timezone": "auto",
            },
        )
        forecast.raise_for_status()
        daily = forecast.json().get("daily", {})
        output = PublicWeatherOutput(
            location=location.get("name", args.location),
            forecast_date=forecast_day.isoformat(),
            weather_code=_first(daily.get("weather_code")),
            temperature_max_c=_first(daily.get("temperature_2m_max")),
            temperature_min_c=_first(daily.get("temperature_2m_min")),
            precipitation_probability_max=_first(daily.get("precipitation_probability_max")),
            source_url="https://open-meteo.com/",
        )
        return ToolResult.ok(
            data=output.model_dump(mode="json"),
            risk_level=RiskLevel.L0_READ_ONLY,
        )

    return ToolDefinition(
        name="get_public_weather",
        description=(
            "Get a public weather forecast for a city or suburb. Use only a place "
            "name; never include child, family, address, email, phone, or internal IDs."
        ),
        category=ToolCategory.SAFETY,
        input_model=PublicWeatherInput,
        output_model=PublicWeatherOutput,
        risk_level=RiskLevel.L0_READ_ONLY,
        permission=ToolPermission.AUTO_EXECUTE,
        domain=ToolDomain.EXTERNAL,
        parallel_safe=True,
        timeout_seconds=12.0,
        handler=handler,
        async_handler=async_handler,
    )


def _first(values):
    if isinstance(values, list) and values:
        return values[0]
    return None
