from datetime import date
import inspect
import json
from pathlib import Path
from typing import Any, Dict, Optional, Protocol

import httpx
from pydantic import BaseModel

from app.schemas import RiskLevel
from app.tools.definition import (
    ToolCategory,
    ToolDefinition,
    ToolDomain,
    ToolErrorCode,
    ToolExecutionContext,
    ToolPermission,
    ToolResult,
)


class HttpClientProtocol(Protocol):
    def get(self, url: str, *, params: Dict[str, Any], timeout: float): ...


class DailyContextInput(BaseModel):
    target_date: date


class DailyContextOutput(BaseModel):
    target_date: str
    suburb: str
    state: str
    public_holiday: Optional[str] = None
    weather_available: bool
    weather_code: Optional[int] = None
    temperature_max_c: Optional[float] = None
    temperature_min_c: Optional[float] = None
    precipitation_probability_max: Optional[int] = None
    uv_index_max: Optional[float] = None
    alerts: list[str]
    source_urls: list[str]


def build_get_daily_context_tool(
    store: Any,
    *,
    client: Optional[HttpClientProtocol] = None,
    calendar_path: Path | None = None,
) -> ToolDefinition:
    holiday_path = calendar_path or Path("data/calendar/au_public_holidays_2026.json")

    async def async_runtime_handler(input_data: BaseModel, context: ToolExecutionContext) -> ToolResult:
        if not context.teacher_id or not context.class_id:
            return ToolResult.fail(
                code=ToolErrorCode.PERMISSION_DENIED,
                message="Daily context requires a trusted teacher and class scope.",
                risk_level=RiskLevel.L3_FORBIDDEN,
                recoverable=False,
            )
        data = DailyContextInput.model_validate(input_data)
        location = store.get_centre_location(
            teacher_id=context.teacher_id,
            class_id=context.class_id,
        )
        if inspect.isawaitable(location):
            location = await location
        holiday_data = json.loads(holiday_path.read_text(encoding="utf-8"))
        public_holiday = (
            holiday_data.get("holidays", {}).get(data.target_date.isoformat())
            if holiday_data.get("state") == location["state"]
            else None
        )
        weather = await _weather(client, location, data.target_date)
        alerts = _alerts(weather, public_holiday)
        output = DailyContextOutput(
            target_date=data.target_date.isoformat(),
            suburb=location["suburb"],
            state=location["state"],
            public_holiday=public_holiday,
            alerts=alerts,
            source_urls=[
                "https://open-meteo.com/",
                holiday_data.get("source", ""),
            ],
            **weather,
        )
        return ToolResult.ok(data=output.model_dump(mode="json"), risk_level=RiskLevel.L0_READ_ONLY)

    return ToolDefinition(
        name="get_daily_context",
        description=(
            "Get weather and the locally maintained public-holiday context for the "
            "current centre. Call it only for a date-sensitive activity, especially "
            "outdoor planning; it sends only the centre suburb to Open-Meteo."
        ),
        category=ToolCategory.SAFETY,
        input_model=DailyContextInput,
        output_model=DailyContextOutput,
        risk_level=RiskLevel.L0_READ_ONLY,
        permission=ToolPermission.AUTO_EXECUTE,
        domain=ToolDomain.EXTERNAL,
        parallel_safe=True,
        async_runtime_handler=async_runtime_handler,
    )


async def _weather(
    client: Optional[HttpClientProtocol],
    location: Dict[str, Any],
    target: date,
) -> Dict[str, Any]:
    async def get(url: str, params: Dict[str, Any]):
        if client is not None:
            response = client.get(url, params=params, timeout=10.0)
            return await response if inspect.isawaitable(response) else response
        async with httpx.AsyncClient() as owned:
            return await owned.get(url, params=params, timeout=10.0)

    latitude = location.get("latitude")
    longitude = location.get("longitude")
    if latitude is None or longitude is None:
        geocoding = await get(
            "https://geocoding-api.open-meteo.com/v1/search",
            {
                "name": location["suburb"],
                "count": 1,
                "language": "en",
                "format": "json",
            },
        )
        geocoding.raise_for_status()
        locations = geocoding.json().get("results", [])
        if not locations:
            return {"weather_available": False}
        latitude = locations[0]["latitude"]
        longitude = locations[0]["longitude"]
    forecast = await get(
        "https://api.open-meteo.com/v1/forecast",
        {
            "latitude": latitude,
            "longitude": longitude,
            "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max,uv_index_max",
            "start_date": target.isoformat(),
            "end_date": target.isoformat(),
            "timezone": location.get("timezone") or "auto",
        },
    )
    forecast.raise_for_status()
    daily = forecast.json().get("daily", {})
    return {
        "weather_available": True,
        "weather_code": _first(daily.get("weather_code")),
        "temperature_max_c": _first(daily.get("temperature_2m_max")),
        "temperature_min_c": _first(daily.get("temperature_2m_min")),
        "precipitation_probability_max": _first(daily.get("precipitation_probability_max")),
        "uv_index_max": _first(daily.get("uv_index_max")),
    }


def _alerts(weather: Dict[str, Any], holiday: Optional[str]) -> list[str]:
    alerts = []
    if holiday:
        alerts.append(f"Public holiday: {holiday}.")
    if (weather.get("uv_index_max") or 0) >= 6:
        alerts.append("High UV: include shade, hats, sunscreen and exposure limits.")
    if (weather.get("temperature_max_c") or 0) >= 30:
        alerts.append("High temperature: add hydration, shade and shorter outdoor periods.")
    if (weather.get("precipitation_probability_max") or 0) >= 60:
        alerts.append("Likely rain: prepare a wet-weather alternative.")
    return alerts


def _first(values):
    return values[0] if isinstance(values, list) and values else None
