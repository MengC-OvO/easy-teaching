"""Allowlisted official-site search through Google Programmable Search."""

from __future__ import annotations

from typing import Any, List, Optional
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, Field


DEFAULT_OFFICIAL_DOMAINS = (
    "acecqa.gov.au",
    "education.gov.au",
    "education.nsw.gov.au",
    "vic.gov.au",
    "qld.gov.au",
    "education.wa.edu.au",
    "education.sa.gov.au",
    "education.tas.gov.au",
    "education.act.gov.au",
    "education.nt.gov.au",
)


class OfficialSearchResult(BaseModel):
    title: str
    snippet: str
    url: str
    domain: str


class OfficialWebSearchResponse(BaseModel):
    query: str
    results: List[OfficialSearchResult]


class GoogleOfficialWebSearchClient:
    def __init__(
        self,
        *,
        api_key: str,
        engine_id: str,
        base_url: str = "https://customsearch.googleapis.com/customsearch/v1",
        timeout_seconds: float = 12.0,
        allowed_domains: tuple[str, ...] = DEFAULT_OFFICIAL_DOMAINS,
        async_client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        if not api_key or not engine_id:
            raise ValueError("Official search requires an API key and engine ID")
        self.api_key = api_key
        self.engine_id = engine_id
        self.base_url = base_url
        self.timeout_seconds = timeout_seconds
        self.allowed_domains = tuple(domain.casefold() for domain in allowed_domains)
        self.client = async_client or httpx.AsyncClient()
        self._owns_client = async_client is None

    async def search(
        self,
        query: str,
        *,
        domains: Optional[List[str]] = None,
        top_k: int = 5,
    ) -> OfficialWebSearchResponse:
        selected = self._validate_domains(domains or list(self.allowed_domains))
        site_query = " OR ".join(f"site:{domain}" for domain in selected)
        response = await self.client.get(
            self.base_url,
            params={
                "key": self.api_key,
                "cx": self.engine_id,
                "q": f"({site_query}) {query}",
                "num": min(top_k, 10),
                "safe": "active",
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
        results: List[OfficialSearchResult] = []
        for item in payload.get("items") or []:
            if not isinstance(item, dict):
                continue
            url = str(item.get("link") or "")
            hostname = (urlparse(url).hostname or "").casefold()
            if not self._host_allowed(hostname, selected):
                continue
            results.append(
                OfficialSearchResult(
                    title=str(item.get("title") or "Untitled")[:300],
                    snippet=str(item.get("snippet") or "")[:1000],
                    url=url,
                    domain=hostname,
                )
            )
            if len(results) >= top_k:
                break
        return OfficialWebSearchResponse(query=query, results=results)

    async def aclose(self) -> None:
        if self._owns_client:
            await self.client.aclose()

    def _validate_domains(self, domains: List[str]) -> List[str]:
        selected = list(dict.fromkeys(domain.casefold().strip() for domain in domains))
        if not selected or any(domain not in self.allowed_domains for domain in selected):
            raise ValueError("Official search domain is not allowlisted")
        return selected

    @staticmethod
    def _host_allowed(hostname: str, domains: List[str]) -> bool:
        return any(hostname == domain or hostname.endswith(f".{domain}") for domain in domains)

