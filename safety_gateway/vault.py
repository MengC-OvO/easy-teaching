"""Ephemeral, local-only storage for plaintext placeholder mappings."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
import secrets
import time

from safety_gateway.redaction import PrivateMapping, restore_with_mappings


class MappingNotFoundError(LookupError):
    """The opaque mapping id is unknown, expired, or already consumed."""


@dataclass(frozen=True)
class VaultEntry:
    expires_at: float
    mappings: tuple[PrivateMapping, ...]


class InMemoryMappingVault:
    """One-process TTL vault; replaceable by an encrypted durable adapter later."""

    def __init__(self, ttl_seconds: int = 3600) -> None:
        if ttl_seconds < 1:
            raise ValueError("ttl_seconds must be positive")
        self._ttl_seconds = ttl_seconds
        self._entries: dict[str, VaultEntry] = {}
        self._lock = asyncio.Lock()

    async def save(self, mappings: tuple[PrivateMapping, ...]) -> str | None:
        if not mappings:
            return None
        mapping_id = secrets.token_urlsafe(32)
        async with self._lock:
            self._purge_expired()
            self._entries[mapping_id] = VaultEntry(
                expires_at=time.monotonic() + self._ttl_seconds,
                mappings=mappings,
            )
        return mapping_id

    async def consume_and_restore(self, mapping_id: str, text: str) -> str:
        """Restore once and delete immediately to reduce plaintext lifetime."""
        async with self._lock:
            self._purge_expired()
            entry = self._entries.pop(mapping_id, None)
        if entry is None:
            raise MappingNotFoundError("mapping is unavailable")
        return restore_with_mappings(text, entry.mappings)

    async def discard(self, mapping_id: str) -> None:
        async with self._lock:
            self._entries.pop(mapping_id, None)

    def _purge_expired(self) -> None:
        now = time.monotonic()
        expired = [key for key, entry in self._entries.items() if entry.expires_at <= now]
        for key in expired:
            del self._entries[key]
