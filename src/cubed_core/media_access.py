from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass

MEDIA_TICKET_TTL_SECONDS = 10 * 60


@dataclass(frozen=True)
class MediaTicket:
    token: str
    capture_id: str
    expires_at: float


class MediaTicketRegistry:
    """Short-lived, read-only grants for one capture's video."""

    def __init__(self) -> None:
        self._items: dict[str, MediaTicket] = {}
        self._lock = threading.Lock()

    def create(self, capture_id: str) -> MediaTicket:
        now = time.time()
        ticket = MediaTicket(
            token=secrets.token_urlsafe(32),
            capture_id=capture_id,
            expires_at=now + MEDIA_TICKET_TTL_SECONDS,
        )
        with self._lock:
            self._purge(now)
            self._items[ticket.token] = ticket
        return ticket

    def authorize(self, capture_id: str, token: str) -> MediaTicket | None:
        if not token:
            return None
        now = time.time()
        with self._lock:
            self._purge(now)
            ticket = self._items.get(token)
            if ticket is None or not secrets.compare_digest(ticket.capture_id, capture_id):
                return None
            return ticket

    def _purge(self, now: float) -> None:
        expired = [token for token, ticket in self._items.items() if ticket.expires_at <= now]
        for token in expired:
            self._items.pop(token, None)
