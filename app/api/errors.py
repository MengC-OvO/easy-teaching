"""Lightweight API-domain exceptions shared without importing database modules."""


class ConversationSessionBusyError(RuntimeError):
    """Raised when another active run wins the session-level race."""

    def __init__(self, active_request_id: str) -> None:
        super().__init__("Conversation session already has an active run")
        self.active_request_id = active_request_id
