"""HTTP route groups for the EduFlow API."""

from app.api.routes.approvals import router as approvals_router
from app.api.routes.drafts import router as drafts_router
from app.api.routes.events import router as events_router
from app.api.routes.messages import router as messages_router
from app.api.routes.sessions import router as sessions_router

__all__ = [
    "approvals_router",
    "drafts_router",
    "events_router",
    "messages_router",
    "sessions_router",
]
