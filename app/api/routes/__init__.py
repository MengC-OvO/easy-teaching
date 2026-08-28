"""HTTP route groups for the EasyTeaching API."""

from app.api.routes.drafts import router as drafts_router
from app.api.routes.events import router as events_router
from app.api.routes.messages import router as messages_router
from app.api.routes.sessions import router as sessions_router
from app.api.routes.approvals import router as approvals_router
from app.api.routes.uploads import router as uploads_router

__all__ = [
    "drafts_router",
    "events_router",
    "messages_router",
    "sessions_router",
    "approvals_router",
    "uploads_router",
]
