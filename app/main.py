from fastapi import FastAPI

from app.config import settings


app = FastAPI(
    title="EduFlow AU Agent",
    description="Teacher workflow agent backend for synthetic Australian early childhood education scenarios.",
    version="0.1.0",
)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {
        "status": "ok",
        "service": settings.app_name,
        "environment": settings.app_env,
    }
