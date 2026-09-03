"""Distributed background execution for EasyTeaching."""

from app.tasks.celery_app import celery_app

__all__ = ["celery_app"]
