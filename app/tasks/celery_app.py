"""Celery configuration shared by API publishers and worker processes."""

from celery import Celery

from app.config import settings


def _broker_url() -> str:
    return settings.celery_broker_url or settings.redis_url


celery_app = Celery("easyteaching", broker=_broker_url())
celery_app.conf.update(
    task_default_queue=settings.celery_queue_name,
    task_routes={"easyteaching.execute_conversation": {"queue": settings.celery_queue_name}},
    task_serializer="json",
    accept_content=["json"],
    result_backend=None,
    task_ignore_result=True,
    task_track_started=False,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=settings.celery_worker_prefetch_multiplier,
    task_soft_time_limit=settings.celery_task_soft_time_limit_seconds,
    task_time_limit=settings.celery_task_time_limit_seconds,
    broker_connection_retry_on_startup=True,
    broker_transport_options={
        # Redelivery must not happen while a healthy long model call is still running.
        "visibility_timeout": settings.celery_task_time_limit_seconds + 120,
        "socket_connect_timeout": 5,
        "socket_timeout": 5,
    },
    timezone="Australia/Sydney",
    enable_utc=True,
    imports=("app.tasks.worker",),
)
