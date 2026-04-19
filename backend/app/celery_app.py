"""Celery-приложение InfoDiode для асинхронной обработки задач."""

from celery import Celery

from app.config import settings

celery_app = Celery(
    "infodiode",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Europe/Moscow",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_concurrency=1,
    worker_prefetch_multiplier=1,
)

# Автообнаружение задач в модулях
celery_app.autodiscover_tasks(["app.core"])

# NOTE: Старые задачи process_video_task и calibrate_video_task удалены.
# Они ссылались на app.core.pipeline (перемещён в deprecated/).
# Для асинхронной обработки видео используйте VLM pipeline через REST API:
#   POST /api/pipeline/start/{video_id}
#
# Если нужны Celery-задачи для VLM pipeline, реализуйте новые задачи в app.core.vlm_pipeline
