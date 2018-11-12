from datetime import timedelta
from celery.schedules import crontab


CELERYBEAT_SCHEDULE = {
    'pask': {
        'task': 'task.period_task',
        'schedule': timedelta(seconds=5),
    }
}
CELERY_RESULT_BACKEND = 'redis://localhost:6379/0'
