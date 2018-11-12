import time
from celery import Task
from celery import Celery
from celery.utils.log import get_task_logger


logger = get_task_logger(__name__)
app = Celery('tasks', backend='redis://localhost:6379/0', broker='redis://localhost:6379/0')  # 配置好celery的backend和broker


# class MyTask(Task):
#     def on_success(self, retval, task_id, args, kwargs):
#         print(f'task done: {retval}')
#         return super(MyTask, self).on_success(retval, task_id, args, kwargs)
#
#     def on_failure(self, exc, task_id, args, kwargs, einfo):
#         print(f'task fail, reason: {exc}')
#         return super(MyTask, self).on_failure(exc, task_id, args, kwargs, einfo)


# @app.task(base=MyTask)  # 普通函数装饰为 celery task
# def add(self, x, y):
#     # raise KeyError
#     return x + y


# @app.task(bind=True)  # 普通函数装饰为 celery task
# def add(self, x, y):
#     logger.info(self.request.__dict__)
#     return x + y
#
#
# @app.task(bind=True)
# def test_mes(self):
#     for i in range(1, 11):
#         time.sleep(0.1)
#         self.update_state(state="PROGRESS", meta={'p': i*10})
#     return 'finish'


app.config_from_object('celery_config')


@app.task(bind=True)
def period_task(self):
    print(f'period task done: {self.request.id}')