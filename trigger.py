import time
from celery import group
from tasks import light_gbm


# g = group([light_gbm.s(i) for i in range(21, 23)]).apply_async()

task_result = [(i, light_gbm.apply_async((i,))) for i in range(21, 60)]

while True:
    tag = 1
    for key in task_result:
        if not key[1].ready():
            tag = 0
            time.sleep(1)
            print("sleep 1")
    if tag:
        break

result = sorted(task_result, key=lambda t: t[-1].info)
print(result[0], result[0][-1].info)
