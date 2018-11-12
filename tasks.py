import lightgbm as lgb
from celery import Celery
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing


def load_data():
    housing = fetch_california_housing()
    return housing.data, housing.target, housing.feature_names


# app = Celery('tasks', broker='amqp://guest:guest@localhost', backend='amqp://guest:guest@localhost')
app = Celery('tasks',  backend='redis://127.0.0.1:6379/0', broker='redis://127.0.0.1:6379/1')

data, target, feature_names = load_data()
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2)
lgb_train = lgb.Dataset(x_train, y_train)
lgb_eval = lgb.Dataset(x_test, y_test, reference=lgb_train)


@app.task
def light_gbm(num_leaves):
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': {'l1', 'l2', 'rmse'},
        'num_leaves': num_leaves,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 1,
    }

    print('Start training ...')
    gbm = lgb.train(params, lgb_train, num_boost_round=500, valid_sets=lgb_eval, early_stopping_rounds=10, feature_name=feature_names)
    print('Start predicting ...')
    loss = mean_squared_error(y_test, gbm.predict(x_test, num_iteration=gbm.best_iteration)) ** 0.5
    return loss
