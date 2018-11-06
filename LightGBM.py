import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing


def load_data():
    housing = fetch_california_housing()
    return housing.data, housing.target, housing.feature_names


def plot():
    data, target, feature_names = load_data()
    housing = pd.DataFrame(data, columns=feature_names)
    housing['price'] = target
    print(housing.describe())
    housing.hist(bins=50, figsize=(12, 8))

    corr_matrix = housing.corr()
    print(corr_matrix['price'].sort_values(ascending=False))

    # attributes = housing.columns.values.tolist()  # 列名
    # scatter_matrix(housing[attributes], figsize=(12, 8))

    scatter_matrix(housing[feature_names[:4] + ['price']], figsize=(20, 16))
    scatter_matrix(housing[feature_names[4:] + ['price']], figsize=(20, 16))

    housing.plot(kind="scatter", x='MedInc', y="price", alpha=0.1)
    plt.show()


def light_gbm():
    data, target, feature_names = load_data()
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2)
    lgb_train = lgb.Dataset(x_train, y_train)
    lgb_eval = lgb.Dataset(x_test, y_test, reference=lgb_train)

    params = {
        'task': 'train',
        'boosting_type': 'gbdt',           # 设置提升类型
        'objective': 'regression',         # 目标函数
        'metric': {'l1', 'l2', 'rmse'},    # 评估函数
        'num_leaves': 31,                  # 叶子节点数
        'learning_rate': 0.05,             # 学习速率
        'feature_fraction': 0.9,           # 建树的特征选择比例
        'bagging_fraction': 0.8,           # 建树的样本采样比例
        'bagging_freq': 5,                 # k 意味着每 k 次迭代执行bagging
        'verbose': 1,                      # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
        # 'max_depth': 5,
    }

    print('Start training ...')
    gbm = lgb.train(params, lgb_train, num_boost_round=500, valid_sets=lgb_eval, early_stopping_rounds=10, feature_name=feature_names)
    print('Save model ...')
    gbm.save_model('model.txt')
    print('Start predicting ...')
    y_pred = gbm.predict(x_test, num_iteration=gbm.best_iteration)
    print('the rmse of predictions is:', mean_squared_error(y_test, y_pred) ** 0.5)
    print(gbm.feature_name())
    print('Feature importances:', gbm.feature_importance())


def cv():
    data, target, feature_names = load_data()
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2)

    estimator = lgb.LGBMRegressor()
    param_grid = {
        'num_leaves': [30, 31, 32],
        # 'feature_fraction': [0.7, 0.8, 0.9],
        # 'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': list(range(2, 8, 1)),
    }
    gbm = GridSearchCV(estimator, param_grid, cv=3)
    gbm.fit(x_train, y_train)
    print('best parameters:', gbm.best_params_)


if __name__ == '__main__':
    light_gbm()
    plot()
    cv()
