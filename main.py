# https://yq.aliyun.com/articles/652016?spm=a2c4e.11153940.blogcont206197.20.38232a2cnE4zm2
import os
import hashlib
import tarfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 或者import urllib
from six.moves import urllib
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import Imputer
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedShuffleSplit


DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = "datasets/housing"
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6

    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        rooms_per_household = X[:, self.rooms_ix] / X[:, self.household_ix]
        population_per_household = X[:, self.population_ix] / X[:, self.household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, self.bedrooms_ix] / X[:, self.rooms_ix]
            return np.c_[rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].values


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    def schedule(a, b, c):
        """
        下载进度函数
        :param a:
        :param b:
        :param c:
        :return:
        """
        per = 100. * a * b / c
        if per > 100:
            per = 100
        print("%.2f%%" % per)

    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path, schedule)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


def plot_data_as_column(data):
    data.hist(bins=50, figsize=(20, 15))
    plt.show()


def split_train_test_by_random(data, test_ration):
    """
    随机数生成测试数据集，每次运行结果不一致
    :param data:
    :param test_ration:
    :return:
    """
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ration)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


def test_set_check(identifier, test_ratio, hash):
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio


def split_train_test_by_id(data, test_ration, id_column, hash=hashlib.md5):
    """
    用数据识别码作为作为是否为测试数据集的判定标准，每次运行结果一致
    :param data:
    :param test_ration:
    :param id_column:
    :param hash:
    :return:
    """
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ration, hash))
    return data.loc[~in_test_set], data.loc[in_test_set]


def get_split_data(housing):
    """
    分割训练数据集和测试数据集
    方式一：  -->随机数生成测试数据集
        train_set, test_set = split_train_test_by_random(housing, 0.2)
    方式二：  -->数字id索引作为识别码
        housing_with_id = housing.reset_index()
        train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")
    方式三：  -->以稳定特征（经度纬度结合）id作为识别码
        housing_with_id = housing
        housing_with_id["id"] = housing_with_id["longitude"] * 1000 + housing_with_id["latitude"]
        train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")
    方式四：  -->scikit-learn提供的方式
        train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
    方式五：  -->分层采样
        housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
        housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_index, test_index in split.split(housing, housing["income_cat"]):
            start_train_set = housing.loc[train_index]
            start_test_set = housing.loc[test_index]
    """

    housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
    housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        start_train_set = housing.loc[train_index]
        start_test_set = housing.loc[test_index]
    for s in (start_train_set, start_test_set):
        s.drop(["income_cat"], axis=1, inplace=True)
    return start_train_set, start_test_set


def plot_scatter_by_column(housing):
    # 地理信息散点图
    housing.plot(kind="scatter", x="longitude", y="latitude")
    # 高密度区域散点图
    housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)

    housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
                 s=housing["population"] / 100, label="population",
                 c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
                 )
    plt.legend()
    plt.show()


def calculate_correlation_matrix(housing):
    """
    属性相关系数
    :param housing:
    :return:
    """
    # 计算属性与属性间的相关关系
    corr_matrix = housing.corr()
    print(corr_matrix["median_house_value"].sort_values(ascending=False))


def plot_correlation_matrix(housing):
    # 画出最可能与房价相关属性的散点图
    attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
    scatter_matrix(housing[attributes], figsize=(12, 8))
    # 收入与房价的关系
    housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)
    plt.show()


def add_new_column(housing):
    housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
    housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
    housing["population_per_household"] = housing["population"] / housing["households"]
    return housing


def analysis_data():
    housing = load_housing_data()
    train, test = get_split_data(housing)
    housing = train.copy()
    plot_scatter_by_column(housing)
    print("原始数据的相关关系：")
    calculate_correlation_matrix(housing)
    housing = add_new_column(housing)
    print("新增属性后的相关关系")
    calculate_correlation_matrix(housing)
    plot_correlation_matrix(housing)


def process_missing_data(housing):
    """
    方法1：去掉对应的数据
    方法2：去掉整个属性
    方法3：赋值（0，平均值，中位数等）

    pandas.DataFrame中的dropna(), drop(), fillna()实现
        housing.dropna(subset=["total_bedrooms"])    # 方法1
        housing.drop("total_bedrooms", axis=1)       # 方法2
        median = housing["total_bedrooms"].median()  # 计算中位数
        housing["total_bedrooms"].fillna(median)     # 方法3

    scikit-learn中的Imputer
        imputer = Imputer(strategy="median")
        housing_num = housing.drop("ocean_proximity", axis=1)
        imputer.fix(housing_num)
        X = imputer.transform(housing_num)
    """
    imputer = Imputer(strategy="median")
    housing_num = housing.drop("ocean_proximity", axis=1)
    imputer.fix(housing_num)
    X = imputer.transform(housing_num)
    housing_tr = pd.DataFrame(X, columns=housing_num.columns)


def transform_object_to_num(housing):
    """
    处理文字属性
    方法1：标签转化
        encoder = LabelEncoder()
        housing_cat = housing["ocean_proximity"]
        housing_cat_encoded = encoder.fix_transform(housing_cat)
    方法2：One_Hot转化
        encoder = LabelEncoder()
        housing_cat = housing["ocean_proximity"]
        housing_cat_encoded = encoder.fit_transform(housing_cat)
        encoder = OneHotEncoder()
        housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1, 1))
    方法3：一步完成方法2
        encoder = LabelBinarizer()
        housing_cat_1hot = encoder.fit_transform(housing_cat)
    """

    housing_cat = housing["ocean_proximity"]
    encoder = LabelBinarizer()
    housing_cat_1hot = encoder.fit_transform(housing_cat)

    # 自定义转化量
    attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
    housing_extra_attribs = attr_adder.transform(housing.values)


def pipeline_transform(housing):
    housing_num = housing.drop("ocean_proximity", axis=1)

    num_attribs = list(housing_num)
    cat_attribs = ["ocean_proximity"]
    num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attribs)),
        ('imputer', Imputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])
    cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_attribs)),
        ('label_binarizer', LabelBinarizer()),
    ])
    full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])
    housing_prepared = full_pipeline.fit_transform(housing)
    return housing_prepared, full_pipeline


def cleaning_data():
    housing = load_housing_data()
    train_data, test_data = get_split_data(housing)
    housing = train_data.drop("median_house_value", axis=1)
    housing_labels = train_data["median_house_value"].copy()
    # process_missing_data(housing)
    # transform_object_to_num(housing)
    housing_prepared, full_pipeline = pipeline_transform(housing)
    return housing, housing_prepared, housing_labels, full_pipeline


def train_model():
    housing, housing_prepared, housing_labels, full_pipeline = cleaning_data()
    linear_reg = LinearRegression()
    linear_reg.fix(housing_prepared, housing_labels)
    some_data = housing.iloc[:5]
    some_labels = housing_labels.iloc[:5]
    some_data_prepared = full_pipeline.transform(some_data)
    print("Predictions:\t", linear_reg.predict(some_data_prepared))
    print("Labels:\t\t", list(some_labels))


if __name__ == '__main__':
    # analysis_data()
    # cleaning_data()
    train_model()