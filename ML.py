import os
import tarfile
from six.moves import urllib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

import numpy as np


# To view data
desired_width = 320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 10)
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = "datasets/housing"
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"

def fetch_housing_data(housing_url = HOUSING_URL, housing_path = HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path = housing_path)
    housing_tgz.close()

def load_housing_data(housing_path = HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

def check_data(hd):
    print(hd.head())
    print(hd.info())
    print(hd.describe())

    hd.hist(bins = 50, figsize = (20,15))
    plt.show()

# Creating a test set
# Scikit has its own built in function that will be used instead
# def split_train_test(data, test_ratio):
#     shuffled_indicies = np.random.permutation(len(data))
#     test_set_size = int(len(data) * test_ratio)
#     test_indicies = shuffled_indicies[:test_set_size]
#     train_indicies = shuffled_indicies[test_set_size:]
#     return data.iloc[train_indicies], data.ilco[test_indicies]

#purely random sampling method
# def get_data_sets(housing):
#     train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
#     return train_set, test_set

#Sampling Based on Income

def get_data_sets(housing):

    #introduce 5 income categories to dataset
    housing["income_cat"] = np.ceil(housing["median_income"]/ 1.5)
    housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace = True)

    split = StratifiedShuffleSplit(n_splits = 1, test_size= 0.2, random_state= 42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]


    # remove Income Category
    for set_ in (strat_test_set, strat_train_set):
        set_.drop("income_cat", axis =1, inplace = True)

    return strat_train_set, strat_train_set

def geo_scatter(housing):

    housing.plot(kind = "scatter", x = "longitude", y = "latitude", alpha = 0.4,
                 s=housing["population"]/100, label = "population", figsize =(10,7),
                 c = "median_house_value", cmap=plt.get_cmap("jet"), colorbar = True)
    plt.legend()

    plt.show()
"""
Data prep now done with pipeline in process data fx
"""
# def data_prep(strat_train_set):
#
#     #seperate predictors and labels
#     housing = strat_train_set.drop("median_house_value", axis = 1)
#     housing_labels = strat_train_set["median_house_value"].copy()
#
#     #drop NA for total bedrooms
#     #housing.dropna(subset = ["total_bedrooms"])
#
#     #Using Imputer to replace NA with medians
#     imputer = SimpleImputer(strategy = "median")
#     housing_num = housing.drop("ocean_proximity", axis = 1)
#     imputer.fit(housing_num)
#     X = imputer.transform(housing_num)
#     housing_tr = pd.DataFrame(X, columns = housing_num.columns)
#
#     #Processing text values (Ocean category)
#     # encoder = LabelEncoder()
#     housing_cat = housing["close_proximity"]
#     # housing_cat_encoded = encoder.fit_transform(housing_cat)
#     # encoder = OneHotEncoder()
#     # housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1,1))
#     #Using Label Binarizer
#     encoder = LabelBinarizer
#     housing_cat_1hot = encoder.fit_transform(housing_cat)

def add_extra_features(X, add_bedrooms_per_room=True):


    rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
    population_per_household = X[:, population_ix] / X[:, household_ix]
    if add_bedrooms_per_room:
        bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
        return np.c_[X, rooms_per_household, population_per_household,
                     bedrooms_per_room]
    else:
        return np.c_[X, rooms_per_household, population_per_household]


def process_data(strat_train_set):

    housing = strat_train_set.drop("median_house_value", axis=1)

    # housing = strat_train_set.copy()
    # #housing.drop("median_house_value", axis = 1)
    housing_num = housing.drop('ocean_proximity', axis = 1)

    num_attribs = list(housing_num)
    cat_attribs = ["ocean_proximity"]

    num_pipeline = Pipeline([
                             ('imputer', SimpleImputer(strategy="median")),
                             ('attribs_adder', FunctionTransformer(add_extra_features, validate= False)),
                             ('std_scaler', StandardScaler()),
                             ])

    #housing_num_tr = num_pipeline.fit_transform(housing_num)

    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs)])

    housing_prepped = full_pipeline.fit_transform(housing)

    return housing_prepped

def RanForestModel(data, label):
    forest_reg = RandomForestRegressor(n_estimators=100, random_state= 42)
    forest_reg.fit(data, label)

    scores = cross_val_score(forest_reg, data, label, scoring= "neg_mean_squared_error", cv = 10)

    return scores

def LinModel(data, label):
    lin_reg = LinearRegression()
    lin_reg.fit(data, label)

    scores = cross_val_score(lin_reg, data, label, scoring= "neg_mean_squared_error", cv = 10)

    return scores



def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard Deviation:", scores.std())


def main():
    fetch_housing_data()
    housing_raw = load_housing_data()
    #check_data(housing_raw)
    _, strat_train_set = get_data_sets(housing_raw)

    housing = strat_train_set.copy()
    housing_labels = strat_train_set["median_house_value"].copy()

    #geo_scatter(housing)

    hp = process_data(strat_train_set)
    #print(hp)

    # scores = RanForestModel(hp, housing_labels)
    scores = LinModel(hp, housing_labels)
    display_scores(scores)

if __name__ == "__main__":
    main()