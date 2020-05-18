import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing as prep
from sklearn.model_selection import cross_val_predict, GridSearchCV
from utils import pre


def read_adata_h5ad(path):
    adata = pre.read_sc_data(path)
    return adata.X, adata.obs['class']


def read_data_csv():
    # the classes of these train samples
    classes = pd.read_csv('pp5i_train_class.txt', sep=' ', header=None)
    classes.drop(index=[0], inplace=True)
    classes = np.array(classes).flatten()

    # the features of these train samples
    features = pd.read_csv('pp5i_train.gr.csv', header=None)
    features.drop(index=[0], inplace=True)
    features.drop(columns=0, inplace=True)
    features = np.array(features).T

    return features, classes


def standard_scale(features):
    preprocessor = prep.StandardScaler()
    features = preprocessor.fit_transform(features)
    return features


def cross_val(model, data, target):
    predicts = cross_val_predict(estimator=model, X=data, y=target, cv=4)
    # X_train, X_test, y_train, y_test = train_test_split(features, classes, test_size=0.3, random_state=42)
    return predicts


def search_parameters(dataset, target):
    param_test1 = {'n_estimators': range(100, 400, 20)}
    search1 = GridSearchCV(estimator=RandomForestClassifier(min_samples_split=100,
                                                            min_samples_leaf=20, max_depth=10, max_features='sqrt',
                                                            random_state=10),
                           param_grid=param_test1,  cv=10)
    search1.fit(dataset, target)
    print(search1.best_params_, search1.best_score_)


