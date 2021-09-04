import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np

from ml.model import train, inference, compute_model_metrics


@pytest.fixture(scope='session')
def data():
    X, y = make_classification(n_samples=100, n_features=5,
                               n_informative=3, n_redundant=0,
                               n_repeated=0, n_classes=2,
                               random_state=42, shuffle=False)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42)
    return X_train, y_train, X_test, y_test


def test_model_return_object(data):
    X_train, y_train, X_test, y_test = data

    lr = LogisticRegression()
    clf = train(X_train, y_train)
    assert type(lr) == type(clf),\
        f'model is not {type(lr)}, got {type(clf)} instead.'


def test_inference(data):
    X_train, y_train, X_test, y_test = data
    clf = train(X_train, y_train)
    preds = inference(clf, X_test)

    assert len(preds) == len(y_test),\
        f'length of predicted values does not match, expected: {len(y_test)},\
            but got {len(preds)} instead.'

    assert isinstance(preds, np.ndarray),\
        f'preds is not an numpy array, got {type(preds)} instead.'


def test_compute_metrics(data):
    X_train, y_train, X_test, y_test = data
    clf = train(X_train, y_train)
    preds = inference(clf, X_test)

    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    assert isinstance(precision, float), \
        f'precision is not a float, got {type(precision)} instead'
    assert isinstance(recall, float), \
        f'recall is not a float, got {type(recall)} instead'
    assert isinstance(fbeta, float), \
        f'fbeta is not a float, got {type(fbeta)} instead'
