import pickle

from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression


# Optional: implement hyperparameter tuning.
def train(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model


def save_to_file(instance, file_path):
    """
    Save model to local.

    Inputs
    ------
    instance: model 
        Trained model.
    file_name: str
        Name of the model to be saved.
    Returns
    -------
        None 

    """
    with open(file_path, 'wb') as f:
        pickle.dump(instance, f)


def load_file(file_path):
    """
    Load trained model by name

    Inputs
    ------
    file_name: str
        Name of the model to be loaded
    """
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : binary classifier 
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)
