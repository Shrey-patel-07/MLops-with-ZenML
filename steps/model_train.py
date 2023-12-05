import logging
import pandas as pd

from zenml import step
from src.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin
from .config import ModelNameConfig

from zenml.client import Client
import mlflow
experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def train_model(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    config: ModelNameConfig
    ) -> RegressorMixin:
    """Train the model on the ingested data.

    Args:
        X_train: Training Data
        y_train: Training Labels,
        X_test: Testing Data,
        y_test: Testing Labels
    """
    try:
        model = None
        if config.model_name == "LinearRegression":
            mlflow.sklearn.autolog()
            model = LinearRegressionModel().train(X_train, y_train)
            return model
        else:
            logging.error("Model not found.")
            raise ValueError    ("Model not found.")
    except Exception as e:
        logging.error("Error in training model: {}".format(e))
        raise e