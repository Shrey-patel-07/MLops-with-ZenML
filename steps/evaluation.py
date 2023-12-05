import logging
from typing import Tuple

import pandas as pd
from sklearn.base import RegressorMixin
from zenml import step
from zenml.client import Client

from typing_extensions import Annotated

from src.evaluation import MSE, RMSE, R2
import mlflow
experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(model: RegressorMixin,
                   X_test: pd.DataFrame,
                   y_test: pd.DataFrame
                ) -> Tuple[
                    Annotated[float, "r2"],
                    Annotated[float, "rmse"]
                ]:
    """
    Evaluate the model on the ingested data.
    
    Args:
        df: the ingested data
    """
    try:
        predictions = model.predict(X_test)
        mse_class = MSE()
        mse = mse_class.calculate_score(y_test, predictions)
        mlflow.log_metric("mse", mse)
        
        r2_class = R2()
        r2 = r2_class.calculate_score(y_test, predictions)
        mlflow.log_metric("r2", r2)
        
        rmse_class = RMSE()
        rmse = rmse_class.calculate_score(y_test, predictions)
        mlflow.log_metric("rmse", rmse)
        
        return r2, rmse
    except Exception as e:
        logging.error("Error in evaluating model: {}".format(e))
        raise e