import logging
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression

class Model(ABC):
    
    """
        Abstract class for Model.
    """
    @abstractmethod
    def train(self, X_train, y_train):
        """
            Train the model on the ingested data.

        Args:
            X_train: Training Data
            y_train: Training Labels
        """
        pass
    
class LinearRegressionModel(Model):
    """
    Linear Model
    """
    def train(self, X_train, y_train, **kwargs):
        """
            Train the model on the ingested data.

        Args:
            X_train: Training Data
            y_train: Training Labels
        """
        try:
            model = LinearRegression()
            model.fit(X_train, y_train)
            logging.info("Model Training Completed.")
            return model
        except Exception as e:
            logging.error("Error in training model: {}".format(e))
            raise e