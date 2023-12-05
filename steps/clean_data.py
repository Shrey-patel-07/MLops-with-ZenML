import logging
import pandas as pd

from src.data_cleaning import DataCleaning, DataDivideStrategy, DataPreProcessingStrategy

from typing_extensions import Annotated
from typing import Tuple

from zenml import step

@step
def clean_df(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "Y_train"],
    Annotated[pd.Series, "Y_test"],
]:
    """
    Cleans the data and divides it into train and test sets.
    
    Args:
        df: raw data
        
    Returns: 
        X_train: Training Data
        X_test: Testing Data
        Y_train: Training Labels
        Y_test: Testing Labels
    """
    try:
        process_strategy = DataPreProcessingStrategy()
        data_cleaning = DataCleaning(df, process_strategy)
        processed_data = data_cleaning.handle_data()
        
        divide_strategy = DataDivideStrategy()
        data_cleaning = DataCleaning(processed_data, divide_strategy)
        X_train, X_test, Y_train, Y_test = data_cleaning.handle_data()
        logging.info("Data cleaning Completed.")
        return X_train, X_test, Y_train, Y_test
    except Exception as e:
        logging.error("Error in cleaning data: {}".format(e))
        raise e