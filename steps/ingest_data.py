import logging
import pandas as pd
from zenml import step


class IngestData:

    """Ingesting data from the data_path.
    """

    def __init__(self, data_path: str):
        """
        Args:
            data_path (str): path to the data.
        """
        self.data_path = data_path

    def get_data(self):
        """
        Ingesting data from the data_path.
        """
        logging.info(f'Getting data from {self.data_path}')
        return pd.read_csv(self.data_path)


@step
def ingest_df(data_path: str) -> pd.DataFrame:
    """
    Ingesting data from a data_path.

    Args:
        data_path (str): path to data

    Returns:
        pd.DataFrame: ingested data
    """
    try:
        ingested_data = IngestData(data_path).get_data()
        return ingested_data
    except Exception as e:
        logging.error(f'Error ingesting data: {e}')
        raise e
