import pandas as pd
import sys
from logger import logging
from exception import CustomException

def data_fetcher(file):
    try:
        data=pd.read_excel(file)
        logging.info("Data Ingested Successfully!")
        return data
    except Exception as e:
        logging.info(e)
        raise CustomException(e,sys)



