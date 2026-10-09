import os 
import sys
from src.exception import CustomException
from src.logger import logging 
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

#GIVING THE PATHS FOR TRAIN, TEST AND RAW DATASET 
@dataclass #DECORATOR IS USED TO CREATE A CLASS THAT WILL HOLD THE CONFIGURATION FOR DATA INGESTION
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")#OUTPUT PATH FOR TRAIN DATASET 
    test_data_path: str = os.path.join("artifacts", "test.csv")#OUTPUT PATH FOR TEST DATASET
    raw_data_path: str = os.path.join("artifacts", "data.csv")#OUTPUT PATH FOR RAW DATASET

class DataIngestion:#GOING WITH INIT FUNCTION TO INITIALIZE THE CLASS AND CREATE AN OBJECT OF THE CONFIGURATION CLASS
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        # CREATING AN OBJECT OF THE CONFIGURATION CLASS TO ACCESS THE PATHS    
        # CALLED THE CONFIGURATION CLASS TO ACCESS THE PATHS FOR TRAIN, TEST AND RAW DATASET
        #CONSISTENTLY USING THE CONFIGURATION CLASS TO ACCESS THE PATHS FOR TRAIN, TEST AND RAW DATASET

#FOR READING THE DATASET FROM THE DATABASE AND SPLITTING IT INTO TRAIN AND TEST SETS
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        #logging is the process of recording events that happen when some software runs. The logging module in Python is used to track events that happen when some software runs. The logging module is part of the standard library, so you don't need to install anything extra to use it.
        try:
            df = pd.read_csv("notebook/data/stud.csv")
            logging.info("Read the dataset as dataframe")
            #os module provides a way of using operating system dependent functionality. The os.path module is submodule of os module which is used for common pathname manipulations. The os.makedirs() method in Python is used to create a directory recursively. It means that all the intermediate-level directories needed to contain the leaf directory will be created. If the target directory already exists, it will not raise an error if exist_ok is set to True.

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            # df to csv() method is used to write the DataFrame to a CSV file. The index=False parameter is used to exclude the index column from the output file. The header=True parameter is used to include the column names in the output file.

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_path,#information required for data transformation and model training is provided by the paths of the train and test datasets
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
            raise CustomException(e, sys)

