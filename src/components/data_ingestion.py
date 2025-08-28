import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            df = pd.read_csv(os.path.join('notebook', 'data', 'stud.csv'))
            logging.info("Dataset read as dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save raw CSV
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Raw data saved successfully")

            # Split train/test
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Train/test CSVs saved successfully")

            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        # Step 1: Data ingestion
        ingestion_obj = DataIngestion()
        train_path, test_path = ingestion_obj.initiate_data_ingestion()

        # Step 2: Data transformation
        data_transformation = DataTransformation()
        train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(
            train_path, test_path
        )

        print("Data ingestion and transformation completed successfully.")
        print(f"Preprocessor saved at: {preprocessor_path}")
        modeltrainer= ModelTrainer()
        print(modeltrainer.initiate_model_trainer(train_arr,test_arr))

    except Exception as e:
        logging.error(f"Error in main execution: {e}")
        raise CustomException(e, sys)
