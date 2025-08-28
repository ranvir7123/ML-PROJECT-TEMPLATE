import os
import sys
import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

def main():
    try:
        logging.info("Starting ML pipeline...")
        
        # Step 1: Data ingestion
        ingestion_obj = DataIngestion()
        train_path, test_path = ingestion_obj.initiate_data_ingestion()
        logging.info("Data ingestion completed")

        # Step 2: Data transformation
        data_transformation = DataTransformation()
        train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(
            train_path, test_path
        )
        logging.info("Data transformation completed")

        # Step 3: Model training
        model_trainer = ModelTrainer()
        r2_score = model_trainer.initiate_model_trainer(train_arr, test_arr)
        
        logging.info(f"Pipeline completed successfully! Final R² score: {r2_score}")
        print(f"\n✅ Pipeline completed! Final R² score: {r2_score:.4f}")

    except Exception as e:
        logging.error(f"Error in main pipeline: {e}")
        print(f"❌ Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()