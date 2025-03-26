## ==============================================
## DATA INGESTION MODULE FOR MACHINE LEARNING PIPELINE
## ==============================================
## Author: Subhash Mothukuru
## Created On: May 10, 2024
## Description:
##     - This script is responsible for reading datasets from various sources
##       such as CSV files, databases, and APIs.
##     - It performs a train-test split and saves the processed datasets
##       into an "artifacts" directory for further processing.
##     - This module is typically used in a data science workflow to ensure
##       data is prepared before training machine learning models.
## ==============================================

import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging

from src.components.data_tranformation import DataTransformation
from src.components.data_tranformation import DataTransformationConfig

from src.components.model_training import ModelTrainingConfig
from src.components.model_training import ModelTrainer


@dataclass
class DataIngestionConfig:
    """
    Configuration class for data ingestion.

    This class defines paths where raw, train, and test data will be stored.
    These paths are used to save preprocessed data as artifacts for further 
    machine learning model training.

    Attributes:
        train_data_path (str): Path where the training dataset will be stored.
        test_data_path (str): Path where the test dataset will be stored.
        raw_data_path (str): Path where the raw dataset will be stored.
    """
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")


class DataIngestion:
    """
    Data ingestion class that reads a dataset, splits it into training and testing sets, 
    and saves the resulting files to predefined locations.

    This class can be extended to read data from databases, APIs, or cloud storage.
    """

    def __init__(self):
        """
        Initialize DataIngestion with default configurations.

        This constructor creates an instance of the DataIngestionConfig class that 
        defines the paths where the raw, training, and testing data will be stored.
        """
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        """
        Reads data from a CSV file, performs a train-test split, and saves the datasets.

        This method performs the following steps:
        1. Read data from a specified source (CSV in this case).
        2. Ensure the "artifacts" directory exists.
        3. Save the raw dataset for reference.
        4. Split the data into training (80%) and testing (20%) sets.
        5. Save the training and testing datasets separately.

        Returns:
            tuple: Paths of train and test datasets.
        
        Raises:
            CustomException: If any error occurs during data ingestion.
        """
        logging.info("Entered the data ingestion method.")

        try:
            # Step 1: Read the dataset
            dataset_path = r'notebook/data/stud.csv'  # Modify this to read from a database if needed
            df = pd.read_csv(dataset_path)  # Read CSV file into a DataFrame
            logging.info("Dataset successfully loaded into a pandas DataFrame.")

            # Step 2: Create directory for artifacts if it does not exist
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Step 3: Save the raw dataset
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info(f"Raw data saved at {self.ingestion_config.raw_data_path}")

            # Step 4: Perform Train-Test Split
            logging.info("Splitting dataset into training and testing sets...")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            logging.info("Train-test split completed.")

            # Step 5: Save train and test datasets
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info(f"Training data saved at {self.ingestion_config.train_data_path}")
            logging.info(f"Testing data saved at {self.ingestion_config.test_data_path}")

            logging.info("Data ingestion process completed successfully.")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.error("Error during data ingestion", exc_info=True)
            raise CustomException(e, sys)


if __name__ == "__main__":
    """
    Entry point for executing the data ingestion process.
    
    This block of code is executed when the script is run directly. It initializes 
    the DataIngestion class, calls the data ingestion process, and then passes 
    the resulting train and test datasets to the DataTransformation and ModelTrainer 
    classes for further processing.
    """
    
    # Data Ingestion Process
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    # Data Transformation Process
    data_transformation = DataTransformation()
    train_array, test_array, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    # Model Training Process
    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_array, test_array))
