## ==============================================
## DATA TRANSFORMATION MODULE FOR MACHINE LEARNING PIPELINE
## ==============================================
## Author: Subhash Mothukuru
## Created On: 05/10/2024
## Description:
##     - This script is responsible for transforming raw datasets by handling missing values,
##       encoding categorical features, and scaling numerical features.
##     - It constructs preprocessing pipelines for numerical and categorical data,
##       applies transformations, and saves the preprocessing object.
##     - This module is typically used to ensure data consistency before feeding it into
##       machine learning models.
## ==============================================

import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj

@dataclass
class DataTransformationConfig:
    """
    Configuration class for data transformation.

    This class defines the file path where the preprocessor object will be stored.

    Attributes:
        preprocessor_obj_file_path (str): Path where the preprocessing pipeline object will be stored.
    """
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")


class DataTransformation:
    """
    Data transformation class to preprocess datasets for machine learning.

    This class applies data cleaning, categorical encoding, and feature scaling
    using predefined transformation pipelines.
    """
    
    def __init__(self):
        """
        Initialize DataTransformation with default configurations.
        
        This constructor sets up the configuration for data transformation and 
        prepares the system for preprocessing the datasets.
        """
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer_object(self):
        """
        Creates and returns a preprocessing object for data transformation.

        The preprocessing steps performed are:
        1. Handles missing values for numerical columns using median imputation.
        2. Standardizes numerical features using StandardScaler.
        3. Handles missing values for categorical columns using most frequent imputation.
        4. Encodes categorical features using OneHotEncoder.
        5. Scales encoded categorical features.
        6. Combines numerical and categorical transformations using ColumnTransformer.

        Returns:
            ColumnTransformer: Preprocessing pipeline for numerical and categorical features.

        Raises:
            CustomException: If any error occurs while creating the transformer object.
        """
        try:
            # Define the columns for numerical and categorical features
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender", "race_ethnicity", "parental_level_of_education", "lunch", "test_preparation_course"
            ]

            # Define numerical pipeline for imputing missing values and scaling
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),  # Handle missing values using median
                    ("scaler", StandardScaler())  # Standardize numerical features
                ]
            )

            # Define categorical pipeline for imputing missing values and encoding
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),  # Handle missing values using most frequent value
                    ("one_hot_encoder", OneHotEncoder()),  # One hot encoding for categorical features
                    ("scaler", StandardScaler(with_mean=False))  # Scale encoded categorical features
                ]
            )

            logging.info("Numerical column standard scaling completed.")
            logging.info("Categorical column encoding completed.")

            # Combine numerical and categorical transformations using ColumnTransformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )
            
            return preprocessor
        except Exception as e:
            # Raise a custom exception if an error occurs during transformer creation
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """
        Reads train and test datasets, applies transformations, and saves the preprocessing object.

        Steps:
        1. Loads training and testing datasets from CSV files.
        2. Extracts input features and target variable.
        3. Applies preprocessing pipelines to the datasets.
        4. Saves the preprocessing object for future inference.

        Args:
            train_path (str): Path to the training dataset.
            test_path (str): Path to the test dataset.

        Returns:
            tuple: Transformed training data, transformed test data, path to the preprocessor object.

        Raises:
            CustomException: If any error occurs during data transformation.
        """
        try:
            # Load training and testing data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed.")
            logging.info("Obtaining preprocessing object.")

            # Get the data transformation pipeline
            preprocessor_obj = self.get_data_transformer_object()

            # Define the target column and numerical features
            target_column_name = "math_score"
            numerical_columns = ["writing_score", "reading_score"]

            # Separate input features and target variables
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training and testing dataframes.")

            # Apply the preprocessing pipeline to the data
            input_feature_train_array = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_array = preprocessor_obj.transform(input_feature_test_df)
            
            # Combine the processed features with the target variable
            train_arr = np.c_[input_feature_train_array, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_array, np.array(target_feature_test_df)]
            
            logging.info("Saved preprocessing object.")

            # Save the preprocessor object for future use
            save_obj(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            # Handle any exception that occurs during data transformation
            raise CustomException(e, sys)
