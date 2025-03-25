import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
## Used to create pipelines (first one-hot encoding, standard scaling)
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
## Used for categorical encoding and feature scaling
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.pipeline import Pipeline
from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")


class DataTransformation:
    ## This function is responsible for data transformation 
    ## It removes all null or missing values with median or most frequent values and then combines all data.
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]

            ## Numerical pipeline: Handling missing values and standard scaling
            num_pipeline = Pipeline(
                steps=[  # Fixed the typo "setps" -> "steps"
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            ## Categorical pipeline: Handling missing values, one-hot encoding, and scaling
            cat_pipeline = Pipeline(
                steps=[  # Fixed incorrect parentheses and missing commas
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))  # StandardScaler needs `with_mean=False` for sparse matrices
                ]
            )

            logging.info("Numerical column standard scaling completed.")
            logging.info("Categorical column encoding completed.")

            ## Combining pipelines into a ColumnTransformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)  # Fixed incorrect variable reference
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed.")
            logging.info("Obtaining preprocessing object.")

            preprocessor_obj = self.get_data_transformer_object()

            ## Target column and numerical features
            target_column_name = "math_score"
            numerical_columns = ["writing_score", "reading_score"]

            ## Splitting input and output features for training
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            ## Splitting input and output features for testing
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training and testing dataframes.")

            ## Applying transformations
            input_feature_train_array = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_array = preprocessor_obj.transform(input_feature_test_df)  # Fixed incorrect variable name

            ## Combining transformed input features with target variable
            train_arr = np.c_[input_feature_train_array, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_array, np.array(target_feature_test_df)]  # Fixed incorrect variable name

            logging.info("Saved preprocessing object.")

            ## Saving the preprocessor object
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
            raise CustomException(e, sys)
