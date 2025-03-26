import os
import sys
from dataclasses import dataclass

# Importing required machine learning models
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

# Importing custom modules for exception handling, logging, and utilities
from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj, evaluate_models

@dataclass
class ModelTrainingConfig:
    """
    Configuration class for model training.
    
    This class defines the path where the trained model will be saved.

    Attributes:
        trained_model_file_path (str): Path where the trained model will be stored.
    """
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    """
    A class to handle training, evaluation, and saving of machine learning models.
    
    This class:
    - Splits data into training and test sets.
    - Trains multiple machine learning models.
    - Evaluates the models and selects the best-performing one.
    - Saves the best model for future use.

    Attributes:
        model_trainer_config (ModelTrainingConfig): Configuration containing model storage path.
    """
    
    def __init__(self):
        """
        Initializes the ModelTrainer with the default configuration for storing the trained model.
        """
        self.model_trainer_config = ModelTrainingConfig()

    def initiate_model_trainer(self, train_array, test_array):
        """
        Initiates the model training and evaluation process.

        This method performs the following steps:
        1. Splits the input data (training and testing arrays) into features and target variables.
        2. Defines and trains multiple models using predefined hyperparameters.
        3. Evaluates the models and selects the best-performing one based on R-squared score.
        4. Saves the best model to a specified path.
        5. Returns the R-squared score of the best model on the test dataset.

        Args:
            train_array (numpy array): The training dataset, with features and target variable.
            test_array (numpy array): The testing dataset, with features and target variable.

        Returns:
            float: The R-squared score of the best model on the test dataset.
        
        Raises:
            CustomException: If any error occurs during model training or evaluation.
        """
        try:
            logging.info("Splitting training and testing input data")
            
            # Split the train_array and test_array into features (X) and target variable (y)
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],  # Features for training
                train_array[:, -1],   # Target variable for training
                test_array[:, :-1],   # Features for testing
                test_array[:, -1]     # Target variable for testing
            )

            # Define models to be trained
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Classifier": KNeighborsRegressor(),
                "XGBClassifier": XGBRegressor(),
                "CatBoosting Classifier": CatBoostRegressor(verbose=False),
                "Adaboost Classifier": AdaBoostRegressor()
            }

            # Define hyperparameter grids for each model
            params = {
                "Random Forest": {
                    'n_estimators': [50, 100],
                    'max_depth': [None, 10],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2],
                    'bootstrap': [True]
                },
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse'],
                    'splitter': ['best'],
                    'max_depth': [None, 10],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                },
                "Gradient Boosting": {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.01, 0.05],
                    'max_depth': [3, 5],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2],
                    'subsample': [0.7]
                },
                "Linear Regression": {
                    'fit_intercept': [True]  # Removed 'normalize' parameter
                },
                "K-Neighbors Classifier": {
                    'n_neighbors': [3, 5],
                    'weights': ['uniform'],
                    'algorithm': ['auto'],
                    'p': [2]  # Euclidean distance only
                },
                "XGBClassifier": {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.01, 0.05],
                    'max_depth': [3],
                    'min_child_weight': [1],
                    'gamma': [0],
                    'subsample': [0.7],
                    'colsample_bytree': [0.5]
                },
                "CatBoosting Classifier": {
                    'iterations': [100, 200],
                    'learning_rate': [0.01, 0.05],
                    'depth': [4, 6],
                    'l2_leaf_reg': [1, 3],
                    'border_count': [32]
                },
                "Adaboost Classifier": {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.01, 0.05],
                    'loss': ['linear']
                }
            }

            # Evaluate models and store the performance results
            model_report: dict = evaluate_models(
                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, param=params
            )

            # Identify the best-performing model based on the R-squared score
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            # Ensure the best model meets a minimum performance threshold (R-squared >= 0.6)
            if best_model_score < 0.6:
                raise CustomException("No best model found")

            logging.info(f"Best model found: {best_model_name} with R2 score: {best_model_score}")

            # Save the best model to the specified file path
            save_obj(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Make predictions on the test set using the best model
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            return r2_square

        except Exception as e:
            # Catch and raise any errors during the training and evaluation process
            raise CustomException(e, sys)
