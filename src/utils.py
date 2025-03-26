import os
import sys
import numpy as np
import pandas as pd
import dill

from src.exception import CustomException
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_obj(file_path, obj):
    """
    Save a Python object to a specified file path using dill.

    Args:
    - file_path (str): The path where the object will be saved.
    - obj (object): The Python object to be saved.

    Raises:
    - CustomException: If there is an error in saving the object.
    """
    try:
        # Extract the directory path from the file path
        dir_path = os.path.dirname(file_path)

        # Create the directory if it doesn't exist
        os.makedirs(dir_path, exist_ok=True)

        # Open the file in write-binary mode and save the object using dill
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        # Raise a custom exception if an error occurs during the process
        raise CustomException(e, sys)

def evaluate_models(X_train, X_test, y_train, y_test, models, param):
    """
    Evaluate the performance of multiple models using GridSearchCV to find the best hyperparameters and 
    calculate the R2 score on both training and testing data.

    Args:
    - X_train (np.ndarray or pd.DataFrame): Training input features.
    - X_test (np.ndarray or pd.DataFrame): Testing input features.
    - y_train (np.ndarray or pd.Series): Training target values.
    - y_test (np.ndarray or pd.Series): Testing target values.
    - models (dict): A dictionary containing machine learning models to evaluate.
                     Keys are model names, values are model instances.
    - param (dict): A dictionary containing hyperparameter grids for each model.
                    Keys match model names, values are dictionaries of hyperparameters.

    Returns:
    - dict: A dictionary with model names as keys and the corresponding R2 scores on the test set as values.

    Raises:
    - CustomException: If there is an error during the evaluation process.
    """
    try:
        # Initialize an empty dictionary to store the results
        report = {}

        # Iterate over the models and hyperparameters
        for i in range(len(list(models))):
            # Get the current model and its corresponding hyperparameters
            model_name = list(models.keys())[i]
            model = models[model_name]
            param_grid = param[model_name]

            # Use GridSearchCV to find the best hyperparameters using cross-validation
            gs = GridSearchCV(model, param_grid, cv=3)
            gs.fit(X_train, y_train)

            # Set the best hyperparameters to the model
            model.set_params(**gs.best_params_)

            # Fit the model on the training data
            model.fit(X_train, y_train)

            # Make predictions on the training and testing data
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Calculate the R2 score for both training and testing predictions
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            # Store the test score in the report dictionary
            report[model_name] = test_model_score
        
        # Return the model evaluation report
        return report

    except Exception as e:
        # Raise a custom exception if an error occurs during model evaluation
        raise CustomException(e, sys)

def load_object(file_path):
    """
    Load a Python object from a specified file path using dill.

    Args:
    - file_path (str): The path of the file containing the saved object.

    Returns:
    - object: The deserialized Python object.

    Raises:
    - CustomException: If an error occurs while loading the object.
    """
    try:
        # Open the file in read-binary mode and load the object using dill
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
        
    except Exception as e: 
        # Raise a custom exception if an error occurs while loading
        raise CustomException(e, sys)

