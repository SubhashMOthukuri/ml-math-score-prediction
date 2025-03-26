import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    """
    This class is responsible for loading the pre-trained model and preprocessor,
    transforming the input data using the preprocessor, and making predictions.

    Methods:
        predict(features): Takes input features, scales them using the preprocessor, 
                           and makes predictions using the trained model.
    """
    
    def __init__(self):
        """
        Initializes the PredictPipeline class.
        """
        pass

    def predict(self, features):
        """
        Predicts the target variable using the pre-trained model and preprocessor.

        Args:
            features (DataFrame or array-like): The input features to be used for prediction.

        Returns:
            preds (numpy array): The predicted values based on the input features.

        Raises:
            CustomException: If there is an issue with loading the model or preprocessor,
                             scaling the data, or making predictions.
        """
        try:
            model_path = 'artifacts/model.pkl'  # Path to the trained model
            preprocessor_path = "artifacts/preprocessor.pkl"  # Path to the preprocessor

            # Loading the pre-trained model and preprocessor
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            # Scaling the input features using the preprocessor
            data_scaled = preprocessor.transform(features)

            # Making predictions using the trained model
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            # Catching any exceptions and raising a custom exception
            raise CustomException(e, sys)


class CustomData:
    """
    This class represents the custom input data that will be used for prediction.

    Attributes:
        gender (str): Gender of the student.
        race_ethnicity (str): Race/ethnicity of the student.
        parental_level_of_education (str): Parental level of education.
        lunch (str): Type of lunch the student receives.
        test_preparation_course (str): Whether the student completed a test preparation course.
        reading_score (int): The reading score of the student.
        writing_score (int): The writing score of the student.

    Methods:
        get_data_as_dataframe(): Converts the custom input data into a pandas DataFrame.
    """
    
    def __init__(self, 
                 gender: str, 
                 race_ethnicity: str,
                 parental_level_of_education: str,
                 lunch: str,
                 test_preparation_course: str,
                 reading_score: int,
                 writing_score: int):
        """
        Initializes the CustomData object with user-provided values.

        Args:
            gender (str): Gender of the student.
            race_ethnicity (str): Race/ethnicity of the student.
            parental_level_of_education (str): Parental level of education.
            lunch (str): Type of lunch the student receives.
            test_preparation_course (str): Whether the student completed a test preparation course.
            reading_score (int): The reading score of the student.
            writing_score (int): The writing score of the student.
        """
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_dataframe(self):
        """
        Converts the custom input data into a pandas DataFrame.

        Returns:
            pandas.DataFrame: A DataFrame containing the input data as a single row.
        
        Raises:
            CustomException: If there is an error while creating the DataFrame.
        """
        try:
            # Creating a dictionary with input data
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score]
            }

            # Converting the dictionary into a pandas DataFrame
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            # Catching any exceptions and raising a custom exception
            raise CustomException(e, sys)
