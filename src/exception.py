import sys
from src.logger import logging

def error_message_details(error, error_detail):
    """
    Extracts error details, such as file name, line number, and the error message.

    This function is used to get detailed information about where the error occurred 
    in the code, including the file name and line number, along with the error message.

    Args:
        error (Exception): The exception object that contains the error message.
        error_detail (traceback): The traceback object which contains detailed information 
                                  about where the exception occurred.

    Returns:
        str: A formatted string containing the file name, line number, and the error message.
    """
    # Extracting the traceback information
    _, _, exc_tb = error_detail.exc_info()
    
    # Getting the file name where the error occurred
    file_name = exc_tb.tb_frame.f_code.co_filename
    
    # Formatting the error message with file name, line number, and error message
    error_message = "Error occurred in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )
    return error_message

class CustomException(Exception):
    """
    A custom exception class to handle exceptions more effectively with detailed error messages.

    This class extends the built-in Exception class to provide custom error messages 
    with additional details like the file name, line number, and error message.

    Attributes:
        error_message (str): A detailed error message that includes the file name, 
                              line number, and the error message.
    
    Methods:
        __str__(): Returns the error message when the exception is converted to a string.
    """
    
    def __init__(self, error, error_detail):
        """
        Initializes the CustomException class with the given error and error details.

        Args:
            error (Exception): The exception that was raised.
            error_detail (traceback): The traceback information for the error.

        This method also formats the error message using the `error_message_details` function.
        """
        super().__init__(str(error))  # Pass the error message to the base Exception class
        self.error_message = error_message_details(error, error_detail)

    def __str__(self):
        """
        Returns the error message when the exception is converted to a string.

        Returns:
            str: The formatted error message.
        """
        return self.error_message
