# Predicting Math Scores: End-to-End Machine Learning Project

This project showcases an end-to-end machine learning workflow designed to predict students' math scores based on their reading and writing scores. The solution leverages multiple regression models, a modular pipeline architecture, and a Flask-based web interface for real-time predictions. The project emphasizes best practices in data ingestion, transformation, model training, fine-tuning, and deployment.

## Project Overview

The goal of this project is to build a predictive model that estimates a student's math score using their reading and writing scores as input features. The workflow includes:

- **Data Ingestion**: Loading and preparing the dataset.
- **Data Transformation**: Preprocessing and feature engineering.
- **Model Training**: Evaluating multiple regression algorithms to identify the best performer.
- **Hyperparameter Tuning**: Fine-tuning the selected model for optimal accuracy.
- **Utility Layers**: Reusable components for common tasks and UI templating.
- **Prediction Pipeline**: Integrating the trained model for real-time predictions.
- **Deployment**: Creating a Flask-based web application with a user-friendly interface.

## Features

- Modular pipeline architecture for scalability and maintainability.
- Comparison of various regression models (e.g., Linear Regression, Decision Trees, Random Forest, etc.).
- Fine-tuning of the best-performing model for improved accuracy.
- Real-time predictions via a Flask web application.
- Responsive UI for user interaction.

## Project Structure

├── data/ # Directory for dataset files
├── src/ # Source code directory
│ ├── data_ingestion.py # Data loading and preprocessing
│ ├── data_transformation.py # Feature engineering and scaling
│ ├── model_training.py # Model training and evaluation
│ ├── prediction_pipeline.py # Prediction logic for deployment
│ ├── utils.py # Utility functions for common tasks
│ └── templates/ # HTML templates for Flask UI
├── app.py # Flask application for deployment
├── requirements.txt # Project dependencies
├── README.md # Project documentation (this file)
└── trained_model.pkl # Saved fine-tuned model (generated after training)

## Technologies Used

- **Programming Language**: Python
- **Machine Learning**: Scikit-learn (for regression models and fine-tuning)
- **Web Framework**: Flask (for deployment and UI)
- **Frontend**: HTML/CSS (via Flask templates)
- **Dependencies**: Pandas, NumPy, Joblib (for model saving), etc.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd predicting-math-scores
   ```

**Install Dependencies**:

```bash
pip install -r requirements.txt
```

## Prepare the Dataset:

Place your dataset (e.g., students_data.csv) in the data/ directory.

Ensure it contains at least three columns: reading_score, writing_score, and math_score.

## Usage

1. Train the Model
   Run the training pipeline to evaluate regression models and save the best one:

```bash
python src/model_training.py
```

This script will:

Load data from the data/ directory.

Preprocess it using data_ingestion.py and data_transformation.py.

Train and compare multiple regression models.

Fine-tune the best model and save it as trained_model.pkl.

## 2. Run the Flask Application

Launch the web application for real-time predictions:

```bash
python app.py
```

Open your browser and navigate to http://127.0.0.1:5000.

Enter a student's reading and writing scores to get a predicted math score.

## Pipeline Details

**Data Ingestion Layer (data_ingestion.py)**:
Loads the dataset and performs initial validation/cleaning.

**Data Transformation Layer (data_transformation.py)**:
Handles missing values, feature scaling, and encoding (if applicable).

**Model Training Layer (model_training.py)**:
Trains multiple regression models (e.g., Linear Regression, Random Forest, etc.).

Evaluates performance using metrics like RMSE, R², and MAE.

Fine-tunes the best model using grid search or random search.

Prediction Pipeline Layer (prediction_pipeline.py):
Loads the trained model and processes user inputs for predictions.

**Utility Layer (utils.py)**:
Contains reusable functions (e.g., model saving/loading, logging).

**UI Templates (templates/)**:
HTML files for the Flask web interface.

## Model Selection

The project evaluates multiple regression algorithms, including:

- Linear Regression

- Decision Tree Regressor

- Random Forest Regressor

- Gradient Boosting Regressor (e.g., XGBoost, LightGBM)

- Support Vector Regressor (SVR)

The model with the highest accuracy (based on R² score or similar metric) is selected and fine-tuned.

## Fine-Tuning

Hyperparameters of the best model are optimized using techniques like Grid Search or Randomized Search to maximize predictive performance.

## Deployment

The Flask application (app.py) integrates the trained model and provides a simple UI where users can:

- Input reading and writing scores.

- Receive the predicted math score instantly.

## Flask Web Application

### Home Page Screenshot

Here is the screenshot of the home page of the Flask web application:

![Home Page Screenshot](screenshots/home_page.png)

### Output Screenshot

This is the screenshot showing the output of the prediction when a user inputs their data:

![Output Screenshot](screenshots/output.png)

## Future Improvements

- Add cross-validation for more robust model evaluation.

- Incorporate additional features (e.g., study time, gender) for better predictions.

- Deploy the app to a cloud platform (e.g., Heroku, AWS).

- Enhance the UI with JavaScript for a more interactive experience.
