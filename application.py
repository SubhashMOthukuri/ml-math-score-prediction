from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.prediction_pipeline import CustomData, PredictPipeline

# Correct Flask initialization
application = Flask(__name__)  # Use 'Flask' not 'flsk'

# Home Page route
@application.route('/')
def index():
    return render_template('index.html')  # Intro page

# Form page route (home.html)
@application.route('/home', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Retrieve the form data
        reading_score = int(request.form['reading_score'])
        writing_score = int(request.form['writing_score'])

        # You can insert your prediction logic here
        predicted_math_score = (reading_score + writing_score) / 2
        
        return render_template('output.html', results=predicted_math_score)
    
    return render_template('home.html')  # Display the form for data entry

# Predict datapoint route (this handles the form submission and prediction)
@application.route('/predict_datapoint', methods=['POST'])
def predict_datapoint():
    if request.method == 'POST':
        reading_score = request.form['reading_score']
        writing_score = request.form['writing_score']
        
        # Insert prediction logic here
        
        predicted_math_score = (int(reading_score) + int(writing_score)) / 2
        return render_template('output.html', results=predicted_math_score)

if __name__ == "__main__":
    application.run(host="0.0.0.0", debug=True)

