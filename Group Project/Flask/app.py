''' 
    Import the NumPy library for numerical computations
    Import the Pandas library for data manipulation and analysis
    Import Flask for web application development
    Import the Pickle module for object serialization
'''

import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
import pickle

# Create an instance of the Flask class

app = Flask(__name__)

# Load a pre-trained model using Pickle

model = pd.read_pickle('./Flask/air_passengers.pkl')

# Render the 'home.html' template when the root URL is accessed

@app.route('/')
def home():
    return render_template('home.html')

''' Check if the HTTP request method is POST
    Get the value of the 'Date' form field
    Create a dictionary with 'ds' as the key and 'dates' as the value
    Create a DataFrame from the dictionary
    Make predictions using the pre-trained model
    Print the predictions
    Extract the predicted value from the DataFrame
    Print the predicted value
    Render the 'home.html' template with the predicted value
    Render the 'home.html' template if the HTTP request method is not POST
'''

@app.route('/predict', methods=['POST'])
def y_predict():
    if request.method == "POST":
        dates = request.form["Date"]
        name = {"ds": [dates]}
        retain = dates
        dates = pd.DataFrame(name)
        prediction = model.predict(dates)
        print(prediction)
        output = round(prediction.iloc[0, 15])
        print(output)
        return render_template('home.html',
                               prediction_text="Commuters Inflow on {} is {}.".format(retain,output))
    return render_template('home.html')

# Run the Flask application in debug mode

if __name__ == "__main__":
    app.run(debug=True)
