import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings

from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
import joblib

app = Flask(__name__)

# Load the trained model and scaler
model = load_model("heart_disease_dl_model.h5")
scaler = joblib.load('scaler.save')

@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    if request.method == 'POST':
        try:
            # Collect input features from the form
            features = [
                float(request.form['age']),
                float(request.form['sex']),
                float(request.form['cp']),
                float(request.form['trestbps']),
                float(request.form['chol']),
                float(request.form['fbs']),
                float(request.form['restecg']),
                float(request.form['thalach']),
                float(request.form['exang']),
                float(request.form['oldpeak']),
                float(request.form['slope']),
                float(request.form['ca']),
                float(request.form['thal'])
            ]
            # Scale features
            features = np.array([features])
            features_scaled = scaler.transform(features)

            # Predict the probability
            prediction = model.predict(features_scaled)[0][0]
            print(f"Predicted Probability: {prediction:.2f}")

            # Simple binary classification
            if prediction > 0.5:
                result = f"Predicted risk score: {prediction:.2f} → High risk of heart disease"
            else:
                result = f"Predicted risk score: {prediction:.2f} → Low risk of heart disease"

        except Exception as e:
            result = f"Error: {str(e)}"

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
