from flask import Flask, render_template, request
import pickle
import numpy as np

# Step 1: Initialize Flask app
app = Flask(__name__)

# Step 2: Load the trained model
model = pickle.load(open("model.pkl", "rb"))


# Step 3: Home route (display form)
@app.route('/')
def home():
    return render_template('index.html')


# Step 4: Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        N = float(request.form['Nitrogen'])
        P = float(request.form['Phosphorus'])
        K = float(request.form['Potassium'])
        temperature = float(request.form['Temperature'])
        humidity = float(request.form['Humidity'])
        ph = float(request.form['pH'])
        rainfall = float(request.form['Rainfall'])

        # Create array for prediction
        features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

        # Predict crop
        prediction = model.predict(features)[0]

        return render_template('index.html', prediction_text=f'Recommended Crop: {prediction}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {e}')


# Step 5: Run app
if __name__ == "__main__":
    app.run(debug=True)
