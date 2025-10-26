# 🌾 Crop Prediction Web App

A **Machine Learning-powered Flask web application** that predicts the most suitable crop based on soil nutrients, temperature, humidity, pH, and rainfall.

---

## 🚀 Features

- Predicts the best crop to grow using a trained **Random Forest model**  
- Simple, elegant, and responsive **frontend UI**
- Flask backend integrated with a trained ML model (`.pkl` file)
- User inputs for **Nitrogen, Phosphorus, Potassium, Temperature, Humidity, pH, and Rainfall**
- Easy deployment on **Vercel**, **Render**, or **Heroku**

---

## 🧠 Machine Learning Model

- **Algorithm:** `RandomForestClassifier`  
- **Accuracy:** ~98% (depending on dataset split)

**Model training snippet:**
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle

data = pd.read_csv("Crop_recommendation.csv")
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

pickle.dump(model, open('model.pkl', 'wb'))
