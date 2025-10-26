# model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Step 1: Load dataset
data = pd.read_csv("Crop_recommendation.csv")

# Step 2: Split features and target
x = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Step 3: Train-Test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Step 4: Train model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Step 5: Check accuracy
accuracy = model.score(x_test, y_test)
print("Accuracy:", accuracy)

# Step 6: Save model using pickle
pickle.dump(model, open("model.pkl", "wb"))
print("Model saved successfully as crop_model.pkl!")
