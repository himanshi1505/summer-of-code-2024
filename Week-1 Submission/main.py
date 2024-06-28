from fastapi import FastAPI
import uvicorn
import pandas as pd
import nest_asyncio
import joblib  # For loading a scikit-learn model saved as a pickle file

# Create FastAPI instance
app = FastAPI()

# Define the root endpoint
@app.get('/')
def main():
    return {'message': 'Welcome to API!'}

# Load the model
model_path = 'model.pkl'  # Update this path to where your model is saved
clf = joblib.load(model_path)

# Endpoint to receive data and make predictions
@app.post('/predict')
def predict(data: dict):
    # Convert the data to a DataFrame
    test_data = pd.DataFrame([data])
    # Make the prediction
    fraudpred = clf.predict(test_data)[0]
    # Return the result
    return {'Fraud_prediction': int(fraudpred)}

# Use nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Run uvicorn server
uvicorn.run("main:app", host='127.0.0.1', port=8000, reload=True)
