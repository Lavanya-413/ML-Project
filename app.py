from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

# Load model
with open("DT.pkl", "rb") as f:
    model = pickle.load(f)

# Create FastAPI app
app = FastAPI(
    title="Bank Customer Churn Prediction API",
    description="Predict whether a customer will leave the bank or not",
    version="1.0"
)

# Input schema
class CustomerData(BaseModel):
    customer_id: int
    credit_score: int
    country: int
    gender: int
    age: int
    tenure: int
    balance: float
    products_number: int
    credit_card: int
    active_member: int
    estimated_salary: float

# Home Route
@app.get("/")
def home():
    return {"message": "Bank Customer Churn Prediction API"}

# Prediction Route
@app.post("/predict")
def predict(data: CustomerData):

    features = np.array([[
        data.customer_id,
        data.credit_score,
        data.country,
        data.gender,
        data.age,
        data.tenure,
        data.balance,
        data.products_number,
        data.credit_card,
        data.active_member,
        data.estimated_salary
    ]])

    prediction = model.predict(features)[0]

    result = "Customer Will Leave" if prediction == 1 else "Customer Will Stay"

    return {
        "prediction": int(prediction),
        "result": result
    }