from fastapi import FastAPI
import joblib
from pydantic import BaseModel
import numpy as np


app = FastAPI()
cluster_model = joblib.load("kmeans_model.pkl")
scaler_model = joblib.load("mallscaler.pkl")

personal_mapping = {
   0: "Standard(Average income, average spend)",
   1: "Target(High income, high spend)",
   2: "Reckless(Low income, high spend)",
   3: "Careful(High income, Low spend)",
   4: "Sensible(Low income, Low spend)"
}
class CustomerData(BaseModel):
  annual_income: float
  spending_score: float

@app.post("/predict")
def predict_customer_segment(data: CustomerData):
  income = data.annual_income 
  if income > 1000 :
    income = income/1000
  features = np.array([
    [income, data.spending_score]
  ])

  scaled_features = scaler_model.transform(features)

  cluster_id = int(cluster_model.predict(scaled_features)[0])
  personal = personal_mapping.get(cluster_id, "unknown_segment")
  return {
    "annual income": income,
    "spendind score": data.spending_score,
    "cluster id": cluster_id,
    "personel": personal
  }
  # print(personal_mapping[cluster_id])