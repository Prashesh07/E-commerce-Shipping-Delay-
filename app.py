import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from typing import Literal
from pydantic import BaseModel, Field

app = FastAPI()
pipeline = joblib.load("xgboost_shipping_delay_model.pkl")
threshold = joblib.load("model_threshold.pkl")

class ShipmentInput(BaseModel):
    Warehouse_block: Literal['A', 'B', 'C', 'D', 'F']
    Mode_of_Shipment: Literal['Flight', 'Ship', 'Road']
    Customer_care_calls: int 
    Customer_rating: int = Field(ge=1, le=5)
    Cost_of_the_Product: int = Field(ge=1)
    Prior_purchases: int 
    Product_importance: Literal['low', 'medium', 'high']
    Discount_offered: int = Field(lt=100)
    Weight_in_gms: int = Field(ge=1)
    Gender: Literal['M', 'F']

@app.post("/predict")
def predict(data: ShipmentInput):
    df = pd.DataFrame([data.model_dump()])
    prob = pipeline.predict_proba(df)[:, 1][0]
    label = "Delayed" if prob >= threshold else "On Time"
    return {"prediction": label, "probability of being late": round(float(prob), 4)}
