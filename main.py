from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
import sklearn

app=FastAPI()

class PredictItem(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.get('/')
async def starter():
    return 'Hello' 

@app.post('/predict')
async def prediction(item: PredictItem):
    df = pd.DataFrame([item.dict().values()], columns=item.dict().keys())
    yhat = model.predict(df)
    return {'prediction': int(yhat)}