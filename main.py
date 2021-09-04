import os

from fastapi import FastAPI, Body
from pydantic import BaseModel
from statistics import mode
import pandas as pd
import numpy as np
import pickle
import uvicorn

from train import CAT_FEATURES
from ml.model import inference
from ml.data import process_data

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")


app = FastAPI()

class Data(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str    



@app.get("/")
async def greeting():
    return {"greeting": "Hello World!"}


@app.post('/inference')
async def make_prediction(
    data: Data = Body(...,
        example={
            "age": 36,
            "workclass": "State-gov",
            "fnlgt": 189778,
            "education": "Masters",
            "education_num": 12,
            "marital_status": "MarriedMarried-civ-spouse",
            "occupation": "Adm-clericalTech-support",
            "relationship": "Husband",
            "race": "WhiteAsian-Pac-Islander",
            "sex": "Male",
            "capital_gain": 1077,
            "capital_loss": 0,
            "hours_per_week": 40,
            "native_country": "China"
        }

    )
):


    df = pd.DataFrame.from_dict({
            "age": [data.age],
            "workclass": [data.workclass],
            "fnlgt": [data.fnlgt],
            "education": [data.education],
            "education_num": [data.education_num],
            "marital-status": [data.marital_status], 
            "occupation": [data.occupation],
            "relationship": [data.relationship], 
            "race": [data.race],
            "sex": [data.sex],
            "capital-gain": [data.capital_gain],
            "capital-loss": [data.capital_loss],
            "hours-per-week": [data.hours_per_week],
            "native-country": [data.native_country]}
    )
    
    with open(f'./model/encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
        
    with open(f'./model/lr.pkl', 'rb') as f: 
        model = pickle.load(f) 
    
    with open(f'./model/lb.pkl','rb') as f:
        lb = pickle.load(f)


    X_test, _, _, _ = process_data(df, CAT_FEATURES, label=None, training=False, encoder=encoder)

    pred = inference(model, X_test)
    results = {"salary":  lb.inverse_transform(np.array(pred))[0]}
    return results

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=5000, log_level="info")