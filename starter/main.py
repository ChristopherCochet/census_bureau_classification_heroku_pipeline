import dill as pickle
import pandas as pd

from fastapi import FastAPI, HTTPException

# Import Union since our Item object will have tags that can be
# strings or a list.
# from typing import Union

# BaseModel from Pydantic is used to define data objects.
from pydantic import BaseModel

from starter.ml.train_model import one_hot_encode_feature_df, inference

import os

# DVC set-up for Heroku
if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    os.system("dvc config core.hardlink_lock true")
    if os.system("dvc pull -r s3remote") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

# Directory Paths
model_dir = "starter/model/"
feature_encoding_file = model_dir + "census_feature_encoding.pkl"
census_model_file = model_dir + "census_model.pkl"


# Declare the data object with its components and their type.
class census_data(BaseModel):
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

    class Config:
        schema_extra = {
            "example": {
                "age": 39,
                "workclass": "State-gov",
                "fnlgt": 77516,
                "education": "Bachelors",
                "education_num": 13,
                "marital_status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "capital_gain": 2174,
                "capital_loss": 0,
                "hours_per_week": 40,
                "native_country": "United-States",
            }
        }


census_app = FastAPI()

# GET must be on the root domain and give a greeting
@census_app.get("/")
async def root():
    return {"message": "Census Bureau Salary Prediction App"}


# https://github.com/bodywork-ml/bodywork-scikit-fastapi-project
# POST on a different path that does model inference
@census_app.post("/predict")
async def get_prediction(payload: census_data):

    # pdb.set_trace()
    print(payload)
    # Convert input data into a dictionary and then pandas dataframe
    census_data_df = pd.DataFrame.from_dict([payload.dict()])
    census_data_df.columns = census_data_df.columns.str.replace("_", "-")

    # load data encoder
    with open(feature_encoding_file, "rb") as file:
        ct = pickle.load(file)

    # process post census data
    encoded_census_df = one_hot_encode_feature_df(census_data_df, ct)

    # load model
    census_model = pickle.load(open(census_model_file, "rb"))

    # generate predictions
    preds = inference(census_model, encoded_census_df)
    if not preds:
        raise HTTPException(status_code=400, detail="Model not found.")

    results = {"predict": f"Predicts {preds} for {payload.dict()}"}
    return results
