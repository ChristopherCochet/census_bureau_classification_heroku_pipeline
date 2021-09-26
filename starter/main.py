import dill as pickle
import pandas as pd

from fastapi import FastAPI, HTTPException, Request, Body
from fastapi.templating import Jinja2Templates

# Import Union since our Item object will have tags that can be
# strings or a list.
# from typing import Union

# BaseModel from Pydantic is used to define data objects.
from pydantic import BaseModel, Field

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
template_dir = "starter/templates/"
feature_encoding_file = model_dir + "census_feature_encoding.pkl"
census_model_file = model_dir + "census_model.pkl"

# html templates
templates = Jinja2Templates(template_dir)

# Declare the data object with its components and their type.
class census_data(BaseModel):
    age: int = Field(..., example=39)
    workclass: str = Field(..., example="State-gov")
    fnlgt: int = Field(..., example=77516)
    education: str = Field(..., example="Bachelors")
    education_num: int = Field(..., example=13, alias="education-num")
    marital_status: str = Field(..., example="Never-married", alias="marital-status")
    occupation: str = Field(..., example="Adm-clerical")
    relationship: str = Field(..., example="Not-in-family")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., example=2174, alias="capital-gain")
    capital_loss: int = Field(..., example=0, alias="capital-loss")
    hours_per_week: int = Field(..., example=40, alias="hours-per-week")
    native_country: str = Field(..., example="United-States", alias="native-country")


# declare fastapi app
census_app = FastAPI()

# Load model artifacts upon startup of the application
@census_app.on_event("startup")
async def startup_event():
    global census_model, ct

    # load data encoder
    with open(feature_encoding_file, "rb") as file:
        ct = pickle.load(file)
        print("census_app - loaded {}".format(feature_encoding_file))

    # load model
    census_model = pickle.load(open(census_model_file, "rb"))
    print("census_app - loaded {}".format(census_model_file))


# GET must be on the root domain and give a greeting
@census_app.get("/")
async def root(request: Request):
    template = templates.TemplateResponse("home.html", {"request": request},)
    return template


# https://github.com/bodywork-ml/bodywork-scikit-fastapi-project
# POST on a different path that does model inference
@census_app.post("/predict")
async def get_prediction(payload: census_data):

    # pdb.set_trace()
    print(payload)
    # Convert input data into a dictionary and then pandas dataframe
    census_data_df = pd.DataFrame.from_dict([payload.dict(by_alias=True)])

    # process post census data
    encoded_census_df = one_hot_encode_feature_df(census_data_df, ct)

    # generate predictions
    preds = inference(census_model, encoded_census_df)
    if not preds:
        raise HTTPException(status_code=400, detail="Model not found.")

    results = {"predict": f"Predicts {preds} for {payload.dict()}"}
    return results
