import pickle, uvicorn, os
import pandas as pd
from fastapi import FastAPI
from typing import Dict, Any, Literal
from pydantic import BaseModel, Field, ConfigDict, field_validator


class Customer(BaseModel):
    model_config = ConfigDict(extra="ignore")
    gender: Literal["male", "female"]
    seniorcitizen: Literal[0, 1]
    partner: Literal["yes", "no"]
    dependents: Literal["yes", "no"]
    phoneservice: Literal["yes", "no"]
    multiplelines: Literal["no", "yes", "no_phone_service"]
    internetservice: Literal["dsl", "fiber_optic", "no"]
    onlinesecurity: Literal["no", "yes", "no_internet_service"]
    onlinebackup: Literal["no", "yes", "no_internet_service"]
    deviceprotection: Literal["no", "yes", "no_internet_service"]
    techsupport: Literal["no", "yes", "no_internet_service"]
    streamingtv: Literal["no", "yes", "no_internet_service"]
    streamingmovies: Literal["no", "yes", "no_internet_service"]
    contract: Literal["month-to-month", "one_year", "two_year"]
    paperlessbilling: Literal["yes", "no"]
    paymentmethod: Literal[
        "electronic_check",
        "mailed_check",
        "bank_transfer_(automatic)",
        "credit_card_(automatic)",
    ]
    tenure: int = Field(..., ge=0)
    monthlycharges: float = Field(..., ge=0.0)
    totalcharges: float = Field(..., ge=0.0)

    @field_validator("*", mode="before")
    @classmethod
    def normalize(cls, v):
        if isinstance(v, str):
            return v.lower().replace(" ", "_")
        return v


class PredictResponse(BaseModel):
    churn_probability: float
    churn: bool


script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "model.bin")
app = FastAPI(title="customer-churn-prediction")

with open(model_path, "rb") as f_in:
    pipeline = pickle.load(f_in)


def prepare_data(customer: Customer) -> pd.DataFrame:
    data = {
        k.lower().replace(" ", "_"): 
        (v.lower().replace(" ", "_") if isinstance(v, str) else v)
        for k, v in customer.model_dump().items()
    }
    df = pd.DataFrame([data])
    df["totalcharges"] = pd.to_numeric(df["totalcharges"], errors="coerce")
    df["totalcharges"] = df["totalcharges"].fillna(0)
    df["seniorcitizen"] = df["seniorcitizen"].astype(str)
    return df


@app.post("/predict")
def predict(customer: Customer) -> PredictResponse:
    df = prepare_data(customer)
    prob = float(pipeline.predict_proba(df)[0][1])
    return PredictResponse(churn_probability=prob, churn=bool(prob >= 0.5))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)
