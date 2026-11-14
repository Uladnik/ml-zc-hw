import pickle, uvicorn, os
import numpy as np
import pandas as pd
from fastapi import FastAPI
from typing import Optional, Literal
from pydantic import BaseModel, Field, ConfigDict


class Customer(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    customerid: str | None = Field(None, alias="customerID")
    gender: Literal["Female", "Male"] | None = Field(None, alias="gender")
    seniorcitizen: int | str | None = Field(None, ge=0, le=1, alias="SeniorCitizen")
    partner: Literal["Yes", "No"] | None = Field(None, alias="Partner")
    dependents: Literal["Yes", "No"] | None = Field(None, alias="Dependents")

    tenure: int | None = Field(None, ge=0, le=72, alias="tenure")

    phoneservice: Literal["No", "Yes"] | None = Field(None, alias="PhoneService")
    multiplelines: Literal["No phone service", "No", "Yes"] | None = Field(
        None, alias="MultipleLines"
    )

    internetservice: Literal["DSL", "Fiber optic", "No"] | None = Field(
        None, alias="InternetService"
    )
    onlinesecurity: Literal["No", "Yes", "No internet service"] | None = Field(
        None, alias="OnlineSecurity"
    )
    onlinebackup: Literal["No", "Yes", "No internet service"] | None = Field(
        None, alias="OnlineBackup"
    )
    deviceprotection: Literal["No", "Yes", "No internet service"] | None = Field(
        None, alias="DeviceProtection"
    )
    techsupport: Literal["No", "Yes", "No internet service"] | None = Field(
        None, alias="TechSupport"
    )
    streamingtv: Literal["No", "Yes", "No internet service"] | None = Field(
        None, alias="StreamingTV"
    )
    streamingmovies: Literal["No", "Yes", "No internet service"] | None = Field(
        None, alias="StreamingMovies"
    )

    contract: Literal["Month-to-month", "One year", "Two year"] | None = Field(
        None, alias="Contract"
    )
    paperlessbilling: Literal["Yes", "No"] | None = Field(
        None, alias="PaperlessBilling"
    )

    paymentmethod: (
        Literal[
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit customerd (automatic)",
        ]
        | None
    ) = Field(None, alias="PaymentMethod")

    monthlycharges: float | None = Field(
        None, ge=18.25, le=118.75, alias="MonthlyCharges"
    )
    totalcharges: float | str | None = Field(None, alias="TotalCharges")


class PredictResponse(BaseModel):
    prediction: float


script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "model.bin")
app = FastAPI(title="churn-prediction")

with open(model_path, "rb") as f:
    preprocessor, model = pickle.load(f)


def prepare_data(customer: Customer) -> pd.DataFrame:
    data = {
        k.lower().replace(" ", "_"): (
            v.lower().replace(" ", "_") if isinstance(v, str) else v
        )
        for k, v in customer.model_dump().items()
    }
    df = pd.DataFrame([data])
    df.totalcharges = pd.to_numeric(df.totalcharges, errors="coerce")
    df.seniorcitizen = df.seniorcitizen.astype(str)
    return df.fillna(0)


@app.post("/predict")
def predict(customer: Customer) -> PredictResponse:
    df = prepare_data(customer)
    X = preprocessor.transform(df)
    response = float(model.predict_proba(X)[0][1])
    return PredictResponse(prediction=response)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)
