import pickle, uvicorn, os
import numpy as np
import pandas as pd
from fastapi import FastAPI
from typing import Optional, Literal
from pydantic import BaseModel, Field, ConfigDict


class CarPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    model_year: int | None = Field(None, ge=1945, le=2027, alias="model_year")
    num_doors: int | None = Field(None, ge=2, le=8, alias="num_doors")
    engine_displacement: int | None = Field(None, ge=1000, le=5000, alias="engine_displacement")
    num_cylinders: int | None = Field(None, ge=4, le=16, alias="num_cylinders")
    horsepower: float | None = Field(None, ge=100.0, le=1000.0, alias="horsepower")
    vehicle_weight: float | None = Field(None, ge=2000.0, le=10000.0, alias="vehicle_weight")
    acceleration: float | None = Field(None, ge=1.0, le=100.0, alias="acceleration")
    
    origin: Literal["Europe", "Asia", "USA"] | None = Field(None, alias="origin")
    fuel_type: Literal["Gasoline", "Diesel", "Hybrid"] | None = Field(None, alias="fuel_type")
    drivetrain: Literal["Front-wheel drive", "Rear-wheel drive", "All-wheel drive"] | None = Field(None, alias="drivetrain")


class PredictResponse(BaseModel):
    prediction: float


script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "model.bin")
app = FastAPI(title="car-fuel-efficiency-prediction")

with open(model_path, "rb") as f_in:
    preprocessor, model = pickle.load(f_in)


def prepare_data(car: CarPredictionRequest) -> pd.DataFrame:
    data = {
        k.lower().replace(" ", "_"): (
            v.lower().replace(" ", "_") if isinstance(v, str) else v
        )
        for k, v in car.model_dump().items()
    }
    df = pd.DataFrame([data])
    df["horsepower_is_na"] = df["horsepower"].isnull().astype("int")
    return df.fillna(0)


@app.post("/predict")
def predict(car: CarPredictionRequest) -> PredictResponse:
    df = prepare_data(car)
    X = preprocessor.transform(df)
    response = float(model.predict(X)[0])
    return PredictResponse(prediction=response)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)
