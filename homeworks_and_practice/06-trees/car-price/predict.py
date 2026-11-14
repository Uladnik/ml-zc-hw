import pickle, uvicorn, os
import numpy as np
import pandas as pd
from fastapi import FastAPI
from typing import Optional, Literal
from pydantic import BaseModel, Field, ConfigDict


class CarPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    make: str | None = Field(None, alias="Make")
    model: str | None = Field(None, alias="Model")

    year: int | None = Field(None, ge=1990, le=2017, alias="Year")
    engine_hp: float | None = Field(None, ge=55.0, le=1001.0, alias="Engine HP")
    engine_cylinders: float | None = Field(None, ge=0.0, le=16.0, alias="Engine Cylinders")
    number_of_doors: float | None = Field(None, ge=2.0, le=4.0, alias="Number of Doors")
    highway_mpg: int | None = Field(None, ge=12, le=354, alias="highway MPG")
    city_mpg: int | None = Field(None, ge=7, le=137, alias="city mpg")
    popularity: int | None = Field(None, ge=2, le=5657, alias="Popularity")

    engine_fuel_type: Optional[
        Literal[
            "premium unleaded (required)",
            "regular unleaded",
            "premium unleaded (recommended)",
            "flex-fuel (unleaded/E85)",
            "diesel",
            "electric",
            "flex-fuel (premium unleaded recommended/E85)",
            "natural gas",
            "flex-fuel (premium unleaded required/E85)",
            "flex-fuel (unleaded/natural gas)",
        ]
    ] = Field(None, alias="Engine Fuel Type")

    transmission_type: Literal[
        "MANUAL", "AUTOMATIC", "AUTOMATED_MANUAL", "DIRECT_DRIVE", "UNKNOWN"
    ] | None = Field(None, alias="Transmission Type")

    driven_wheels: Literal[
        "rear wheel drive", "front wheel drive", "all wheel drive", "four wheel drive"
    ] | None = Field(None, alias="Driven_Wheels")

    market_category: Optional[str] = Field(None, alias="Market Category")

    vehicle_size: Literal["Compact", "Midsize", "Large"] | None = Field (
        None, alias="Vehicle Size"
    )

    vehicle_style: Literal[
        "Coupe",
        "Convertible",
        "Sedan",
        "Wagon",
        "4dr Hatchback",
        "2dr Hatchback",
        "4dr SUV",
        "Passenger Minivan",
        "Cargo Minivan",
        "Crew Cab Pickup",
        "Regular Cab Pickup",
        "Extended Cab Pickup",
        "2dr SUV",
        "Cargo Van",
        "Convertible SUV",
        "Passenger Van",
    ] | None = Field(None, alias="Vehicle Style")


class PredictResponse(BaseModel):
    prediction: int


script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "model.bin")
app = FastAPI(title="car-price-prediction")

with open(model_path, "rb") as f:
    preprocessor, model = pickle.load(f)


def prepare_data(car: CarPredictionRequest) -> pd.DataFrame:
    data = {
        k.lower().replace(" ", "_"): (
            v.lower().replace(" ", "_") if isinstance(v, str) else v
        )
        for k, v in car.model_dump().items()
    }
    return pd.DataFrame([data]).fillna(0)


@app.post("/predict")
def predict(car: CarPredictionRequest) -> PredictResponse:
    df = prepare_data(car)
    X = preprocessor.transform(df)
    response = int(np.expm1(model.predict(X)[0]))
    return PredictResponse(prediction=response)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)