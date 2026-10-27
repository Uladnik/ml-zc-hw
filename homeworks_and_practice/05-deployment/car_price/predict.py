import pickle, uvicorn, os
import numpy as np
import pandas as pd
from fastapi import FastAPI
from typing import Optional, Literal
from pydantic import BaseModel, Field, ConfigDict


class CarPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    make: str = Field(..., alias="Make")
    model: str = Field(..., alias="Model")

    year: int = Field(..., ge=1990, le=2017, alias="Year")
    engine_hp: float = Field(..., ge=55.0, le=1001.0, alias="Engine HP")
    engine_cylinders: float = Field(..., ge=0.0, le=16.0, alias="Engine Cylinders")
    number_of_doors: float = Field(..., ge=2.0, le=4.0, alias="Number of Doors")
    highway_mpg: int = Field(..., ge=12, le=354, alias="highway MPG")
    city_mpg: int = Field(..., ge=7, le=137, alias="city mpg")
    popularity: int = Field(..., ge=2, le=5657, alias="Popularity")

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
    ] = Field(..., alias="Transmission Type")

    driven_wheels: Literal[
        "rear wheel drive", "front wheel drive", "all wheel drive", "four wheel drive"
    ] = Field(..., alias="Driven_Wheels")

    market_category: Optional[str] = Field(None, alias="Market Category")

    vehicle_size: Literal["Compact", "Midsize", "Large"] = Field(
        ..., alias="Vehicle Size"
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
    ] = Field(..., alias="Vehicle Style")


class PredictResponse(BaseModel):
    prediction: int


script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "model.bin")
app = FastAPI(title="car-price-prediction")

with open(model_path, "rb") as f_in:
    pipeline = pickle.load(f_in)


def prepare_data(car: CarPredictionRequest) -> pd.DataFrame:
    data = {
        k.lower().replace(" ", "_"): (
            v.lower().replace(" ", "_") if isinstance(v, str) else v
        )
        for k, v in car.model_dump().items()
    }
    df = pd.DataFrame([data])
    for col in df.select_dtypes(include=["number"]).columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(exclude=["number"]).columns:
        df[col] = df[col].fillna("missing")
    df.number_of_doors = df.number_of_doors.astype("str")
    df.engine_cylinders = df.engine_cylinders.astype("str")
    df.year -= df.year.min()
    return df


@app.post("/predict")
def predict(car: CarPredictionRequest) -> PredictResponse:
    df = prepare_data(car)
    price = int(np.expm1(pipeline.predict(df)[0]))
    return PredictResponse(prediction=price)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)
