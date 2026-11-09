import requests

url = "http://localhost:9696/predict"

car = {
    "model_year": 1995,
    "origin": "USA",
    "fuel_type": None,
    "drivetrain": "Rear-wheel drive",
    "num_doors": 2,
    "engine_displacement": 2200,
    "num_cylinders": 6,
    "horsepower": None, 
    "vehicle_weight": 4100.0,
    "acceleration": 16.5,
}

response = requests.post(url, json=car)
predictions = response.json()

print(predictions)
