import requests

url = "http://localhost:9696/predict"

car = {
    "Make": "BMW",
    "Model": "3 Series",
    "Year": 2015,
    "Engine Fuel Type": "premium unleaded (required)",
    "Engine HP": 300.0,
    "Engine Cylinders": 6.0,
    "Number of Doors": 4.0,
    "Transmission Type": "AUTOMATIC",
    "Driven_Wheels": "rear wheel drive",
    "Market Category": "Luxury,Performance",
    "Vehicle Size": "Compact",
    "Vehicle Style": "Sedan",
    "highway MPG": 31,
    "city mpg": 22,
    "Popularity": 3916,
}

response = requests.post(url, json=car)
predictions = response.json()

print(predictions)
