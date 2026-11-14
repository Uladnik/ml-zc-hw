import requests
url = "http://localhost:9696/predict"
sample_car_dict = {
    "Make": "BMW",
    "Model": "3 Series",
    "Year": 2015,
    "Engine HP": 240.0,
    "Engine Cylinders": 4.0,
    "Number of Doors": 4.0,
    "highway MPG": 35,
    "city mpg": 23,
    "Popularity": None,  
    "Engine Fuel Type": "premium unleaded (required)",
    "Transmission Type": "AUTOMATIC",
    "Driven_Wheels": "rear wheel drive",
    "Market Category": "Luxury,Factory Tuner",
    "Vehicle Size": "Compact",
    "Vehicle Style": "Sedan"
}

print(requests.post(url, json=car).json())