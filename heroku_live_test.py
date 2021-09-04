import requests
import json


data = {
    "age": 36,
    "workclass": "State-gov",
    "fnlgt": 189778,
    "education": "Masters",
    "education_num": 12,
    "marital_status": "MarriedMarried-civ-spouse",
    "occupation": "Adm-clericalTech-support",
    "relationship": "Husband",
    "race": "WhiteAsian-Pac-Islander",
    "sex": "Male",
    "capital_gain": 1077,
    "capital_loss": 0,
    "hours_per_week": 40,
    "native_country": "China" }
          

response = requests.post('https://boiling-journey-72876.herokuapp.com/', data=json.dumps(data))

print(response.status_code)
print(response.json())