import pytest
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

@pytest.fixture(scope="session")
def data():
    return {
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
            "native_country": "China"
        }



def test_get_home():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"greeting": "Hello World!"}


def test_post_inference(data):
    r = client.post("/inference", json=data)
    assert r.status_code == 200
    assert r.json()["salary"] in ['>50K', '<=50K']
