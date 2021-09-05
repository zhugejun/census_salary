import pytest
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

@pytest.fixture(scope="session")
def data_less_than_50k():
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


@pytest.fixture(scope='session')
def data_greater_than_50k():
    example = {
        "age": 52,
        "workclass": "Self-emp-inc",
        "fnlgt": 287927,
        "education": "Doctorate",
        "education_num": 16,
        "marital_status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital_gain": 16485,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States"
    }
    return example


def test_get_home():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"greeting": "Hello World!"}


def test_less_than_fiftyk(data_less_than_50k):
    r = client.post("/inference", json=data_less_than_50k)
    assert r.status_code == 200
    assert r.json()["salary"] == '<=50K'


def test_greater_than_fiftyk(data_greater_than_50k): 
    r = client.post("/inference", json=data_greater_than_50k)
    assert r.status_code == 200
    assert r.json()["salary"] == '>50K'

