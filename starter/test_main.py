from fastapi.testclient import TestClient

# Import our app from main.py.
from main import census_app

# Instantiate the testing client with our app.
client = TestClient(census_app)

# test Fast API root
def test_api_locally_get_root():
    r = client.get("/")
    assert r.status_code == 200


def test_api_locally_get_predictions():
    r = client.post(
        "/predict",
        # headers={"X-Token": "hailhydra"},
        json={
            "age": 39,
            "workclass": "State-gov",
            "fnlgt": 77516,
            "education": "Bachelors",
            "education_num": 13,
            "marital_status": "Never-married",
            "occupation": "Adm-clerical",
            "relationship": "Not-in-family",
            "race": "White",
            "sex": "Male",
            "capital_gain": 2174,
            "capital_loss": 0,
            "hours_per_week": 40,
            "native_country": "United-States",
        },
    )
    assert r.status_code == 200


def test_main():
    pass
