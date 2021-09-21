from fastapi.testclient import TestClient

# Import our app from main.py.
from main import census_app

# Instantiate the testing client with our app.
client = TestClient(census_app)

# test Fast API root
def test_api_locally_get_root():
    """ Test Fast API root route"""

    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"message": "Census Bureau Salary Prediction App"}


def test_api_locally_get_predictions_inf1():
    """ Test Fast API predict route with a '<=50K' salary prediction result """

    expected_res = "Predicts ['<=50K']"

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
    assert (r.json()["predict"][: len(expected_res)]) == expected_res


def test_api_locally_get_predictions_inf2():
    """ Test Fast API predict route with a '>50K' salary prediction result """

    expected_res = "Predicts ['>50K']"

    r = client.post(
        "/predict",
        # headers={"X-Token": "hailhydra"},
        json={
            "age": 35,
            "workclass": "Private",
            "fnlgt": 159449,
            "education": "Masters",
            "education_num": 9,
            "marital_status": "Married-civ-spouse",
            "occupation": "Exec-managerial",
            "relationship": "Husband",
            "race": "White",
            "sex": "Male",
            "capital_gain": 20000,
            "capital_loss": 0,
            "hours_per_week": 45,
            "native_country": "United-States",
        },
    )
    assert r.status_code == 200
    assert (r.json()["predict"][: len(expected_res)]) == expected_res
