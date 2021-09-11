from fastapi.testclient import TestClient

# Import our app from main.py.
# from data import app

# # Instantiate the testing client with our app.
# client = TestClient(app)

# # Write tests using the same syntax as with the requests module.
# def test_get_path():
#     r = client.get("/items/42")
#     assert r.status_code == 200
#     assert r.json() == {"fetch": "Fetched 1 of 42"}


# def test_get_path_query():
#     r = client.get("/items/42?count=5")
#     assert r.status_code == 200
#     assert r.json() == {"fetch": "Fetched 5 of 42"}


# def test_get_malformed():
#     r = client.get("/items")
#     assert r.status_code != 200

def test_main():
    pass