import requests

API_URL = "https://datasets-server.huggingface.co/valid"

def query():
    response = requests.request("GET", API_URL)
    return response.json()["valid"]