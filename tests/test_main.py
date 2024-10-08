# tests/test_main.py

import pytest
from fastapi.testclient import TestClient
from app.main import app
import nltk

client = TestClient(app)

@pytest.fixture(scope='module', autouse=True)
def setup_nltk():
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')

def test_predict_1():
    payload = {"text": "How can I learn Python online using Google Colab?"}
    response = client.post("/predict", json=payload)
    response_json = response.json()
    assert response.status_code == 200
    assert response_json["unsupervised_tags"] == ["google"]
    assert response_json["supervised_tags"] == ["python", "google"]


def test_predict_2():
    payload = {"text": "How can i setup pipelines using gitLab CI ?"}
    response = client.post("/predict", json=payload)
    response_json = response.json()
    assert response.status_code == 200
    assert response_json["unsupervised_tags"] == []
    assert response_json["supervised_tags"] == []


def test_predict_3():
    payload = {"text": "C++17 Tuple Deduction Guides (CTAD): Implicitly-Generated vs User-Defined"}
    response = client.post("/predict", json=payload)
    response_json = response.json()
    assert response.status_code == 200
    assert response_json["unsupervised_tags"] == ["user"]
    assert response_json["supervised_tags"] == []