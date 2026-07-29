import os
import sys
import pytest
from fastapi.testclient import TestClient

# Ensure backend folder is on python path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.main import app
from app.database.connection import init_db

client = TestClient(app)

@pytest.fixture(scope="module", autouse=True)
def setup_db():
    init_db()
    yield

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "service": "onconode-api"}

def test_static_home():
    response = client.get("/")
    assert response.status_code == 200

def test_login_unauthorized():
    response = client.post("/api/auth/login", json={"username": "fake", "password": "fake"})
    assert response.status_code == 401

def test_login_success():
    response = client.post("/api/auth/login", json={
        "username": "admin_clinician", 
        "password": "OncoPassSecure99!"
    })
    assert response.status_code == 200
    assert "access_token" in response.json()

def test_run_diagnostics_without_patient():
    features = {
        "mean radius": 14.1, "mean texture": 19.2, "mean perimeter": 91.9, "mean area": 654.0,
        "mean compactness": 0.104, "mean concavity": 0.088, "radius error": 0.4, "area error": 40.3,
        "concavity error": 0.031, "worst radius": 16.2, "worst perimeter": 107.2, "worst area": 880.0,
        "worst concavity": 0.272, "worst concave points": 0.114, "worst symmetry": 0.29
    }
    # Should return 404 because patient not created yet
    response = client.post("/api/diagnostics", json={
        "patient_id": "PAT-NONEXIST",
        "features": features
    })
    assert response.status_code == 404
