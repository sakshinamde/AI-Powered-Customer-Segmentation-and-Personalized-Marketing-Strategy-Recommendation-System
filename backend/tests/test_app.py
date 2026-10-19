import os
import time
from pathlib import Path

from uuid import uuid4

from fastapi.testclient import TestClient

os.environ["DATABASE_URL"] = "sqlite:///./smartseg_test.db"
os.environ["SECRET_KEY"] = "test-secret-key"

if Path("smartseg_test.db").exists():
    Path("smartseg_test.db").unlink()

from app.main import app
from app.db.session import Base, engine
from app.core.config import settings
from app.services.ml_service import dbscan_cluster, hierarchical_cluster
import pandas as pd
from sklearn.datasets import make_blobs

Base.metadata.create_all(bind=engine)


def test_health_endpoint() -> None:
    client = TestClient(app)
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_auth_route_is_registered() -> None:
    client = TestClient(app)
    unique_email = f"test-{uuid4().hex[:8]}@example.com"
    response = client.post(
        "/api/auth/register",
        json={"name": "Test User", "email": unique_email, "password": "secret123"},
    )

    assert response.status_code != 404


def test_protected_route_requires_authentication() -> None:
    response = TestClient(app).get("/api/clusters")
    assert response.status_code == 401


def test_frontend_dataset_flow_works() -> None:
    client = TestClient(app)
    email = f"frontend-flow-{uuid4().hex[:8]}@example.com"
    register = client.post(
        "/api/auth/register",
        json={"name": "Frontend Flow", "email": email, "password": "secret123"},
    )
    assert register.status_code == 200, register.text
    token = register.json()["access_token"]

    upload = client.post(
        "/api/upload",
        json={
            "filename": "demo.csv",
            "data": [
                {
                    "customer_id": "C0001",
                    "age": 30,
                    "income": 50000,
                    "satisfaction": 4.2,
                    "browsing_frequency": 8,
                    "purchase_count": 5,
                },
                {
                    "customer_id": "C0002",
                    "age": 38,
                    "income": 62000,
                    "satisfaction": 3.4,
                    "browsing_frequency": 4,
                    "purchase_count": 3,
                },
            ],
        },
        headers={"Authorization": f"Bearer {token}"},
    )
    assert upload.status_code == 200, upload.text
    dataset_id = upload.json()["dataset_id"]

    dataset = client.get(
        "/api/dataset",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert dataset.status_code == 200, dataset.text
    assert dataset.json()["totalCount"] >= 2

    clusters = client.get(
        "/api/clusters",
        params={"dataset_id": dataset_id, "method": "kmeans", "k": 2},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert clusters.status_code == 200, clusters.text
    assert len(clusters.json()["points"]) == 2

    customers = client.get(
        "/api/customers",
        params={"dataset_id": dataset_id, "search": "C0001"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert customers.status_code == 200, customers.text
    assert len(customers.json()) >= 1

    strategy_save = client.put(
        f"/api/strategies/0",
        json={"title": "Retention Focus", "strategy": "Focus on high-value retention offers."},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert strategy_save.status_code == 200, strategy_save.text
    assert strategy_save.json()["cluster_label"] == 0

    comparison = client.get(
        "/api/compare",
        params={"dataset_id": dataset_id, "k": 2},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert comparison.status_code == 200, comparison.text
    for result in comparison.json()["results"]:
        assert result["status"] in {"ok", "failed"}
        assert "accuracy" not in result


def test_dbscan_scales_without_dense_pairwise_matrix() -> None:
    values, _ = make_blobs(n_samples=5000, centers=4, n_features=5, random_state=42)
    frame = pd.DataFrame(values, columns=[f"feature_{i}" for i in range(5)])
    started = time.perf_counter()
    result = dbscan_cluster(frame)
    elapsed = time.perf_counter() - started
    assert 0.5 <= result["params"]["eps"] < 10
    assert elapsed < 5


def test_hierarchical_rejects_oversized_dataset() -> None:
    values, _ = make_blobs(n_samples=settings.hierarchical_max_rows + 1, centers=3, n_features=3, random_state=42)
    frame = pd.DataFrame(values, columns=["a", "b", "c"])
    try:
        hierarchical_cluster(frame, 3)
    except ValueError as exc:
        assert "Choose another algorithm" in str(exc)
    else:
        raise AssertionError("Expected hierarchical clustering row limit error")
