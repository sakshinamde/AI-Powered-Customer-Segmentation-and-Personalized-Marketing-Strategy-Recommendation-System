from __future__ import annotations

import io
import json
from typing import Any

import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_db
from app.core.config import settings
from app.core.security import create_access_token, hash_password, verify_password
from app.models import ClusterAssignment, ClusterRun, Customer, Dataset, Strategy, User
from app.schemas import StrategyUpdate, TokenResponse, UserCreate, UserOut
from app.services.ml_service import build_cluster_summary, explain_customer_cluster, prepare_feature_frame, run_algorithm, sanitize_for_json

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

router = APIRouter()


def read_csv_file(file: UploadFile) -> list[dict[str, Any]]:
    content = file.file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(content) > settings.upload_max_size_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File exceeds {settings.upload_max_size_mb} MB limit.")

    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError:
            raise HTTPException(status_code=400, detail="CSV file is not valid UTF-8 text.")

    import pandas as pd
    df = pd.read_csv(io.StringIO(text), dtype=str)
    if df.empty:
        raise HTTPException(status_code=400, detail="CSV file has no rows.")
    if "customer_id" not in {str(c).lower() for c in df.columns}:
        df.insert(0, "customer_id", [f"C{index:04d}" for index in range(1, len(df) + 1)])
    return df.to_dict(orient="records")


@router.post("/auth/register", response_model=TokenResponse)
def register_user(payload: UserCreate, db: Session = Depends(get_db)) -> TokenResponse:
    if not payload.name or not payload.email or not payload.password:
        raise HTTPException(status_code=400, detail="Name, email, and password are required.")
    if db.query(User).filter(User.email == payload.email.lower()).first():
        raise HTTPException(status_code=400, detail="User already exists.")

    user = User(
        name=payload.name.strip(),
        email=payload.email.lower(),
        password_hash=hash_password(payload.password),
        role="admin",
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_access_token(user.email)
    return TokenResponse(
        access_token=token,
        token_type="bearer",
        user=UserOut(id=user.id, name=user.name, email=user.email, role=user.role),
    )


@router.post("/auth/login", response_model=TokenResponse)
def login_user(payload: dict[str, str], db: Session = Depends(get_db)) -> TokenResponse:
    email = (payload.get("email") or "").strip().lower()
    password = payload.get("password") or ""
    if not email or not password:
        raise HTTPException(status_code=400, detail="Email and password are required.")

    user = db.query(User).filter(User.email == email).first()
    if not user or not verify_password(password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    token = create_access_token(user.email)
    return TokenResponse(
        access_token=token,
        token_type="bearer",
        user=UserOut(id=user.id, name=user.name, email=user.email, role=user.role),
    )


@router.get("/dataset")
def get_dataset_info(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict[str, Any]:
    dataset = db.query(Dataset).filter(Dataset.owner_id == current_user.id).order_by(Dataset.id.desc()).first()
    if dataset is None:
        return {"name": "Demo customer dataset (602 records)", "totalCount": 602, "numericFeatures": ["satisfaction", "browsing_frequency", "purchase_count", "income", "age"], "allFeatures": ["customer_id", "satisfaction", "browsing_frequency", "purchase_count", "income", "age"], "metrics": {"avgSatisfaction": 3.7, "avgPurchases": 5.8, "avgIncome": 48500, "avgAge": 34.2}, "sample": [], "customers": []}

    customers = db.query(Customer).filter(Customer.dataset_id == dataset.id).all()
    customer_rows = [{"customer_id": c.customer_id, **c.feature_values} for c in customers]
    sample = customer_rows[:50]
    numeric_features = list(dataset.columns) if dataset.columns else ["satisfaction", "browsing_frequency", "purchase_count", "income", "age"]
    values = [float(v) for c in customers for k, v in c.feature_values.items() if k in numeric_features]
    avg_satisfaction = sum(float(c.feature_values.get("satisfaction", 0)) for c in customers) / len(customers) if customers else 0.0
    avg_purchases = sum(float(c.feature_values.get("purchase_count", 0)) for c in customers) / len(customers) if customers else 0.0
    avg_income = sum(float(c.feature_values.get("income", 0)) for c in customers) / len(customers) if customers else 0.0
    avg_age = sum(float(c.feature_values.get("age", 0)) for c in customers) / len(customers) if customers else 0.0
    return {
        "dataset_id": dataset.id,
        "name": dataset.filename,
        "totalCount": len(customers),
        "numericFeatures": numeric_features,
        "allFeatures": ["customer_id", *numeric_features],
        "metrics": {
            "avgSatisfaction": round(avg_satisfaction, 2),
            "avgPurchases": round(avg_purchases, 2),
            "avgIncome": int(round(avg_income)),
            "avgAge": round(avg_age, 1),
        },
        "sample": sample,
        "customers": customer_rows,
    }


def get_default_customer_records() -> list[dict[str, Any]]:
    import csv
    from pathlib import Path

    candidate_paths = [
        Path(__file__).parent.parent.parent / "default_customers.csv",
        Path("backend/default_customers.csv"),
        Path("default_customers.csv"),
    ]

    csv_file = None
    for p in candidate_paths:
        if p.exists():
            csv_file = p
            break

    purchase_map = {
        'Less than once a month': 1,
        'Once a month': 2,
        'Few times a month': 3,
        'Once a week': 4,
        'Multiple times a week': 5,
    }
    browsing_map = {
        'Rarely': 1,
        'Few times a month': 2,
        'Few times a week': 3,
        'Multiple times a day': 4,
    }
    cart_completion_map = {
        'Never': 1,
        'Rarely': 2,
        'Sometimes': 3,
        'Often': 4,
        'Always': 5,
    }

    if csv_file:
        try:
            with open(csv_file, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = []
                for i, r in enumerate(reader):
                    age = int(r.get('age', 30)) if str(r.get('age', '')).strip().isdigit() else 30
                    sat = float(r.get('Shopping_Satisfaction', 3.0)) if str(r.get('Shopping_Satisfaction', '')).strip() else 3.0
                    pf = purchase_map.get(str(r.get('Purchase_Frequency', '')).strip(), 3)
                    bf = browsing_map.get(str(r.get('Browsing_Frequency', '')).strip(), 2)
                    cc = cart_completion_map.get(str(r.get('Cart_Completion_Frequency', '')).strip(), 3)
                    est_income = 25000 + (pf * 8500) + (age * 340) + (cc * 1800)
                    rows.append({
                        "customer_id": f"C{i+1:04d}",
                        "age": age,
                        "satisfaction": sat,
                        "browsing_frequency": bf,
                        "purchase_count": pf,
                        "cart_completion": cc,
                        "income": est_income,
                        "purchase_categories": str(r.get('Purchase_Categories', '')).strip(),
                        "cart_abandonment_factors": str(r.get('Cart_Abandonment_Factors', '')).strip(),
                        "service_appreciation": str(r.get('Service_Appreciation', '')).strip(),
                        "improvement_areas": str(r.get('Improvement_Areas', '')).strip(),
                    })
                if rows:
                    return rows
        except Exception:
            pass

    return [
        {
            "customer_id": f"C{index:04d}",
            "age": 20 + ((index * 3) % 45),
            "income": 30000 + ((index * 137) % 65000),
            "satisfaction": round(1.0 + ((index * 11) % 40) / 10, 2),
            "browsing_frequency": 1 + (index % 4),
            "purchase_count": 1 + ((index * 2) % 5),
        }
        for index in range(1, 603)
    ]


def ensure_user_dataset(db: Session, user_id: int, dataset_id: int | None = None) -> Dataset:
    if dataset_id is not None:
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id, Dataset.owner_id == user_id).first()
        if dataset:
            return dataset

    dataset = db.query(Dataset).filter(Dataset.owner_id == user_id).order_by(Dataset.id.desc()).first()
    if dataset is not None:
        return dataset

    demo_rows = get_default_customer_records()
    ds = Dataset(
        owner_id=user_id,
        filename="Amazon Customer Behavior Survey (602 records)",
        row_count=len(demo_rows),
        columns=["satisfaction", "purchase_count", "browsing_frequency", "cart_completion", "income", "age"],
    )
    db.add(ds)
    db.commit()
    db.refresh(ds)

    for row in demo_rows:
        db.add(Customer(dataset_id=ds.id, customer_id=row["customer_id"], feature_values={k: v for k, v in row.items() if k != "customer_id"}, raw_row=row))
    db.commit()
    return ds


@router.post("/dataset/demo")
def load_demo_dataset(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict[str, Any]:
    demo_rows = get_default_customer_records()
    ds = Dataset(
        owner_id=current_user.id,
        filename="Amazon Customer Behavior Survey (602 records)",
        row_count=len(demo_rows),
        columns=["satisfaction", "purchase_count", "browsing_frequency", "cart_completion", "income", "age"],
    )
    db.add(ds)
    db.commit()
    db.refresh(ds)

    for row in demo_rows:
        db.add(Customer(dataset_id=ds.id, customer_id=row["customer_id"], feature_values={k: v for k, v in row.items() if k != "customer_id"}, raw_row=row))
    db.commit()
    return {"success": True, "message": f"Loaded Amazon customer survey dataset with {len(demo_rows)} customers.", "count": len(demo_rows)}


@router.post("/upload")
async def upload_dataset(
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    file: UploadFile | None = None,
) -> dict[str, Any]:
    request_json = None
    try:
        request_json = await request.json()
    except Exception:
        request_json = None

    if request_json is None and file is not None:
        rows = read_csv_file(file)
        request_json = {"filename": file.filename, "data": rows}
    if request_json is None:
        raise HTTPException(status_code=400, detail="Uploaded file data is required.")

    rows = request_json.get("data", []) if isinstance(request_json, dict) else []
    filename = request_json.get("filename", "uploaded_dataset.csv") if isinstance(request_json, dict) else "uploaded_dataset.csv"
    if not rows and file is not None:
        rows = read_csv_file(file)
    if not rows:
        raise HTTPException(status_code=400, detail="No rows provided for upload.")

    import pandas as pd
    df = pd.DataFrame(rows)
    numeric_columns = [
        col for col in df.columns
        if str(col).lower() != "customer_id" and pd.api.types.is_numeric_dtype(pd.to_numeric(df[col], errors="coerce"))
    ]
    if not numeric_columns:
        raise HTTPException(status_code=400, detail="CSV file contains no numeric columns besides customer_id.")

    ds = Dataset(owner_id=current_user.id, filename=filename, row_count=len(df), columns=numeric_columns)
    db.add(ds)
    db.commit()
    db.refresh(ds)

    feature_data = []
    for row in rows:
        feature_values = {key: row.get(key) for key in numeric_columns}
        customer_id = str(row.get("customer_id") or f"C{len(feature_data)+1:04d}")
        db.add(Customer(dataset_id=ds.id, customer_id=customer_id, feature_values=feature_values, raw_row=row))
        feature_data.append({"customer_id": customer_id, **feature_values})
    db.commit()

    return {
        "dataset_id": ds.id,
        "filename": filename,
        "row_count": len(feature_data),
        "columns": numeric_columns,
        "preview": feature_data[:5],
    }


@router.get("/customers")
def list_customers(
    dataset_id: int = Query(...),
    search: str = Query(default=""),
    limit: int = Query(default=100),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[dict[str, Any]]:
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id, Dataset.owner_id == current_user.id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    query = db.query(Customer).filter(Customer.dataset_id == dataset.id)
    if search:
        query = query.filter(Customer.customer_id.ilike(f"%{search}%"))

    customers = query.limit(limit).all()
    result = []
    for c in customers:
        result.append({
            "customer_id": c.customer_id,
            "feature_values": c.feature_values,
            "cluster_label": None,
        })
    return result


@router.get("/customers/{customer_id}")
def get_customer(
    customer_id: str,
    dataset_id: int = Query(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict[str, Any]:
    customer = (
        db.query(Customer)
        .filter(Customer.dataset_id == dataset_id, Customer.customer_id == customer_id, Dataset.owner_id == current_user.id)
        .join(Dataset, Dataset.id == Customer.dataset_id)
        .first()
    )
    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found.")

    cluster_label = None
    latest_cluster = (
        db.query(ClusterAssignment)
        .join(ClusterRun, ClusterRun.id == ClusterAssignment.cluster_run_id)
        .filter(ClusterRun.dataset_id == dataset_id, ClusterAssignment.customer_id == customer.id)
        .order_by(ClusterRun.created_at.desc())
        .first()
    )
    if latest_cluster:
        cluster_label = latest_cluster.cluster_label
    return {"customer_id": customer.customer_id, **customer.feature_values, "cluster_label": cluster_label}


@router.get("/clusters")
def get_clusters(
    dataset_id: int | None = Query(default=None),
    method: str = Query(default="kmeans"),
    k: int = Query(default=4),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict[str, Any]:
    dataset = ensure_user_dataset(db, current_user.id, dataset_id)

    customers = db.query(Customer).filter(Customer.dataset_id == dataset.id).all()
    if not customers:
        raise HTTPException(status_code=400, detail="No customers available in dataset.")

    df = __import__("pandas").DataFrame([{"customer_id": c.customer_id, **c.feature_values} for c in customers])
    feature_df = prepare_feature_frame(df)
    try:
        result = run_algorithm(feature_df, method, k=k)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    labels = result["labels"]
    projection = result["projection"]

    cluster_run = ClusterRun(
        dataset_id=dataset.id,
        algorithm=method,
        k=k if method in {"kmeans", "hierarchical", "pca", "embedding"} else None,
        params=result.get("params", {}),
        silhouette_score=float(result["silhouette_score"]),
        n_clusters=int(len(set(labels)) - (1 if -1 in labels else 0)),
    )
    db.add(cluster_run)
    db.commit()
    db.refresh(cluster_run)

    assignments = []
    for idx, customer in enumerate(customers):
        point = {
            "cluster_run_id": cluster_run.id,
            "customer_id": customer.id,
            "cluster_label": int(labels[idx]),
            "proj_x": float(projection[idx][0]) if len(projection[idx]) > 0 else 0.0,
            "proj_y": float(projection[idx][1]) if len(projection[idx]) > 1 else 0.0,
        }
        assignments.append(
            ClusterAssignment(
                cluster_run_id=cluster_run.id,
                customer_id=customer.id,
                cluster_label=int(labels[idx]),
                proj_x=point["proj_x"],
                proj_y=point["proj_y"],
            )
        )
    db.add_all(assignments)
    db.commit()

    points = [{
        "customer_id": customer.customer_id,
        "cluster_label": int(labels[i]),
        "x": float(projection[i][0]) if len(projection[i]) > 0 else 0.0,
        "y": float(projection[i][1]) if len(projection[i]) > 1 else 0.0,
    } for i, customer in enumerate(customers)]

    summary = build_cluster_summary(feature_df, labels)
    cluster_summaries = [
        {
            "clusterId": int(item["cluster_label"]),
            "name": f"Cluster {int(item['cluster_label']) + 1}",
            "count": int(item["count"]),
            "avgSatisfaction": round(float(item.get('avg_satisfaction', 0.0)), 2),
        }
        for item in summary
    ]
    return {
        "run_id": cluster_run.id,
        "algorithm": method,
        "k": k,
        "n_clusters": cluster_run.n_clusters,
        "silhouette_score": float(cluster_run.silhouette_score),
        "silhouette": float(cluster_run.silhouette_score),
        "labels": labels.tolist(),
        "points": points,
        "activeClustersCount": cluster_run.n_clusters,
        "clusterSummaries": cluster_summaries,
        "cluster_summary": summary,
        "projectionX": [float(p[0]) for p in projection],
        "projectionY": [float(p[1]) if len(p) > 1 else 0.0 for p in projection],
        "totalCustomers": len(customers),
        "params": result.get("params", {}),
        "suggested_k": result.get("params", {}).get("suggested_k"),
    }


@router.get("/explain-cluster/{customer_id}")


def explain_cluster(
    customer_id: str,
    dataset_id: int | None = Query(default=None),
    run_id: int | None = Query(default=None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict[str, Any]:
    dataset = ensure_user_dataset(db, current_user.id, dataset_id)
    customer = db.query(Customer).filter(Customer.dataset_id == dataset.id, Customer.customer_id == customer_id).first()
    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found.")

    cluster_run = db.query(ClusterRun).filter(ClusterRun.dataset_id == dataset.id).order_by(ClusterRun.created_at.desc()).first()
    if run_id is not None:
        cluster_run = db.query(ClusterRun).filter(ClusterRun.id == run_id, ClusterRun.dataset_id == dataset.id).first()
    if not cluster_run:
        raise HTTPException(status_code=404, detail="No cluster run available for this dataset.")

    assignments = db.query(ClusterAssignment).filter(ClusterAssignment.cluster_run_id == cluster_run.id, ClusterAssignment.customer_id == customer.id).all()
    cluster_label = int(assignments[0].cluster_label) if assignments else 0
    customers = db.query(Customer).filter(Customer.dataset_id == dataset.id).all()
    df = __import__("pandas").DataFrame([{"customer_id": c.customer_id, **c.feature_values} for c in customers])
    feature_df = prepare_feature_frame(df)
    labels = np.array([assignment.cluster_label for assignment in db.query(ClusterAssignment).filter(ClusterAssignment.cluster_run_id == cluster_run.id).order_by(ClusterAssignment.id).all()])
    current_idx = [c.customer_id for c in customers].index(customer.customer_id)
    explanation = explain_customer_cluster(feature_df, labels, current_idx)
    return {
        "customer_id": customer.customer_id,
        "cluster_label": cluster_label,
        "contributions": explanation["contributions"],
        "summary_text": explanation["summary_text"],
    }


@router.get("/strategies")
def get_strategies(
    dataset_id: int | None = Query(default=None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict[str, Any] | list[dict[str, Any]]:
    dataset = ensure_user_dataset(db, current_user.id, dataset_id)
    rows = db.query(Strategy).filter(Strategy.dataset_id == dataset.id).all()
    if dataset_id is None:
        return {"strategies": [{"clusterId": r.cluster_label, "title": r.title, "strategy": r.recommendation, "updatedAt": r.updated_at.isoformat()} for r in rows]}
    return [{
        "cluster_label": r.cluster_label,
        "title": r.title,
        "recommendation": r.recommendation,
        "updated_at": r.updated_at.isoformat(),
    } for r in rows]


@router.put("/strategies/{cluster_label}")
def upsert_strategy(
    cluster_label: int,
    payload: StrategyUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict[str, Any]:
    dataset = ensure_user_dataset(db, current_user.id, payload.dataset_id)

    recommendation = payload.recommendation_text.strip()
    title = (payload.title_text or "Strategy").strip()
    if not recommendation:
        raise HTTPException(status_code=400, detail="Strategy text is required.")

    strategy = db.query(Strategy).filter(Strategy.dataset_id == dataset.id, Strategy.cluster_label == cluster_label).first()
    if not strategy:
        strategy = Strategy(dataset_id=dataset.id, cluster_label=cluster_label, title=title, recommendation=recommendation)
        db.add(strategy)
    else:
        strategy.title = title
        strategy.recommendation = recommendation
    db.commit()
    db.refresh(strategy)
    return {
        "id": strategy.id,
        "dataset_id": dataset.id,
        "cluster_label": strategy.cluster_label,
        "title": strategy.title,
        "recommendation": strategy.recommendation,
        "updated_at": strategy.updated_at.isoformat(),
    }


@router.get("/compare")
def compare_models(
    dataset_id: int | None = Query(default=None),
    k: int = Query(default=4),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict[str, Any]:
    dataset = ensure_user_dataset(db, current_user.id, dataset_id)

    customers = db.query(Customer).filter(Customer.dataset_id == dataset.id).all()
    if not customers:
        raise HTTPException(status_code=400, detail="No customer data for comparison.")

    df = __import__("pandas").DataFrame([{"customer_id": c.customer_id, **c.feature_values} for c in customers])
    feature_df = prepare_feature_frame(df)
    result = []
    for method in ["kmeans", "dbscan", "hierarchical", "pca", "embedding"]:
        try:
            algo = run_algorithm(feature_df, method, k=k)
            result.append({
                "method": method,
                "status": "ok",
                "silhouette_score": float(algo["silhouette_score"]),
                "n_clusters": int(len(set(algo["labels"])) - (1 if -1 in algo["labels"] else 0)),
                "params": algo.get("params", {}),
            })
        except Exception as exc:
            result.append({
                "method": method,
                "status": "failed",
                "silhouette_score": 0.0,
                "n_clusters": 0,
                "params": {"error": str(exc)},
            })

    return {"dataset_id": dataset.id, "results": result}


@router.get("/comparison")
def comparison_alias(
    dataset_id: int | None = Query(default=None),
    k: int = Query(default=4),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict[str, Any]:
    return compare_models(dataset_id=dataset_id, k=k, db=db, current_user=current_user)



