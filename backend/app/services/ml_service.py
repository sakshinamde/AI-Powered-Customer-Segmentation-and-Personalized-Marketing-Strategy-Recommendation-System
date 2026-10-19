from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import shap
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception:  # pragma: no cover
    torch = None
    nn = None
    optim = None


def clean_numeric(value: Any) -> float | None:
    try:
        if value is None or (isinstance(value, str) and value.strip() == ""):
            return None
        out = float(value)
        if np.isnan(out) or np.isinf(out):
            return None
        return out
    except (TypeError, ValueError):
        return None


def sanitize_for_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, tuple):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        if np.isnan(obj) or np.isinf(obj):
            return 0.0
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def numeric_columns(df: pd.DataFrame) -> list[str]:
    columns = []
    for col in df.columns:
        if str(col).lower() == "customer_id":
            continue
        sample = df[col].dropna()
        if sample.empty:
            continue
        try:
            pd.to_numeric(sample, errors="raise")
            columns.append(str(col))
        except Exception:
            continue
    return columns


def prepare_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    columns = numeric_columns(df)
    if not columns:
        raise ValueError("CSV must contain at least one numeric column besides customer_id.")

    feature_df = df[columns].copy()
    for col in columns:
        feature_df[col] = pd.to_numeric(feature_df[col], errors="coerce")
    feature_df = feature_df.dropna(axis=0, how="any")
    if feature_df.empty:
        raise ValueError("No valid numeric rows found after validation.")
    return feature_df.reset_index(drop=True)


def standardized_features(df: pd.DataFrame) -> tuple[np.ndarray, StandardScaler, list[str]]:
    columns = list(df.columns)
    scaler = StandardScaler()
    matrix = scaler.fit_transform(df[columns].values)
    return matrix, scaler, columns


def compute_silhouette(standardized: np.ndarray, labels: np.ndarray) -> float:
    valid = labels != -1
    if standardized.shape[0] < 3 or valid.sum() < 2 or len(set(labels[valid].tolist())) < 2:
        return 0.0
    score = silhouette_score(standardized[valid], labels[valid])
    return float(np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0))


def kmeans_cluster(df: pd.DataFrame, k: int) -> dict[str, Any]:
    X, _, columns = standardized_features(df)
    model = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = model.fit_predict(X)
    suggested_k = min(max(2, k), len(df)) if len(df) > 1 else k
    return {
        "labels": labels,
        "projection": model.transform(X),
        "silhouette_score": compute_silhouette(X, labels),
        "params": {"n_clusters": k, "n_init": 10, "max_iter": 300, "suggested_k": suggested_k},
        "columns": columns,
    }


def dbscan_cluster(df: pd.DataFrame) -> dict[str, Any]:
    X, _, columns = standardized_features(df)
    # heuristic based on median nearest-neighbor distance for a sane default eps
    if X.shape[0] > 1:
        neighbor_count = min(8, X.shape[0])
        distances, _ = NearestNeighbors(n_neighbors=neighbor_count).fit(X).kneighbors(X)
        med = np.nanmedian(distances[:, -1]) if distances.size else 1.0
        eps = max(float(med * 0.8), 0.5)
    else:
        eps = 0.5
    min_samples = max(4, min(10, int(np.ceil(np.sqrt(len(df)) / 2))))
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)
    return {
        "labels": labels,
        "projection": X[:, :2],
        "silhouette_score": compute_silhouette(X, labels),
        "params": {"eps": round(float(eps), 4), "min_samples": min_samples, "heuristic": "k-distance median"},
        "columns": columns,
    }


def hierarchical_cluster(df: pd.DataFrame, k: int) -> dict[str, Any]:
    from app.core.config import settings

    if len(df) > settings.hierarchical_max_rows:
        raise ValueError(
            f"Hierarchical clustering supports at most {settings.hierarchical_max_rows} rows. "
            "Choose another algorithm or upload a smaller dataset."
        )
    X, _, columns = standardized_features(df)
    model = AgglomerativeClustering(n_clusters=k, linkage="average")
    labels = model.fit_predict(X)
    return {
        "labels": labels,
        "projection": X[:, :2],
        "silhouette_score": compute_silhouette(X, labels),
        "params": {"n_clusters": k, "linkage": "average"},
        "columns": columns,
    }


def pca_kmeans_cluster(df: pd.DataFrame, k: int) -> dict[str, Any]:
    X, _, columns = standardized_features(df)
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(X)
    model = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = model.fit_predict(reduced)
    return {
        "labels": labels,
        "projection": reduced,
        "silhouette_score": compute_silhouette(X, labels),
        "params": {"n_components": 2, "method": "pca_then_kmeans", "explained_variance_ratio": pca.explained_variance_ratio_.tolist()},
        "columns": columns,
    }


def encoder_model(input_dim: int):
    if torch is None or nn is None or optim is None:
        return None

    class Autoencoder(nn.Module):
        def __init__(self, in_dim: int):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(in_dim, max(8, in_dim // 2)),
                nn.ReLU(),
                nn.Linear(max(8, in_dim // 2), 2),
            )
            self.decoder = nn.Sequential(
                nn.Linear(2, max(8, in_dim // 2)),
                nn.ReLU(),
                nn.Linear(max(8, in_dim // 2), in_dim),
            )

        def forward(self, x):
            z = self.encoder(x)
            return z, self.decoder(z)

    return Autoencoder(input_dim)


def embedding_cluster(df: pd.DataFrame, k: int) -> dict[str, Any]:
    X, _, columns = standardized_features(df)
    model = None
    if torch is not None:
        model = encoder_model(X.shape[1])
        if model is not None:
            optimizer = optim.Adam(model.parameters(), lr=0.01)
            criterion = nn.MSELoss()
            tensor = torch.tensor(X, dtype=torch.float32)
            for _ in range(25):
                optimizer.zero_grad()
                z, recon = model(tensor)
                loss = criterion(recon, tensor)
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                embedding = model.encoder(tensor).numpy()
            kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels = kmeans.fit_predict(embedding)
            return {
                "labels": labels,
                "projection": embedding,
                "silhouette_score": compute_silhouette(X, labels),
                "params": {"method": "torch_autoencoder", "bottleneck_dim": 2, "fallback_used": False},
                "columns": columns,
            }

    # Fallback if torch is unavailable
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(X)
    model = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = model.fit_predict(reduced)
    return {
        "labels": labels,
        "projection": reduced,
        "silhouette_score": compute_silhouette(X, labels),
        "params": {"method": "pca_fallback", "bottleneck_dim": 2, "fallback_used": True, "note": "PyTorch unavailable; PCA used as a fallback stand-in for embedding."},
        "columns": columns,
    }


def run_algorithm(df: pd.DataFrame, method: str, k: int = 4) -> dict[str, Any]:
    method = (method or "kmeans").lower()
    if method == "kmeans":
        return kmeans_cluster(df, k)
    if method == "dbscan":
        return dbscan_cluster(df)
    if method == "hierarchical":
        return hierarchical_cluster(df, k)
    if method == "pca":
        return pca_kmeans_cluster(df, k)
    if method == "embedding":
        return embedding_cluster(df, k)
    raise ValueError(f"Unsupported clustering method: {method}")


def explain_customer_cluster(df: pd.DataFrame, labels: np.ndarray, customer_index: int) -> dict[str, Any]:
    X = df.copy()
    y = np.array(labels)
    if len(y) == 0 or customer_index >= len(y) or customer_index < 0:
        return {"contributions": [], "summary_text": "Customer index out of bounds.", "cluster_label": 0}

    target_cluster = int(y[customer_index])
    unique_labels = [lbl for lbl in set(y.tolist()) if lbl != -1]

    # Try SHAP + RandomForest
    row_values = None
    try:
        if len(unique_labels) >= 2 and len(X) >= 4:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X, y)

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)

            if isinstance(shap_values, list):
                class_idx = list(model.classes_).index(target_cluster) if target_cluster in model.classes_ else 0
                values = shap_values[class_idx]
                row_values = values[customer_index]
            elif isinstance(shap_values, np.ndarray):
                if shap_values.ndim == 3:
                    class_idx = list(model.classes_).index(target_cluster) if target_cluster in model.classes_ else 0
                    row_values = shap_values[customer_index, :, class_idx]
                elif shap_values.ndim == 2:
                    row_values = shap_values[customer_index]
                else:
                    row_values = shap_values.flatten()[:len(X.columns)]
    except Exception:
        pass

    if row_values is None or len(row_values) != len(X.columns):
        # Fallback to normalized feature deviation from dataset mean
        means = X.mean()
        stds = X.std().replace(0, 1)
        cust_row = X.iloc[customer_index]
        row_values = ((cust_row - means) / stds).values

    contributions = []
    for col, val in zip(X.columns, row_values):
        try:
            scalar_val = float(np.ravel(val)[0])
        except Exception:
            scalar_val = 0.0
        contributions.append((str(col), scalar_val))

    contributions = sorted(contributions, key=lambda item: abs(item[1]), reverse=True)
    top = contributions[:5]

    summary = (
        f"The strongest drivers for customer {customer_index} are {', '.join(f'{feature} ({value:+.3f})' for feature, value in top)}. "
        f"This profile is most aligned to cluster {target_cluster + 1 if target_cluster >= 0 else 'Noise'}."
    )

    return {
        "contributions": [{"feature": feature, "value": round(float(value), 4)} for feature, value in top],
        "summary_text": summary,
        "cluster_label": target_cluster,
    }


def build_cluster_summary(df: pd.DataFrame, labels: np.ndarray) -> list[dict[str, Any]]:
    summary = []
    total_count = len(df) if len(df) > 0 else 1
    num_cols = [c for c in df.columns if str(c).lower() != "customer_id" and pd.api.types.is_numeric_dtype(df[c])]
    
    for label in sorted(set(labels.tolist())):
        indices = np.where(labels == label)[0]
        subset = df.iloc[indices]
        feature_means = {col: round(float(subset[col].mean()), 2) for col in num_cols if len(subset) > 0}
        
        item: dict[str, Any] = {
            "cluster_label": int(label),
            "name": f"Cluster {int(label) + 1}" if int(label) >= 0 else "Noise / Outliers",
            "count": int(len(subset)),
            "percentage": round(float(len(subset) / total_count * 100), 1),
            "avg_satisfaction": feature_means.get("satisfaction", feature_means.get("Satisfaction", 0.0)),
            "feature_averages": feature_means,
        }
        
        # Convenience aliases
        for alias in ["income", "Income", "annual_income"]:
            if alias in feature_means:
                item["avg_income"] = feature_means[alias]
        for alias in ["age", "Age"]:
            if alias in feature_means:
                item["avg_age"] = feature_means[alias]
        for alias in ["purchases", "Purchases", "total_purchases", "purchase_count", "spending"]:
            if alias in feature_means:
                item["avg_purchases"] = feature_means[alias]

        summary.append(item)
    return summary


def to_points(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [sanitize_for_json(r) for r in records]
