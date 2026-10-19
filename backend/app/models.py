from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from app.db.session import Base


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(50), default="admin")
    created_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)

    datasets = relationship("Dataset", back_populates="owner")


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    filename = Column(String(255), nullable=False)
    uploaded_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)
    row_count = Column(Integer, default=0)
    columns = Column(JSON, nullable=False, default=list)

    owner = relationship("User", back_populates="datasets")
    customers = relationship("Customer", back_populates="dataset")
    cluster_runs = relationship("ClusterRun", back_populates="dataset")
    strategies = relationship("Strategy", back_populates="dataset")


class Customer(Base):
    __tablename__ = "customers"

    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False)
    customer_id = Column(String(64), nullable=False)
    feature_values = Column(JSON, nullable=False, default=dict)
    raw_row = Column(JSON, nullable=False, default=dict)

    dataset = relationship("Dataset", back_populates="customers")
    cluster_assignments = relationship("ClusterAssignment", back_populates="customer")


class ClusterRun(Base):
    __tablename__ = "cluster_runs"

    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False)
    algorithm = Column(String(30), nullable=False)
    k = Column(Integer, nullable=True)
    params = Column(JSON, nullable=False, default=dict)
    silhouette_score = Column(Float, nullable=False)
    n_clusters = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)

    dataset = relationship("Dataset", back_populates="cluster_runs")
    assignments = relationship("ClusterAssignment", back_populates="cluster_run")


class ClusterAssignment(Base):
    __tablename__ = "cluster_assignments"

    id = Column(Integer, primary_key=True, index=True)
    cluster_run_id = Column(Integer, ForeignKey("cluster_runs.id"), nullable=False)
    customer_id = Column(Integer, ForeignKey("customers.id"), nullable=False)
    cluster_label = Column(Integer, nullable=False)
    proj_x = Column(Float, nullable=True)
    proj_y = Column(Float, nullable=True)

    cluster_run = relationship("ClusterRun", back_populates="assignments")
    customer = relationship("Customer", back_populates="cluster_assignments")


class Strategy(Base):
    __tablename__ = "strategies"

    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False)
    cluster_label = Column(Integer, nullable=False)
    title = Column(String(255), nullable=False)
    recommendation = Column(Text, nullable=False)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)

    dataset = relationship("Dataset", back_populates="strategies")
