from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class UserCreate(BaseModel):
    name: str
    email: str
    password: str


class UserOut(BaseModel):
    id: int
    name: str
    email: str
    role: str = "admin"


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


class UploadResponse(BaseModel):
    dataset_id: int
    filename: str
    row_count: int
    columns: list[str]
    preview: list[dict[str, Any]]


class StrategyUpdate(BaseModel):
    dataset_id: int | None = None
    title: str | None = None
    recommendation: str | None = None
    strategy: str | None = None

    @property
    def recommendation_text(self) -> str:
        return self.recommendation or self.strategy or ""

    @property
    def title_text(self) -> str:
        return self.title or "Strategy"


class CustomerListItem(BaseModel):
    customer_id: str
    cluster_label: int | None = None
    feature_values: dict[str, Any]
