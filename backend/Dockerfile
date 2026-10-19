FROM python:3.12-slim AS builder

WORKDIR /build
COPY requirements.txt /build/requirements.txt
RUN python -m venv /opt/venv \
	&& /opt/venv/bin/pip install --no-cache-dir --retries 5 --timeout 120 -r /build/requirements.txt

FROM python:3.12-slim AS runtime

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv

COPY backend /app/backend
COPY alembic.ini /app/alembic.ini
WORKDIR /app
ENV PYTHONPATH=/app/backend
ENV PATH="/opt/venv/bin:$PATH"
EXPOSE 8000
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
