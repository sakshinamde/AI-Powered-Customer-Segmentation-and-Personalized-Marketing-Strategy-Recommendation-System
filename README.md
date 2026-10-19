# SmartSeg AI

SmartSeg AI uses the FastAPI backend in `backend/app` as its API and PostgreSQL as its production database.

## Local development

Prerequisites: Node.js, Python 3.12+, and PostgreSQL (or Docker).

1. Copy `.env.example` to `.env` and set a random `SECRET_KEY`.
2. Install dependencies with `npm install` and `pip install -r requirements.txt`.
3. Start the database, then apply migrations from the repository root with `alembic upgrade head`.
4. Start FastAPI from `backend` with `uvicorn app.main:app --reload --port 8000`.
5. Start Vite in another terminal with `npm run dev`.

The frontend defaults to `http://localhost:8000`; override it with `VITE_API_BASE_URL`.

## Docker Compose

Run `docker compose up --build` to start PostgreSQL, apply migrations before FastAPI, and serve the Vite build through Nginx at `http://localhost:8080`. The API is available at `http://localhost:8000`.

The frontend and backend are separate containers so static assets can be served efficiently while the API scales independently.

## Tests

Backend tests use a throwaway SQLite database. From `backend`, run `python -m pytest tests -q`. Production deployments use PostgreSQL through `DATABASE_URL`.
