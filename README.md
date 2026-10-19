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

## Vercel deployment

Set the Vercel project Root Directory to the repository root (`.`), where both
`package.json` and `requirements.txt` live. `vercel.json` builds the Vite
frontend into `dist` and routes `/api/*` to the FastAPI function in
`api/index.py`; `.python-version` (and `backend/.python-version`) selects Python 3.12.

Configure `DATABASE_URL` and a strong `SECRET_KEY` in Vercel's environment
variables. Both `psycopg[binary]==3.2.1` (for `postgresql+psycopg://`) and
`psycopg2-binary==2.9.9` (for standard `postgresql://`) are supported and pinned
for Python 3.12. Do not commit environment files or credentials. The frontend
uses same-origin `/api` requests in production; set `VITE_API_BASE_URL` only if
the API is hosted on a different origin.

## Docker Compose

Run `docker compose up --build` to start PostgreSQL, apply migrations before FastAPI, and serve the Vite build through Nginx at `http://localhost:8080`. The API is available at `http://localhost:8000`.

The frontend and backend are separate containers so static assets can be served efficiently while the API scales independently.

## Tests

Backend tests use a throwaway SQLite database. From `backend`, run `python -m pytest tests -q`. Production deployments use PostgreSQL through `DATABASE_URL`.
