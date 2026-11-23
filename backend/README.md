# Intro to this RAG System

- Setup Ruff and Its Keys - Done
```py
uv run ruff check .
uv run ruff check . --fix
uv run ruff format .
```

- Setup Alembic and its Keys - Done
```py
# TODO -- Env Updates Remaning
# Edit migrations/env.py 

# Alembic Commands
alembic revision --autogenerate -m "initial migration"
alembic upgrade head

```

<!-- AI Related Things -->
- Add Logging - Initialize FastAPI Logging & Add Request Middleware (app/main.py) - Done
- Add Fastapi / SQLModel - Done

- Add AI Related Code / Test It
- Add Pinecone - for Vector Store
- Add LangSmtih - for Observability

<!-- Backend Related Things -->
- Add Supabase - for Database / Auth / Storage
- Add Inngest - for Background Jobs
- Add Sentry  - for Error Tracking


<!-- Skills to Added -->
- Supabase Skills -- npx skills add supabase/agent-skills