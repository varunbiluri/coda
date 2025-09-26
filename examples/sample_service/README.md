# Sample Service

A minimal FastAPI service for testing the Coda system.

## Structure

- `app/main.py` - Main FastAPI application
- `tests/test_health.py` - Test that initially fails (missing /health endpoint)

## Running

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Testing

```bash
pytest
```

The test will initially fail because the `/health` endpoint is not implemented.
This is intentional to demonstrate the Coda system adding the missing endpoint.
