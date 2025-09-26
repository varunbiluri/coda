"""Sample FastAPI service for testing Coda."""

from fastapi import FastAPI

app = FastAPI(title="Sample Service", version="0.1.0")


@app.get("/")
def read_root():
    """Root endpoint."""
    return {"message": "Hello World"}
