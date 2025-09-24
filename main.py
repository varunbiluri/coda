#!/usr/bin/env python3
"""
Coda - AI-powered code orchestration and testing system.

Main entry point for the Coda server application.
"""

import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.coda.main import app

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
