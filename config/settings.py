"""Configuration settings for Coda."""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
RUNS_DIR = PROJECT_ROOT / ".runs"

# Ensure directories exist
RUNS_DIR.mkdir(exist_ok=True)

# Server configuration
SERVER_HOST = os.getenv("CODA_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("CODA_PORT", "8000"))

# Database configuration
CHROMA_DB_PATH = PROJECT_ROOT / "chroma_db"
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "coda_repo")

# Docker configuration
DOCKER_IMAGE_NAME = os.getenv("DOCKER_IMAGE", "coda-sandbox")
DOCKER_NETWORK_MODE = os.getenv("DOCKER_NETWORK", "none")

# LLM configuration
USE_MOCK_LLM = os.getenv("USE_MOCK_LLM", "false").lower() in ("true", "1", "yes")

# OpenAI configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

# Azure OpenAI configuration
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
AZURE_OPENAI_MODEL = os.getenv("AZURE_OPENAI_MODEL", "gpt-35-turbo")

# Embedding configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Git configuration
DEFAULT_BRANCH = os.getenv("DEFAULT_BRANCH", "main")

# File extensions to index
INDEXABLE_EXTENSIONS = [".py", ".md", ".txt", ".yml", ".yaml", ".json", ".toml"]
