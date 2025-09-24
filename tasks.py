"""Invoke tasks for Coda development workflow."""

from invoke import task


@task
def install(c):
    """Install production dependencies."""
    c.run("uv pip install -e .")


@task
def install_dev(c, name="install-dev"):
    """Install development dependencies."""
    c.run("uv pip install -e '.[dev]'")


@task
def test(c):
    """Run all tests."""
    c.run("pytest")


@task
def test_unit(c, name="test-unit"):
    """Run unit tests only."""
    c.run("pytest tests/unit -m 'not integration'")


@task
def test_integration(c, name="test-integration"):
    """Run integration tests only."""
    c.run("pytest tests/integration -m integration")


@task
def test_coverage(c, name="test-coverage"):
    """Run tests with coverage report."""
    c.run("pytest --cov=src/coda --cov-report=html --cov-report=term-missing")


@task
def lint(c):
    """Run linting checks."""
    c.run("ruff check src/ tests/")
    c.run("mypy src/")


@task
def format(c):
    """Format code with black and isort."""
    c.run("black src/ tests/")
    c.run("isort src/ tests/")
    c.run("ruff check --fix src/ tests/")


@task
def pre_commit_install(c):
    """Install pre-commit hooks."""
    print("Installing pre-commit hooks...")
    c.run("pre-commit install")
    c.run("pre-commit install --hook-type pre-push")
    print("Pre-commit hooks installed!")


@task
def pre_commit_run(c, all_files=False):
    """Run pre-commit hooks."""
    print("Running pre-commit hooks...")
    if all_files:
        c.run("pre-commit run --all-files")
    else:
        c.run("pre-commit run")
    print("Pre-commit hooks completed!")


@task
def pre_commit_update(c):
    """Update pre-commit hooks to latest versions."""
    print("Updating pre-commit hooks...")
    c.run("pre-commit autoupdate")
    print("Pre-commit hooks updated!")


@task
def clean(c):
    """Clean up generated files."""
    c.run("find . -type f -name '*.pyc' -delete")
    c.run("find . -type d -name '__pycache__' -delete")
    c.run("find . -type d -name '*.egg-info' -exec rm -rf {} + || true")
    c.run("rm -rf build/")
    c.run("rm -rf dist/")
    c.run("rm -rf .coverage")
    c.run("rm -rf htmlcov/")
    c.run("rm -rf .pytest_cache/")
    c.run("rm -rf .mypy_cache/")


@task
def run(c):
    """Start the Coda server."""
    c.run("python main.py")


@task
def demo(c):
    """Run the demo workflow."""
    c.run("python scripts/demo_run.py")


@task
def dev(c):
    """Development server with auto-reload."""
    c.run("uvicorn src.coda.main:app --reload --host 0.0.0.0 --port 8000")


@task
def docker_build(c, name="docker-build"):
    """Build Docker image for testing."""
    c.run("docker build -t coda-sandbox -f sandbox/Dockerfile sandbox/")


@task
def docker_run(c, name="docker-run"):
    """Run tests in Docker container."""
    c.run("docker run --rm -it coda-sandbox")


@task
def check_structure(c, name="check-structure"):
    """Check project structure."""
    print("Project structure:")
    c.run("tree -I '__pycache__|*.pyc|.git|.venv|node_modules' -L 3 || ls -la")


@task
def check_runs(c, name="check-runs"):
    """Check current runs."""
    c.run("ls -la .runs/ 2>/dev/null || echo 'No runs directory found'")


@task
def setup(c):
    """Complete project setup."""
    print("Setting up Coda development environment...")
    c.run("uv venv")
    print("Virtual environment created.")
    c.run("source .venv/bin/activate && uv pip install -e '.[dev]'")
    print("Dependencies installed.")
    c.run("docker build -t coda-sandbox -f sandbox/Dockerfile sandbox/")
    print("Docker image built.")
    print("Setup complete! Run 'invoke run' to start the server.")
