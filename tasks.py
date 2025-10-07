"""
Coda Development Workflow Tasks.

This module provides comprehensive task automation for the Coda AI-powered
code orchestration and testing system. Tasks cover development, testing,
deployment, and maintenance workflows.
"""

from typing import Any

from invoke import task


@task
def install(c: Any) -> None:
    """
    Install production dependencies.

    Installs the Coda package and all production dependencies using uv.
    This is the standard installation for production deployments.
    """
    c.run("uv pip install -e .")


@task
def install_dev(c: Any, name="install-dev") -> None:
    """
    Install development dependencies.

    Installs the Coda package with all development dependencies including
    testing frameworks, linting tools, and development utilities.
    """
    c.run("uv pip install -e '.[dev]'")


@task
def test(c: Any) -> None:
    """
    Execute comprehensive test suite.

    Runs all unit and integration tests with proper error reporting
    and test discovery across the entire codebase.
    """
    c.run("pytest")


@task
def test_unit(c: Any, name="test-unit") -> None:
    """
    Execute unit test suite.

    Runs only unit tests, excluding integration tests for faster
    feedback during development cycles.
    """
    c.run("pytest tests/unit -m 'not integration'")


@task
def test_integration(c: Any, name="test-integration") -> None:
    """
    Execute integration test suite.

    Runs only integration tests that require external dependencies
    or system-level interactions.
    """
    c.run("pytest tests/integration -m integration")


@task
def test_coverage(c: Any, name="test-coverage") -> None:
    """
    Execute test suite with coverage analysis.

    Runs all tests and generates comprehensive coverage reports
    in both HTML and terminal formats for code quality assessment.
    """
    c.run("pytest --cov=src/coda --cov-report=html --cov-report=term-missing")


@task
def lint(c: Any) -> None:
    """
    Execute code quality and style checks.

    Runs comprehensive linting using Ruff for Python code analysis
    and MyPy for static type checking to ensure code quality standards.
    """
    c.run("ruff check src/ tests/")
    c.run("mypy src/")


@task
def format_code(c: Any) -> None:
    """
    Format codebase with automated code formatting.

    Applies consistent code formatting using Black, isort, and Ruff
    to ensure uniform code style across the entire codebase.
    """
    c.run("black src/ tests/")
    c.run("isort src/ tests/")
    c.run("ruff check --fix src/ tests/")


@task
def pre_commit_install(c: Any) -> None:
    """
    Install pre-commit hooks for automated code quality.

    Sets up pre-commit hooks for both commit and push operations
    to ensure code quality standards are maintained automatically.
    """
    print("Installing pre-commit hooks...")
    c.run("pre-commit install")
    c.run("pre-commit install --hook-type pre-push")
    print("Pre-commit hooks installed successfully!")


@task
def pre_commit_run(c: Any, all_files=False) -> None:
    """
    Execute pre-commit hooks manually.

    Runs pre-commit hooks on staged files or all files if specified.
    Useful for testing hook configuration or running hooks outside of git.
    """
    print("Executing pre-commit hooks...")
    if all_files:
        c.run("pre-commit run --all-files")
    else:
        c.run("pre-commit run")
    print("Pre-commit hooks completed successfully!")


@task
def pre_commit_update(c: Any) -> None:
    """
    Update pre-commit hooks to latest versions.

    Updates all pre-commit hooks to their latest versions to ensure
    compatibility with current development tools and security patches.
    """
    print("Updating pre-commit hooks to latest versions...")
    c.run("pre-commit autoupdate")
    print("Pre-commit hooks updated successfully!")


@task
def clean(c: Any) -> None:
    """
    Clean up generated files and build artifacts.

    Removes all generated files, build artifacts, cache directories,
    and temporary files to ensure a clean development environment.
    """
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
def run(c: Any) -> None:
    """
    Start the Coda production server.

    Launches the Coda FastAPI server with production configuration
    for handling AI-powered code orchestration requests.
    """
    c.run("python main.py")


@task
def workflow_demo(c: Any) -> None:
    """
    Execute the complete Coda workflow demonstration.

    Runs the end-to-end demonstration showing the full Coda workflow
    including repository analysis, code generation, and testing.
    """
    c.run("python scripts/demo_workflow.py")


@task
def repo_analysis_demo(
    c: Any,
    repo_url: str = "https://github.com/varunbiluri/coda.git",
    branch: str = "main",
    query: str = "analyze the repo and provide me highlevel summary",
    name="repo-analysis-demo",
) -> None:
    """
    Execute repository analysis demonstration.

    Demonstrates the repository analysis system focused on
    the git repo summary use case with intelligent Git strategies and
    enhanced LLM summarization.

    Args:
        repo_url: Git repository URL to analyze
        branch: Git branch to analyze (default: main)
        query: Analysis query/focus
    """
    c.run(
        f"python scripts/demo_repo_analysis.py --repo-url {repo_url} --branch {branch} --query '{query}'"
    )


@task
def dev(c: Any) -> None:
    """
    Start development server with auto-reload.

    Launches the Coda FastAPI server in development mode with automatic
    reloading for rapid development and testing cycles.
    """
    c.run("uvicorn src.coda.main:app --reload --host 0.0.0.0 --port 8000")


@task
def docker_build(c: Any, name="docker-build") -> None:
    """
    Build Docker image for testing environment.

    Creates a Docker image containing the testing sandbox environment
    for isolated code execution and testing workflows.
    """
    c.run("docker build -t coda-sandbox -f sandbox/Dockerfile sandbox/")


@task
def docker_run(c: Any, name="docker-run") -> None:
    """
    Execute tests in Docker container.

    Runs the testing environment inside the Docker container to ensure
    isolated and reproducible test execution.
    """
    c.run("docker run --rm -it coda-sandbox")


@task
def check_structure(c: Any, name="check-structure") -> None:
    """
    Display project structure overview.

    Shows the current project structure excluding cache files and
    temporary directories for project organization verification.
    """
    print("Project structure:")
    c.run("tree -I '__pycache__|*.pyc|.git|.venv|node_modules' -L 3 || ls -la")


@task
def check_runs(c: Any, name="check-runs") -> None:
    """
    Display current execution runs.

    Lists all current execution runs stored in the .runs directory
    for monitoring and debugging purposes.
    """
    c.run("ls -la .runs/ 2>/dev/null || echo 'No runs directory found'")


@task
def setup(c: Any) -> None:
    """
    Complete project setup and initialization.

    Performs comprehensive project setup including virtual environment
    creation, dependency installation, and Docker image building for
    a complete development environment.
    """
    print("Initializing Coda development environment...")
    c.run("uv venv")
    print("Virtual environment created successfully.")
    c.run("source .venv/bin/activate && uv pip install -e '.[dev]'")
    print("Development dependencies installed successfully.")
    c.run("docker build -t coda-sandbox -f sandbox/Dockerfile sandbox/")
    print("Docker testing environment built successfully.")
    print("Setup completed! Execute 'invoke run' to start the server.")
