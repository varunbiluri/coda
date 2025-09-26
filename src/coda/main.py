"""
Coda API Server - Production orchestrator for multi-agent code generation and testing.

This module provides the main FastAPI application that coordinates multiple specialized
agents to analyze, generate, apply, and test code changes in an automated workflow.
"""

import asyncio
import json
import logging
import shutil
import uuid
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, status
from git import Repo

from config.settings import LITELLM_MODEL, LITELLM_PROVIDER, USE_MOCK_LLM

from .core.graph import CodaGraph
from .core.indexer import RepositoryIndexer
from .core.llm_client import LLMClient, create_llm_client
from .core.models import RunContext, RunRequest, RunResult, RunStatus

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI application
app = FastAPI(
    title="Coda Multi-Agent Code Orchestration System",
    description="Production-grade AI system for automated code generation, testing, and deployment",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# Global component instances - initialized lazily to prevent startup issues
_indexer: RepositoryIndexer | None = None
_llm_client: LLMClient | None = None
_coda_graph: CodaGraph | None = None


def _initialize_components() -> tuple[RepositoryIndexer, LLMClient, CodaGraph]:
    """
    Initialize and return the core system components.

    This function uses lazy initialization to prevent startup issues and ensures
    components are only created when needed.

    Returns:
        Tuple containing the repository indexer, LLM client, and orchestration graph
    """
    global _indexer, _llm_client, _coda_graph

    if _indexer is None:
        logger.info("Initializing system components...")
        _indexer = RepositoryIndexer()
        _llm_client = create_llm_client(use_mock=USE_MOCK_LLM, model=LITELLM_MODEL, provider=LITELLM_PROVIDER)
        _coda_graph = CodaGraph(_indexer, _llm_client)

        # Log which LLM client is being used
        if USE_MOCK_LLM:
            client_type = "Mock"
        else:
            client_type = type(_llm_client).__name__.replace("LLMClient", "")
        logger.info(f"System components initialized successfully (LLM: {client_type})")

    # Type assertions to ensure components are initialized
    assert _indexer is not None
    assert _llm_client is not None
    assert _coda_graph is not None

    return _indexer, _llm_client, _coda_graph


def _cleanup_old_runs(runs_dir: Path, current_run_id: str) -> None:
    """
    Clean up old runs, keeping only the last run and current run.

    This function removes all run directories except for the most recent
    completed run and the current run being created.

    Args:
        runs_dir: Directory containing all run folders
        current_run_id: ID of the current run being created
    """
    if not runs_dir.exists():
        return

    try:
        # Get all run directories (excluding current run)
        run_dirs = [d for d in runs_dir.iterdir() if d.is_dir() and d.name != current_run_id]

        if len(run_dirs) <= 1:
            # Keep at most 1 old run + current run = 2 total
            return

        # Sort by modification time (most recent first)
        run_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        # Keep only the most recent old run, remove the rest
        # This ensures we have at most: 1 most recent old run + 1 current run = 2 total
        runs_to_remove = run_dirs[1:]  # Skip the most recent old run

        for run_dir in runs_to_remove:
            logger.info(f"Cleaning up old run: {run_dir.name}")
            shutil.rmtree(run_dir)

        logger.info(f"Cleaned up {len(runs_to_remove)} old runs")

    except Exception as e:
        logger.warning(f"Failed to cleanup old runs: {e}")
        # Don't fail the main workflow if cleanup fails


@app.post("/runs", status_code=status.HTTP_201_CREATED)
async def create_run(request: RunRequest) -> dict[str, Any]:
    """
    Execute a complete multi-agent workflow for code generation and testing.

    This endpoint orchestrates the following workflow:
    1. Repository analysis and indexing
    2. Task planning based on the specified goal
    3. Code generation and diff creation
    4. Patch application to a new git branch
    5. Automated testing in an isolated Docker environment

    Args:
        request: Workflow request containing goal, repository path, and target branch

    Returns:
        Execution result with run ID, status, and detailed outcome information

    Raises:
        HTTPException: If the workflow fails or encounters an unrecoverable error
    """
    # Generate run ID
    run_id = uuid.uuid4()

    # Setup run context
    runs_dir = Path(".runs")
    runs_dir.mkdir(exist_ok=True)

    # Clean up old runs before creating new one
    _cleanup_old_runs(runs_dir, str(run_id))

    run_dir = runs_dir / str(run_id)
    run_dir.mkdir(exist_ok=True)

    workspace_path = run_dir / "workspace"
    log_path = run_dir / "log.json"
    result_path = run_dir / "result.json"

    run_context = RunContext(
        run_id=run_id,
        goal=request.goal,
        repo_path=request.repo_path,
        branch=request.branch,
        workspace_path=str(workspace_path),
        log_path=str(log_path),
        result_path=str(result_path),
    )

    try:
        # Initialize system components
        indexer, llm_client, coda_graph = _initialize_components()

        # Get event loop for running blocking operations in thread pool
        loop = asyncio.get_event_loop()

        # Setup workspace with repository clone
        logger.info(f"Setting up workspace for run {run_id}")
        await _setup_workspace(request.repo_path, workspace_path, request.branch)

        # Index repository content (executed in thread pool to prevent blocking)
        logger.info("Indexing repository content...")
        await loop.run_in_executor(None, indexer.clear_collection)
        await loop.run_in_executor(None, indexer.ingest_repository, str(workspace_path))

        # Execute multi-agent workflow (executed in thread pool to prevent blocking)
        logger.info("Starting multi-agent workflow execution...")
        final_state = await loop.run_in_executor(
            None, coda_graph.execute, run_context, request.goal
        )

        # Determine final status based on workflow completion
        if "error" in final_state and final_state["error"]:
            status = RunStatus.FAILED
            error_message = final_state["error"]
        else:
            tester_output = final_state.get("tester_output")
            if tester_output:
                # Workflow completed - check test results
                if tester_output.success:
                    status = RunStatus.SUCCESS
                    error_message = None
                else:
                    status = RunStatus.FAILED
                    error_message = "Tests failed"
            else:
                # Workflow didn't complete - determine where it failed
                if not final_state.get("planner_spec"):
                    status = RunStatus.FAILED
                    error_message = "Planning phase failed"
                elif not final_state.get("coder_output"):
                    status = RunStatus.FAILED
                    error_message = "Code generation phase failed"
                elif not final_state.get("apply_patch_output"):
                    status = RunStatus.FAILED
                    error_message = "Patch application phase failed"
                else:
                    status = RunStatus.FAILED
                    error_message = "Testing phase failed"

        # Create result
        result = RunResult(
            run_id=run_id,
            status=status,
            planner_spec=final_state.get("planner_spec"),
            coder_output=final_state.get("coder_output"),
            apply_patch_output=final_state.get("apply_patch_output"),
            tester_output=final_state.get("tester_output"),
            error_message=error_message,
        )

        # Write result to file
        with open(result_path, "w") as f:
            json.dump(
                {
                    "run_id": str(result.run_id),
                    "status": result.status.value,
                    "error_message": result.error_message,
                    "planner_spec": (
                        {
                            "tasks": (result.planner_spec.tasks if result.planner_spec else []),
                            "context": (result.planner_spec.context if result.planner_spec else ""),
                            "estimated_changes": (
                                result.planner_spec.estimated_changes if result.planner_spec else []
                            ),
                        }
                        if result.planner_spec
                        else None
                    ),
                    "coder_output": (
                        {
                            "diff": (result.coder_output.diff if result.coder_output else ""),
                            "commit_message": (
                                result.coder_output.commit_message if result.coder_output else ""
                            ),
                            "explanation": (
                                result.coder_output.explanation if result.coder_output else ""
                            ),
                        }
                        if result.coder_output
                        else None
                    ),
                    "apply_patch_output": (
                        {
                            "commit_hash": (
                                result.apply_patch_output.commit_hash
                                if result.apply_patch_output
                                else ""
                            ),
                            "branch_name": (
                                result.apply_patch_output.branch_name
                                if result.apply_patch_output
                                else ""
                            ),
                            "success": (
                                result.apply_patch_output.success
                                if result.apply_patch_output
                                else False
                            ),
                            "error_message": (
                                result.apply_patch_output.error_message
                                if result.apply_patch_output
                                else None
                            ),
                        }
                        if result.apply_patch_output
                        else None
                    ),
                    "tester_output": (
                        {
                            "success": (
                                result.tester_output.success if result.tester_output else False
                            ),
                            "stdout": (result.tester_output.stdout if result.tester_output else ""),
                            "stderr": (result.tester_output.stderr if result.tester_output else ""),
                            "exit_code": (
                                result.tester_output.exit_code if result.tester_output else 1
                            ),
                        }
                        if result.tester_output
                        else None
                    ),
                },
                f,
                indent=2,
            )

        return {
            "run_id": str(run_id),
            "status": status.value,
            "result_path": str(result_path),
        }

    except Exception as e:
        logger.error(f"Workflow execution failed for run {run_id}: {str(e)}")

        # Write error result to file
        error_result = {
            "run_id": str(run_id),
            "status": RunStatus.FAILED.value,
            "error_message": str(e),
        }

        with open(result_path, "w") as f:
            json.dump(error_result, f, indent=2)

        raise HTTPException(
            status_code=500,
            detail=f"Workflow execution failed: {str(e)}",
        ) from e


@app.get("/health")
async def health_check() -> dict[str, str]:
    """
    System health check endpoint.

    Returns:
        Health status information
    """
    return {"status": "healthy", "service": "coda-orchestrator"}


async def _setup_workspace(repo_path: str, workspace_path: Path, branch: str) -> None:
    """
    Setup workspace by cloning or copying the target repository.

    This function prepares an isolated workspace for workflow execution by
    either cloning a git repository or copying a local directory.

    Args:
        repo_path: Path to the source repository (local path or git URL)
        workspace_path: Destination path for the workspace
        branch: Git branch to checkout (for git repositories)

    Raises:
        ValueError: If the source repository path does not exist
        Exception: If repository setup fails
    """
    repo_path_obj = Path(repo_path)

    if not repo_path_obj.exists():
        raise ValueError(f"Repository path does not exist: {repo_path}")

    # Copy repository to workspace
    shutil.copytree(repo_path_obj, workspace_path)

    # Initialize git if not already a git repo
    try:
        repo = Repo(workspace_path)
    except Exception:
        repo = Repo.init(workspace_path)

        # Add all files and make initial commit
        repo.git.add(A=True)
        repo.index.commit("Initial commit")

    # Checkout specified branch if it exists
    try:
        repo.git.checkout(branch)
    except Exception:
        # Branch doesn't exist, stay on current branch
        pass


def main():
    """Run the FastAPI server."""
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
