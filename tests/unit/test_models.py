"""Unit tests for Coda models."""

from uuid import UUID

import pytest

from src.coda.core.models import (
    ApplyPatchOutput,
    CoderOutput,
    PlannerSpec,
    RunContext,
    RunRequest,
    RunResult,
    RunStatus,
    TesterOutput,
)


class TestRunRequest:
    """Test cases for RunRequest model."""

    def test_run_request_creation(self):
        """Test creating a RunRequest with valid data."""
        request = RunRequest(
            goal="Add health endpoint", repo_path="/path/to/repo", branch="main"
        )
        assert request.goal == "Add health endpoint"
        assert request.repo_path == "/path/to/repo"
        assert request.branch == "main"

    def test_run_request_validation(self):
        """Test RunRequest with different branch names."""
        # Test with different branch names
        request = RunRequest(goal="Fix bug", repo_path="/repo", branch="develop")
        assert request.branch == "develop"

        request = RunRequest(
            goal="Add feature", repo_path="/repo", branch="feature/new-api"
        )
        assert request.branch == "feature/new-api"


class TestRunContext:
    """Test cases for RunContext model."""

    def test_run_context_creation(self):
        """Test creating a RunContext with all required fields."""
        run_id = UUID("12345678-1234-5678-9012-123456789012")
        context = RunContext(
            run_id=run_id,
            goal="Test goal",
            repo_path="/repo",
            branch="main",
            workspace_path="/workspace",
            log_path="/log.json",
            result_path="/result.json",
        )
        assert context.run_id == run_id
        assert context.goal == "Test goal"
        assert context.workspace_path == "/workspace"
        assert context.log_path == "/log.json"
        assert context.result_path == "/result.json"

    def test_run_context_with_different_paths(self):
        """Test RunContext with various path configurations."""
        run_id = UUID("87654321-4321-8765-2109-876543210987")
        context = RunContext(
            run_id=run_id,
            goal="Complex task",
            repo_path="/home/user/projects/my-repo",
            branch="feature/ai-integration",
            workspace_path="/tmp/coda/workspace/123",
            log_path="/tmp/coda/logs/123.json",
            result_path="/tmp/coda/results/123.json",
        )
        assert context.repo_path == "/home/user/projects/my-repo"
        assert context.branch == "feature/ai-integration"


class TestRunStatus:
    """Test cases for RunStatus enum."""

    def test_run_status_values(self):
        """Test RunStatus enum values are correct."""
        assert RunStatus.PENDING.value == "pending"
        assert RunStatus.RUNNING.value == "running"
        assert RunStatus.SUCCESS.value == "success"
        assert RunStatus.FAILED.value == "failed"

    def test_run_status_completeness(self):
        """Test that all expected status values are present."""
        expected_statuses = {"pending", "running", "success", "failed"}
        actual_statuses = {status.value for status in RunStatus}
        assert actual_statuses == expected_statuses


class TestPlannerSpec:
    """Test cases for PlannerSpec model."""

    def test_planner_spec_creation(self):
        """Test creating a PlannerSpec with valid data."""
        tasks = [
            {
                "id": "task1",
                "description": "Add endpoint",
                "files_to_modify": ["app.py"],
                "priority": "high",
            },
            {
                "id": "task2",
                "description": "Add tests",
                "files_to_modify": ["test_app.py"],
                "priority": "medium",
            },
        ]
        spec = PlannerSpec(
            tasks=tasks,
            context="Adding new API endpoint with comprehensive testing",
            estimated_changes=[
                "Create new route handler",
                "Add unit tests",
                "Update documentation",
            ],
        )
        assert spec.tasks == tasks
        assert spec.context == "Adding new API endpoint with comprehensive testing"
        assert len(spec.estimated_changes) == 3
        assert "Create new route handler" in spec.estimated_changes

    def test_planner_spec_empty_tasks(self):
        """Test PlannerSpec with empty tasks list."""
        spec = PlannerSpec(
            tasks=[],
            context="No changes needed",
            estimated_changes=[],
        )
        assert spec.tasks == []
        assert spec.estimated_changes == []

    def test_planner_spec_complex_tasks(self):
        """Test PlannerSpec with complex task structures."""
        tasks = [
            {
                "id": "refactor_auth",
                "description": "Refactor authentication system",
                "files_to_modify": ["auth.py", "middleware.py", "models.py"],
                "priority": "high",
                "dependencies": ["update_database_schema"],
            },
            {
                "id": "add_middleware",
                "description": "Add request logging middleware",
                "files_to_modify": ["middleware.py"],
                "priority": "medium",
            },
        ]
        spec = PlannerSpec(
            tasks=tasks,
            context="Major authentication system overhaul",
            estimated_changes=["Refactor auth logic", "Add middleware", "Update tests"],
        )
        assert len(spec.tasks) == 2
        assert spec.tasks[0]["dependencies"] == ["update_database_schema"]


class TestCoderOutput:
    """Test cases for CoderOutput model."""

    def test_coder_output_creation(self):
        """Test creating a CoderOutput with valid diff."""
        diff = """--- a/app/main.py
+++ b/app/main.py
@@ -5,6 +5,10 @@ app = FastAPI(title="Sample Service")
 @app.get("/")
 def read_root():
     return {"message": "Hello World"}
+
+@app.get("/health")
+def health_check():
+    return {"status": "healthy"}"""

        output = CoderOutput(
            diff=diff,
            commit_message="Add health endpoint",
            explanation="Added a health check endpoint for service monitoring",
        )
        assert "--- a/app/main.py" in output.diff
        assert '@app.get("/health")' in output.diff
        assert output.commit_message == "Add health endpoint"
        assert (
            output.explanation == "Added a health check endpoint for service monitoring"
        )

    def test_coder_output_with_complex_diff(self):
        """Test CoderOutput with complex multi-file diff."""
        diff = """--- a/app/main.py
+++ b/app/main.py
@@ -1,3 +1,5 @@
 from fastapi import FastAPI
+from typing import Dict
+
 app = FastAPI(title="Sample Service")
@@ -5,6 +7,10 @@ app = FastAPI(title="Sample Service")
 @app.get("/")
 def read_root():
     return {"message": "Hello World"}
+
+@app.get("/health")
+def health_check() -> Dict[str, str]:
+    return {"status": "healthy"}
--- a/tests/test_health.py
+++ b/tests/test_health.py
@@ -0,0 +1,8 @@
+import pytest
+from fastapi.testclient import TestClient
+from app.main import app
+
+client = TestClient(app)
+
+def test_health_endpoint():
+    response = client.get("/health")
+    assert response.status_code == 200
+    assert response.json() == {"status": "healthy"}"""

        output = CoderOutput(
            diff=diff,
            commit_message="Add health endpoint with comprehensive tests",
            explanation="Implemented health check endpoint with proper typing and comprehensive test coverage",
        )
        assert "app/main.py" in output.diff
        assert "tests/test_health.py" in output.diff
        assert "def health_check() -> Dict[str, str]:" in output.diff


class TestApplyPatchOutput:
    """Test cases for ApplyPatchOutput model."""

    def test_apply_patch_output_success(self):
        """Test creating a successful ApplyPatchOutput."""
        output = ApplyPatchOutput(
            commit_hash="abc123def456",
            branch_name="feature/health-endpoint",
            success=True,
            error_message=None,
        )
        assert output.commit_hash == "abc123def456"
        assert output.branch_name == "feature/health-endpoint"
        assert output.success is True
        assert output.error_message is None

    def test_apply_patch_output_failure(self):
        """Test creating a failed ApplyPatchOutput."""
        output = ApplyPatchOutput(
            commit_hash="",
            branch_name="",
            success=False,
            error_message="Git apply failed: corrupt patch at line 15",
        )
        assert output.success is False
        assert output.error_message == "Git apply failed: corrupt patch at line 15"
        assert output.commit_hash == ""
        assert output.branch_name == ""

    def test_apply_patch_output_partial_success(self):
        """Test ApplyPatchOutput with partial success scenario."""
        output = ApplyPatchOutput(
            commit_hash="def456ghi789",
            branch_name="feature/partial-implementation",
            success=False,
            error_message="Some files applied successfully, but test file failed to apply",
        )
        assert output.commit_hash == "def456ghi789"
        assert output.success is False
        assert "Some files applied successfully" in output.error_message


class TestTesterOutput:
    """Test cases for TesterOutput model."""

    def test_tester_output_success(self):
        """Test creating a successful TesterOutput."""
        output = TesterOutput(
            success=True,
            stdout="test_health.py::test_health_endpoint PASSED\n1 passed in 0.05s",
            stderr="",
            exit_code=0,
        )
        assert output.success is True
        assert "PASSED" in output.stdout
        assert output.exit_code == 0
        assert output.stderr == ""

    def test_tester_output_failure(self):
        """Test creating a failed TesterOutput."""
        output = TesterOutput(
            success=False,
            stdout="",
            stderr="FAILED test_health.py::test_health_endpoint - AssertionError: Expected status code 200, got 404",
            exit_code=1,
        )
        assert output.success is False
        assert "FAILED" in output.stderr
        assert "AssertionError" in output.stderr
        assert output.exit_code == 1

    def test_tester_output_with_warnings(self):
        """Test TesterOutput with warnings but successful execution."""
        output = TesterOutput(
            success=True,
            stdout="test_health.py::test_health_endpoint PASSED\n1 passed in 0.05s",
            stderr="WARNING: Deprecated function used in test_health.py:12",
            exit_code=0,
        )
        assert output.success is True
        assert output.exit_code == 0
        assert "WARNING" in output.stderr

    def test_tester_output_timeout(self):
        """Test TesterOutput with timeout scenario."""
        output = TesterOutput(
            success=False,
            stdout="",
            stderr="ERROR: Test execution timed out after 30 seconds",
            exit_code=124,  # Standard timeout exit code
        )
        assert output.success is False
        assert output.exit_code == 124
        assert "timed out" in output.stderr


class TestRunResult:
    """Test cases for RunResult model."""

    def test_run_result_success(self):
        """Test creating a successful RunResult."""
        result = RunResult(
            run_id=UUID("12345678-1234-5678-9012-123456789012"),
            status=RunStatus.SUCCESS,
            planner_spec={"tasks": [{"id": "task1", "description": "Add endpoint"}]},
            coder_output={"diff": "--- a/file.py", "commit_message": "Add feature"},
            apply_patch_output={"success": True, "commit_hash": "abc123"},
            tester_output={"success": True, "exit_code": 0},
        )
        assert result.status == RunStatus.SUCCESS
        assert result.planner_spec["tasks"][0]["id"] == "task1"
        assert result.coder_output["commit_message"] == "Add feature"

    def test_run_result_failure(self):
        """Test creating a failed RunResult."""
        result = RunResult(
            run_id=UUID("87654321-4321-8765-2109-876543210987"),
            status=RunStatus.FAILED,
            error_message="Git apply failed: corrupt patch",
            planner_spec={"tasks": [{"id": "task1", "description": "Add endpoint"}]},
            coder_output={"diff": "invalid diff", "commit_message": "Add feature"},
            apply_patch_output={"success": False, "error_message": "Git apply failed"},
            tester_output=None,
        )
        assert result.status == RunStatus.FAILED
        assert "corrupt patch" in result.error_message
        assert result.apply_patch_output["success"] is False
