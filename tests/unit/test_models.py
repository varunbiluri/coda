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
        """Test creating a RunRequest."""
        request = RunRequest(goal="Add health endpoint", repo_path="/path/to/repo", branch="main")
        assert request.goal == "Add health endpoint"
        assert request.repo_path == "/path/to/repo"
        assert request.branch == "main"


class TestRunContext:
    """Test cases for RunContext model."""

    def test_run_context_creation(self):
        """Test creating a RunContext."""
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


class TestRunStatus:
    """Test cases for RunStatus enum."""

    def test_run_status_values(self):
        """Test RunStatus enum values."""
        assert RunStatus.PENDING.value == "pending"
        assert RunStatus.RUNNING.value == "running"
        assert RunStatus.SUCCESS.value == "success"
        assert RunStatus.FAILED.value == "failed"


class TestPlannerSpec:
    """Test cases for PlannerSpec model."""

    def test_planner_spec_creation(self):
        """Test creating a PlannerSpec."""
        tasks = [{"id": "task1", "description": "Test task"}]
        spec = PlannerSpec(
            tasks=tasks,
            context="Test context",
            estimated_changes=["Change 1", "Change 2"],
        )
        assert spec.tasks == tasks
        assert spec.context == "Test context"
        assert len(spec.estimated_changes) == 2


class TestCoderOutput:
    """Test cases for CoderOutput model."""

    def test_coder_output_creation(self):
        """Test creating a CoderOutput."""
        output = CoderOutput(
            diff="--- a/file.py\n+++ b/file.py",
            commit_message="Add feature",
            explanation="Added new feature",
        )
        assert "--- a/file.py" in output.diff
        assert output.commit_message == "Add feature"
        assert output.explanation == "Added new feature"


class TestApplyPatchOutput:
    """Test cases for ApplyPatchOutput model."""

    def test_apply_patch_output_success(self):
        """Test creating a successful ApplyPatchOutput."""
        output = ApplyPatchOutput(
            commit_hash="abc123",
            branch_name="feature-branch",
            success=True,
            error_message=None,
        )
        assert output.commit_hash == "abc123"
        assert output.branch_name == "feature-branch"
        assert output.success is True
        assert output.error_message is None

    def test_apply_patch_output_failure(self):
        """Test creating a failed ApplyPatchOutput."""
        output = ApplyPatchOutput(
            commit_hash="",
            branch_name="",
            success=False,
            error_message="Git apply failed",
        )
        assert output.success is False
        assert output.error_message == "Git apply failed"


class TestTesterOutput:
    """Test cases for TesterOutput model."""

    def test_tester_output_success(self):
        """Test creating a successful TesterOutput."""
        output = TesterOutput(success=True, stdout="All tests passed", stderr="", exit_code=0)
        assert output.success is True
        assert output.stdout == "All tests passed"
        assert output.exit_code == 0

    def test_tester_output_failure(self):
        """Test creating a failed TesterOutput."""
        output = TesterOutput(success=False, stdout="", stderr="Test failed", exit_code=1)
        assert output.success is False
        assert output.stderr == "Test failed"
        assert output.exit_code == 1
