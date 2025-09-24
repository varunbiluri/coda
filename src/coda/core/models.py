"""Data models for the Coda system."""

from dataclasses import dataclass
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel


class RunStatus(str, Enum):
    """Status of a run."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


class RunRequest(BaseModel):
    """Request model for starting a run."""

    goal: str
    repo_path: str
    branch: str = "main"


@dataclass
class RunContext:
    """Context for a run execution."""

    run_id: UUID
    goal: str
    repo_path: str
    branch: str
    workspace_path: str
    log_path: str
    result_path: str


@dataclass
class PlannerSpec:
    """Specification from the planner."""

    tasks: list[dict[str, Any]]
    context: str
    estimated_changes: list[str]


@dataclass
class CoderOutput:
    """Output from the coder."""

    diff: str
    commit_message: str
    explanation: str


@dataclass
class ApplyPatchOutput:
    """Output from applying a patch."""

    commit_hash: str
    branch_name: str
    success: bool
    error_message: str | None = None


@dataclass
class TesterOutput:
    """Output from the tester."""

    success: bool
    stdout: str
    stderr: str
    exit_code: int


@dataclass
class RunResult:
    """Final result of a run."""

    run_id: UUID
    status: RunStatus
    planner_spec: PlannerSpec | None
    coder_output: CoderOutput | None
    apply_patch_output: ApplyPatchOutput | None
    tester_output: TesterOutput | None
    error_message: str | None = None
