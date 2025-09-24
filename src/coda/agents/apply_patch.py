"""Apply Patch Agent - Applies code diffs to the repository."""

import logging
import subprocess
import tempfile
from pathlib import Path

from git import Repo

from ..core.models import ApplyPatchOutput, CoderOutput

logger = logging.getLogger(__name__)


class ApplyPatchAgent:
    """Agent that applies code diffs to the repository."""

    def __init__(self, workspace_path: str):
        """Initialize the apply patch agent.

        Args:
            workspace_path: Path to the workspace repository
        """
        self.workspace_path = Path(workspace_path)

    def execute(self, coder_output: CoderOutput, run_id: str) -> ApplyPatchOutput:
        """Execute patch application.

        Args:
            coder_output: Output from the coder agent
            run_id: Run ID for branch naming

        Returns:
            Apply patch output with commit hash and success status
        """
        logger.info("Applying code changes")

        try:
            # Validate diff format
            if "@@" not in coder_output.diff:
                logger.error("Invalid diff format - missing @@ markers")
                return ApplyPatchOutput(
                    commit_hash="",
                    branch_name="",
                    success=False,
                    error_message="Invalid diff format: missing @@ markers",
                )

            # Ensure workspace path exists and is a git repo
            if not self.workspace_path.exists():
                logger.error(f"Workspace path does not exist: {self.workspace_path}")
                return ApplyPatchOutput(
                    commit_hash="",
                    branch_name="",
                    success=False,
                    error_message=f"Workspace path does not exist: {self.workspace_path}",
                )

            # Initialize git repo
            try:
                repo = Repo(self.workspace_path)
            except Exception as e:
                logger.error(f"Not a git repository: {e}")
                return ApplyPatchOutput(
                    commit_hash="",
                    branch_name="",
                    success=False,
                    error_message=f"Not a git repository: {e}",
                )

            # Create and switch to new branch
            branch_name = f"run/{run_id}"
            logger.info(f"Creating branch '{branch_name}'")

            new_branch = repo.create_head(branch_name)
            new_branch.checkout()

            # Write diff to temporary file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".patch", delete=False) as f:
                f.write(coder_output.diff)
                patch_file = f.name

            try:
                # Apply patch using git apply
                logger.info("Applying patch")
                subprocess.run(
                    ["git", "apply", patch_file],
                    cwd=self.workspace_path,
                    capture_output=True,
                    text=True,
                    check=True,
                )

                # Stage all changes
                repo.git.add(A=True)

                # Commit changes
                commit = repo.index.commit(coder_output.commit_message)

                logger.info("Patch applied successfully")
                logger.info(f"Commit: {commit.hexsha[:8]}")

                return ApplyPatchOutput(
                    commit_hash=commit.hexsha, branch_name=branch_name, success=True
                )

            except subprocess.CalledProcessError as e:
                logger.error(f"Git apply failed: {e.stderr}")
                return ApplyPatchOutput(
                    commit_hash="",
                    branch_name=branch_name,
                    success=False,
                    error_message=f"Git apply failed: {e.stderr}",
                )
            finally:
                # Clean up patch file
                Path(patch_file).unlink(missing_ok=True)

        except Exception as e:
            logger.error(f"Apply patch failed: {str(e)}")
            return ApplyPatchOutput(
                commit_hash="",
                branch_name="",
                success=False,
                error_message=f"Patch application failed: {str(e)}",
            )
