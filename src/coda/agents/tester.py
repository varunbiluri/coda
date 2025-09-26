"""Tester Agent - Runs tests in a Docker sandbox."""

import logging
from pathlib import Path

import docker

from ..core.models import TesterOutput

logger = logging.getLogger(__name__)


class TesterAgent:
    """Agent that runs tests in a Docker sandbox."""

    def __init__(self, workspace_path: str):
        """Initialize the tester agent.

        Args:
            workspace_path: Path to the workspace repository
        """
        self.workspace_path = Path(workspace_path)
        self.docker_client = docker.from_env()

    def execute(self) -> TesterOutput:
        """Execute tests in Docker sandbox.

        Returns:
            Tester output with test results and status
        """
        logger.info("Running tests in Docker sandbox")

        try:
            # Build Docker image if it doesn't exist
            self._ensure_docker_image()

            # Run tests in container with network isolation
            logger.info("Executing pytest in container")

            # Convert to absolute path for Docker volume mounting
            abs_workspace_path = str(self.workspace_path.resolve())

            container = self.docker_client.containers.run(
                "coda-sandbox",
                command=["pytest", "-q"],
                volumes={abs_workspace_path: {"bind": "/workspace", "mode": "rw"}},
                working_dir="/workspace",
                network_mode="none",  # No network access for security
                remove=True,
                detach=False,
                stdout=True,
                stderr=True,
            )

            # Get output
            logs = (
                container.decode("utf-8")
                if isinstance(container, bytes)
                else str(container)
            )

            # Determine success based on pytest output
            success = (
                "FAILED" not in logs
                and "ERROR" not in logs
                and "failed" not in logs.lower()
            )

            if success:
                logger.info("Tests passed successfully")
            else:
                logger.warning("Tests failed")
                logger.debug(f"Test output: {logs[:200]}...")

            return TesterOutput(
                success=success, stdout=logs, stderr="", exit_code=0 if success else 1
            )

        except docker.errors.ContainerError as e:
            logger.error(f"Container error: {e}")
            # Extract information from ContainerError safely
            stdout = ""
            stderr = str(e)
            exit_code = 1

            # Try to extract stdout/stderr if available
            if hasattr(e, "stdout") and e.stdout:
                stdout = (
                    e.stdout.decode("utf-8")
                    if isinstance(e.stdout, bytes)
                    else str(e.stdout)
                )
            if hasattr(e, "stderr") and e.stderr:
                stderr = (
                    e.stderr.decode("utf-8")
                    if isinstance(e.stderr, bytes)
                    else str(e.stderr)
                )
            if hasattr(e, "exit_status"):
                exit_code = e.exit_status

            return TesterOutput(
                success=False, stdout=stdout, stderr=stderr, exit_code=exit_code
            )
        except Exception as e:
            logger.error(f"Test execution failed: {str(e)}")
            return TesterOutput(
                success=False,
                stdout="",
                stderr=f"Test execution failed: {str(e)}",
                exit_code=1,
            )

    def _ensure_docker_image(self) -> None:
        """Ensure the Docker image exists."""
        try:
            self.docker_client.images.get("coda-sandbox")
            logger.info("Using existing coda-sandbox image")
        except docker.errors.ImageNotFound:
            logger.info("Building coda-sandbox image")
            # Build image from sandbox/Dockerfile
            dockerfile_path = Path(__file__).parent.parent / "sandbox"
            if dockerfile_path.exists():
                self.docker_client.images.build(
                    path=str(dockerfile_path), tag="coda-sandbox", rm=True
                )
                logger.info("Successfully built coda-sandbox image")
            else:
                raise RuntimeError(
                    "sandbox/Dockerfile not found. Please create it first."
                ) from None
