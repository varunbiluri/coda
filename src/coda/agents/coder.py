"""Coder Agent - Generates code diffs based on planner specifications."""

import logging

from ..core.indexer import RepositoryIndexer
from ..core.llm_client import LLMClient
from ..core.models import CoderOutput, PlannerSpec

logger = logging.getLogger(__name__)


class CoderAgent:
    """Agent that generates code diffs based on planner specifications."""

    def __init__(self, indexer: RepositoryIndexer, llm_client: LLMClient):
        """Initialize the coder agent.

        Args:
            indexer: Repository indexer for context retrieval
            llm_client: LLM client for code generation
        """
        self.indexer = indexer
        self.llm_client = llm_client

    def execute(self, planner_spec: PlannerSpec) -> CoderOutput:
        """Execute the coder to generate a unified diff.

        Args:
            planner_spec: Planner specification with tasks and context

        Returns:
            Coder output with diff and commit message
        """
        logger.info("Generating code changes")

        # Get additional context based on files to modify
        files_to_modify = []
        for task in planner_spec.tasks:
            files_to_modify.extend(task.get("files_to_modify", []))

        logger.info(f"Target files: {', '.join(files_to_modify)}")

        # Query for context on specific files
        context_query = f"files: {' '.join(files_to_modify)} {planner_spec.context}"
        context_results = self.indexer.query(context_query, top_k=5)
        context = "\n\n".join(
            [
                f"File: {result['metadata'].get('file_name', 'unknown')}\n{result['content']}"
                for result in context_results
            ]
        )

        # Generate code diff using LLM
        result = self.llm_client.generate_code_diff(
            {
                "tasks": planner_spec.tasks,
                "context": planner_spec.context,
                "estimated_changes": planner_spec.estimated_changes,
            },
            context,
        )

        coder_output = CoderOutput(
            diff=result["diff"],
            commit_message=result["commit_message"],
            explanation=result["explanation"],
        )

        logger.info(f"Generated diff ({len(coder_output.diff)} chars)")
        logger.info(f"Commit message: {coder_output.commit_message}")

        return coder_output
