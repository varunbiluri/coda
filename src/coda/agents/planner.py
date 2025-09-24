"""
Strategic Planning Agent for Code Generation Workflows.

This module implements the planning phase of the multi-agent system, responsible for
analyzing user requirements and repository context to create detailed execution plans.
"""

import logging

from ..core.indexer import RepositoryIndexer
from ..core.llm_client import LLMClient
from ..core.models import PlannerSpec

logger = logging.getLogger(__name__)


class PlannerAgent:
    """
    Strategic planning agent for code generation and modification workflows.

    This agent analyzes user goals against repository context to create comprehensive
    execution plans that guide subsequent agents in the workflow chain.
    """

    def __init__(self, indexer: RepositoryIndexer, llm_client: LLMClient):
        """
        Initialize the planning agent with required dependencies.

        Args:
            indexer: Repository indexer for semantic context retrieval
            llm_client: Language model client for plan generation
        """
        self.indexer = indexer
        self.llm_client = llm_client

    def execute(self, goal: str) -> PlannerSpec:
        """
        Execute strategic planning phase for the given objective.

        This method analyzes the goal against repository context to create a comprehensive
        execution plan that will guide the subsequent code generation and testing phases.

        Args:
            goal: High-level objective describing the desired outcome

        Returns:
            Detailed planning specification containing tasks, context, and change estimates

        Raises:
            ValueError: If the goal is empty or invalid
            RuntimeError: If planning analysis fails
        """
        if not goal or not goal.strip():
            raise ValueError("Goal cannot be empty")

        logger.info(f"Analyzing objective: '{goal}'")

        try:
            # Retrieve relevant repository context through semantic search
            context_results = self.indexer.query(goal, top_k=5)
            context = "\n\n".join(
                [
                    f"File: {result['metadata'].get('file_name', 'unknown')}\n{result['content']}"
                    for result in context_results
                ]
            )

            logger.info(f"Retrieved context from {len(context_results)} relevant files")

            # Generate comprehensive execution plan using language model
            spec_dict = self.llm_client.generate_planner_spec(goal, context)

            planner_spec = PlannerSpec(
                tasks=spec_dict["tasks"],
                context=spec_dict["context"],
                estimated_changes=spec_dict["estimated_changes"],
            )

            logger.info(f"Generated execution plan with {len(planner_spec.tasks)} tasks")
            for i, task in enumerate(planner_spec.tasks, 1):
                logger.debug(f"Task {i}: {task.get('description', 'No description')}")

            return planner_spec

        except Exception as e:
            logger.error(f"Planning phase failed: {e}")
            raise RuntimeError(f"Strategic planning failed: {e}") from e
