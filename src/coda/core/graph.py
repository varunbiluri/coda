"""LangGraph workflow for the Coda system."""

from typing import Any, NotRequired, TypedDict

from langgraph.graph import END, StateGraph

from ..agents.apply_patch import ApplyPatchAgent
from ..agents.coder import CoderAgent
from ..agents.planner import PlannerAgent
from ..agents.tester import TesterAgent
from .indexer import RepositoryIndexer
from .llm_client import LLMClient
from .models import ApplyPatchOutput, CoderOutput, PlannerSpec, RunContext, TesterOutput


class GraphState(TypedDict):
    """State for the LangGraph workflow."""

    run_context: RunContext
    goal: str
    planner_spec: NotRequired[PlannerSpec | None]
    coder_output: NotRequired[CoderOutput | None]
    apply_patch_output: NotRequired[ApplyPatchOutput | None]
    tester_output: NotRequired[TesterOutput | None]
    error: str


class CodaGraph:
    """LangGraph workflow for the Coda system."""

    def __init__(self, indexer: RepositoryIndexer, llm_client: LLMClient):
        """Initialize the Coda graph.

        Args:
            indexer: Repository indexer
            llm_client: LLM client
        """
        self.indexer = indexer
        self.llm_client = llm_client
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(GraphState)

        # Add nodes
        workflow.add_node("planner", self._planner_node)
        workflow.add_node("coder", self._coder_node)
        workflow.add_node("apply_patch", self._apply_patch_node)
        workflow.add_node("tester", self._tester_node)

        # Add edges
        workflow.set_entry_point("planner")
        workflow.add_edge("planner", "coder")
        workflow.add_edge("coder", "apply_patch")
        workflow.add_edge("apply_patch", "tester")
        workflow.add_edge("tester", END)

        return workflow.compile()

    def _planner_node(self, state: GraphState) -> dict[str, Any]:
        """Execute the planner node."""
        try:
            agent = PlannerAgent(self.indexer, self.llm_client)
            planner_spec = agent.execute(state["goal"])
            return {"planner_spec": planner_spec}
        except Exception as e:
            return {"error": f"Planner failed: {str(e)}"}

    def _coder_node(self, state: GraphState) -> dict[str, Any]:
        """Execute the coder node."""
        try:
            if state.get("error"):  # Only skip if there's an actual error
                return {}

            planner_spec = state.get("planner_spec")
            if not planner_spec:
                return {"error": "No planner spec available for code generation"}

            agent = CoderAgent(self.indexer, self.llm_client)
            coder_output = agent.execute(planner_spec)
            return {"coder_output": coder_output}
        except Exception as e:
            return {"error": f"Coder failed: {str(e)}"}

    def _apply_patch_node(self, state: GraphState) -> dict[str, Any]:
        """Execute the apply patch node."""
        try:
            if state.get("error"):  # Only skip if there's an actual error
                return {}

            coder_output = state.get("coder_output")
            if not coder_output:
                return {"error": "No coder output available for patch application"}

            run_context = state["run_context"]
            agent = ApplyPatchAgent(run_context.workspace_path)
            apply_patch_output = agent.execute(coder_output, str(run_context.run_id))
            return {"apply_patch_output": apply_patch_output}
        except Exception as e:
            return {"error": f"Apply patch failed: {str(e)}"}

    def _tester_node(self, state: GraphState) -> dict[str, Any]:
        """Execute the tester node."""
        try:
            if state.get("error"):  # Only skip if there's an actual error
                return {}

            if not state.get("apply_patch_output"):
                return {"error": "No patch output available for testing"}

            run_context = state["run_context"]
            agent = TesterAgent(run_context.workspace_path)
            tester_output = agent.execute()
            return {"tester_output": tester_output}
        except Exception as e:
            return {"error": f"Tester failed: {str(e)}"}

    def execute(self, run_context: RunContext, goal: str) -> GraphState:
        """Execute the complete workflow.

        Args:
            run_context: Run context
            goal: Goal to achieve

        Returns:
            Final graph state
        """
        initial_state = GraphState(
            run_context=run_context,
            goal=goal,
            planner_spec=None,
            coder_output=None,
            apply_patch_output=None,
            tester_output=None,
            error="",
        )

        # Execute the graph
        final_state = self.graph.invoke(initial_state)
        return final_state  # type: ignore
