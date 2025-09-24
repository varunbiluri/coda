"""LLM client interface with OpenAI and mock implementations."""

import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)

try:
    from openai import AzureOpenAI, OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    def generate_planner_spec(self, goal: str, context: str) -> dict[str, Any]:
        """Generate a planner specification.

        Args:
            goal: The goal to achieve
            context: Repository context from indexer

        Returns:
            Planner specification as JSON
        """
        pass

    @abstractmethod
    def generate_code_diff(self, planner_spec: dict[str, Any], context: str) -> dict[str, str]:
        """Generate code diff and commit message.

        Args:
            planner_spec: Planner specification
            context: Repository context

        Returns:
            Dictionary with 'diff', 'commit_message', and 'explanation'
        """
        pass


class MockLLMClient(LLMClient):
    """Mock LLM client for testing and demo purposes."""

    def __init__(self):
        """Initialize mock client with predefined responses."""
        self.responses = {
            "health_endpoint": {
                "planner_spec": {
                    "tasks": [
                        {
                            "id": "add_health_endpoint",
                            "description": "Add /health endpoint to FastAPI app",
                            "files_to_modify": ["app/main.py"],
                            "priority": "high",
                        },
                        {
                            "id": "add_health_test",
                            "description": "Add test for /health endpoint",
                            "files_to_modify": ["tests/test_health.py"],
                            "priority": "high",
                        },
                    ],
                    "context": "Adding health check endpoint for service monitoring",
                    "estimated_changes": [
                        'Add GET /health endpoint returning {"status": "healthy"}',
                        "Add corresponding test case",
                    ],
                },
                "diff": '''--- a/app/main.py
+++ b/app/main.py
@@ -9,3 +9,7 @@ app = FastAPI(title="Sample Service", version="0.1.0")
 def read_root():
     """Root endpoint."""
     return {"message": "Hello World"}
+
+@app.get("/health")
+def health_check():
+    return {"status": "healthy"}
''',
                "commit_message": "Add /health endpoint with tests",
                "explanation": "Added a health check endpoint that returns service status and corresponding test",
            }
        }

    def generate_planner_spec(self, goal: str, context: str) -> dict[str, Any]:
        """Generate a mock planner specification."""
        # For demo purposes, return health endpoint spec for any goal containing "health"
        if "health" in goal.lower():
            return self.responses["health_endpoint"]["planner_spec"]

        # Default generic response
        return {
            "tasks": [
                {
                    "id": "implement_goal",
                    "description": f"Implement: {goal}",
                    "files_to_modify": ["app/main.py"],
                    "priority": "high",
                }
            ],
            "context": f"Working on goal: {goal}",
            "estimated_changes": [f"Implement {goal}"],
        }

    def generate_code_diff(self, planner_spec: dict[str, Any], context: str) -> dict[str, str]:
        """Generate a mock code diff."""
        # Check if this is a health endpoint task
        tasks = planner_spec.get("tasks", [])
        if any("health" in task.get("description", "").lower() for task in tasks):
            return {
                "diff": self.responses["health_endpoint"]["diff"],
                "commit_message": self.responses["health_endpoint"]["commit_message"],
                "explanation": self.responses["health_endpoint"]["explanation"],
            }

        # Default generic diff
        return {
            "diff": """--- a/app/main.py
+++ b/app/main.py
@@ -6,3 +6,7 @@
 @app.get("/")
 def read_root():
     return {"message": "Hello World"}
+
+@app.get("/new-feature")
+def new_feature():
+    return {"message": "New feature implemented"}""",
            "commit_message": "Implement new feature",
            "explanation": "Added new feature endpoint",
        }


class OpenAILLMClient(LLMClient):
    """OpenAI LLM client for production use."""

    def __init__(self, model: str = "gpt-3.5-turbo"):
        """Initialize OpenAI client.

        Args:
            model: OpenAI model to use (default: gpt-3.5-turbo)

        Raises:
            ImportError: If OpenAI package is not available
            ValueError: If OPENAI_API_KEY is not set
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not available. Install with: pip install openai")

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required for OpenAI client")

        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate_planner_spec(self, goal: str, context: str) -> dict[str, Any]:
        """Generate a planner specification using OpenAI."""
        system_prompt = """You are a code planning assistant. Given a goal and repository context,
create a detailed execution plan as JSON.

Return a JSON object with this structure:
{
    "tasks": [
        {
            "id": "unique_task_id",
            "description": "What needs to be done",
            "files_to_modify": ["list", "of", "files"],
            "priority": "high|medium|low"
        }
    ],
    "context": "Brief explanation of what we're building",
    "estimated_changes": ["List of specific changes to make"]
}"""

        user_prompt = f"""Goal: {goal}

Repository Context:
{context}

Create a detailed plan to achieve this goal."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=1000,
            )

            content = response.choices[0].message.content
            return json.loads(content)

        except json.JSONDecodeError:
            # Fallback to a simple plan if JSON parsing fails
            return {
                "tasks": [
                    {
                        "id": "implement_goal",
                        "description": goal,
                        "files_to_modify": ["app/main.py"],
                        "priority": "high",
                    }
                ],
                "context": f"Implementing: {goal}",
                "estimated_changes": [goal],
            }
        except Exception as e:
            raise RuntimeError(f"Failed to generate planner spec: {e}") from e

    def generate_code_diff(self, planner_spec: dict[str, Any], context: str) -> dict[str, str]:
        """Generate code diff and commit message using OpenAI."""
        system_prompt = """You are a code generation assistant. Given a plan and repository context,
generate a unified git diff that implements the plan.

Return a JSON object with this structure:
{
    "diff": "unified git diff string",
    "commit_message": "concise commit message",
    "explanation": "brief explanation of changes"
}

The diff must be a valid unified diff format that can be applied with 'git apply'."""

        user_prompt = f"""Plan to implement:
{json.dumps(planner_spec, indent=2)}

Repository Context:
{context}

Generate a unified git diff to implement this plan."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=2000,
            )

            content = response.choices[0].message.content
            result = json.loads(content)

            # Ensure required keys exist
            if not all(key in result for key in ["diff", "commit_message", "explanation"]):
                raise ValueError("Response missing required keys")

            return result

        except (json.JSONDecodeError, ValueError):
            # Fallback to a simple diff if parsing fails
            tasks = planner_spec.get("tasks", [])
            task_desc = (
                tasks[0].get("description", "Implement feature") if tasks else "Implement feature"
            )

            return {
                "diff": """--- a/app/main.py
+++ b/app/main.py
@@ -6,3 +6,7 @@
 @app.get("/")
 def read_root():
     return {"message": "Hello World"}
+
+@app.get("/new-feature")
+def new_feature():
+    return {"message": "Feature implemented"}""",
                "commit_message": f"Implement: {task_desc}",
                "explanation": f"Added implementation for: {task_desc}",
            }
        except Exception as e:
            raise RuntimeError(f"Failed to generate code diff: {e}") from e


class AzureOpenAILLMClient(LLMClient):
    """Azure OpenAI LLM client for enterprise use."""

    def __init__(self, model: str = "gpt-35-turbo"):
        """Initialize Azure OpenAI client.

        Args:
            model: Azure OpenAI deployment name (default: gpt-35-turbo)

        Raises:
            ImportError: If OpenAI package is not available
            ValueError: If required Azure environment variables are not set
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not available. Install with: pip install openai")

        # Required Azure OpenAI environment variables
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

        if not api_key:
            raise ValueError("AZURE_OPENAI_API_KEY environment variable is required")
        if not endpoint:
            raise ValueError("AZURE_OPENAI_ENDPOINT environment variable is required")

        self.client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version,
        )
        self.model = model

    def generate_planner_spec(self, goal: str, context: str) -> dict[str, Any]:
        """Generate a planner specification using Azure OpenAI."""
        system_prompt = """You are a code planning assistant. Given a goal and repository context,
create a detailed execution plan as JSON.

Return a JSON object with this structure:
{
    "tasks": [
        {
            "id": "unique_task_id",
            "description": "What needs to be done",
            "files_to_modify": ["list", "of", "files"],
            "priority": "high|medium|low"
        }
    ],
    "context": "Brief explanation of what we're building",
    "estimated_changes": ["List of specific changes to make"]
}"""

        user_prompt = f"""Goal: {goal}

Repository Context:
{context}

Create a detailed plan to achieve this goal."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=1000,
            )

            content = response.choices[0].message.content
            return json.loads(content)

        except json.JSONDecodeError:
            # Fallback to a simple plan if JSON parsing fails
            return {
                "tasks": [
                    {
                        "id": "implement_goal",
                        "description": goal,
                        "files_to_modify": ["app/main.py"],
                        "priority": "high",
                    }
                ],
                "context": f"Implementing: {goal}",
                "estimated_changes": [goal],
            }
        except Exception as e:
            raise RuntimeError(f"Failed to generate planner spec: {e}") from e

    def generate_code_diff(self, planner_spec: dict[str, Any], context: str) -> dict[str, str]:
        """Generate code diff and commit message using Azure OpenAI."""
        system_prompt = """You are a code generation assistant. Given a plan and repository context,
generate a unified git diff that implements the plan.

Return a JSON object with this structure:
{
    "diff": "unified git diff string",
    "commit_message": "concise commit message",
    "explanation": "brief explanation of changes"
}

The diff must be a valid unified diff format that can be applied with 'git apply'."""

        user_prompt = f"""Plan to implement:
{json.dumps(planner_spec, indent=2)}

Repository Context:
{context}

Generate a unified git diff to implement this plan."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=2000,
            )

            content = response.choices[0].message.content
            result = json.loads(content)

            # Ensure required keys exist
            if not all(key in result for key in ["diff", "commit_message", "explanation"]):
                raise ValueError("Response missing required keys")

            return result

        except (json.JSONDecodeError, ValueError):
            # Fallback to a simple diff if parsing fails
            tasks = planner_spec.get("tasks", [])
            task_desc = (
                tasks[0].get("description", "Implement feature") if tasks else "Implement feature"
            )

            return {
                "diff": """--- a/app/main.py
+++ b/app/main.py
@@ -6,3 +6,7 @@
 @app.get("/")
 def read_root():
     return {"message": "Hello World"}
+
+@app.get("/new-feature")
+def new_feature():
+    return {"message": "Feature implemented"}""",
                "commit_message": f"Implement: {task_desc}",
                "explanation": f"Added implementation for: {task_desc}",
            }
        except Exception as e:
            raise RuntimeError(f"Failed to generate code diff: {e}") from e


def create_llm_client(use_mock: bool = False, model: str = "gpt-3.5-turbo") -> LLMClient:
    """Factory function to create appropriate LLM client.

    Args:
        use_mock: If True, use mock client. If False, try Azure OpenAI first, then OpenAI
        model: Model/deployment name to use

    Returns:
        LLM client instance

    Priority order (when use_mock=False):
    1. Azure OpenAI (if AZURE_OPENAI_API_KEY is set)
    2. OpenAI (if OPENAI_API_KEY is set)
    3. Mock client (fallback)
    """
    if use_mock:
        return MockLLMClient()

    # Try Azure OpenAI first (if Azure credentials are available)
    if os.getenv("AZURE_OPENAI_API_KEY") and os.getenv("AZURE_OPENAI_ENDPOINT"):
        try:
            # Convert OpenAI model names to Azure deployment names if needed
            azure_model = model.replace("gpt-3.5-turbo", "gpt-35-turbo")
            return AzureOpenAILLMClient(model=azure_model)
        except (ImportError, ValueError) as e:
            logger.warning(f"Failed to create Azure OpenAI client ({e}), trying OpenAI")

    # Try regular OpenAI client
    if os.getenv("OPENAI_API_KEY"):
        try:
            return OpenAILLMClient(model=model)
        except (ImportError, ValueError) as e:
            logger.warning(f"Failed to create OpenAI client ({e}), falling back to mock")

    # Fallback to mock client
    logger.warning("No API keys found for OpenAI or Azure OpenAI, using mock client")
    return MockLLMClient()
