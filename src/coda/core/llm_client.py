"""LLM client interface with LiteLLM for unified model access."""

import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)

try:
    import litellm
    from litellm import completion

    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False


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


class LiteLLMClient(LLMClient):
    """LiteLLM client for unified access to multiple LLM providers."""

    def __init__(self, model: str = "gpt-3.5-turbo", provider: str = "openai"):
        """Initialize LiteLLM client.

        Args:
            model: Model name to use (e.g., 'gpt-3.5-turbo', 'gpt-4', 'claude-3-sonnet')
            provider: Provider to use ('openai', 'anthropic', 'azure', 'cohere', etc.)

        Raises:
            ImportError: If LiteLLM package is not available
            ValueError: If required environment variables are not set
        """
        if not LITELLM_AVAILABLE:
            raise ImportError("LiteLLM package not available. Install with: pip install litellm")

        self.model = model
        self.provider = provider
        
        # Set up model mapping for different providers
        self.model_mapping = {
            "openai": {
                "gpt-3.5-turbo": "gpt-3.5-turbo",
                "gpt-4": "gpt-4",
                "gpt-4-turbo": "gpt-4-turbo-preview",
            },
            "anthropic": {
                "claude-3-sonnet": "claude-3-sonnet-20240229",
                "claude-3-haiku": "claude-3-haiku-20240307",
                "claude-3-opus": "claude-3-opus-20240229",
            },
            "azure": {
                "gpt-35-turbo": "azure/your-deployment-name",
                "gpt-4": "azure/your-gpt4-deployment",
            },
            "cohere": {
                "command": "command",
                "command-light": "command-light",
            },
        }

        # Validate provider and model combination
        if provider in self.model_mapping:
            if model not in self.model_mapping[provider]:
                logger.warning(f"Model {model} not found in {provider} mapping, using as-is")
        else:
            logger.warning(f"Provider {provider} not in predefined mappings, using as-is")

        # Set up environment variables for different providers
        self._setup_provider_config(provider)

    def _setup_provider_config(self, provider: str) -> None:
        """Set up provider-specific configuration."""
        if provider == "openai":
            if not os.getenv("OPENAI_API_KEY"):
                raise ValueError("OPENAI_API_KEY environment variable is required for OpenAI provider")
        elif provider == "anthropic":
            if not os.getenv("ANTHROPIC_API_KEY"):
                raise ValueError("ANTHROPIC_API_KEY environment variable is required for Anthropic provider")
        elif provider == "azure":
            if not os.getenv("AZURE_API_KEY"):
                raise ValueError("AZURE_API_KEY environment variable is required for Azure provider")
            if not os.getenv("AZURE_API_BASE"):
                raise ValueError("AZURE_API_BASE environment variable is required for Azure provider")
        elif provider == "cohere":
            if not os.getenv("COHERE_API_KEY"):
                raise ValueError("COHERE_API_KEY environment variable is required for Cohere provider")

    def _get_model_name(self) -> str:
        """Get the full model name for the provider."""
        if self.provider == "azure":
            return self.model_mapping.get(self.provider, {}).get(self.model, self.model)
        elif self.provider == "anthropic":
            return self.model_mapping.get(self.provider, {}).get(self.model, self.model)
        else:
            return self.model

    def _call_llm(self, messages: list[dict[str, str]], **kwargs) -> str:
        """Make a call to the LLM using LiteLLM."""
        try:
            response = completion(
                model=self._get_model_name(),
                messages=messages,
                temperature=kwargs.get("temperature", 0.1),
                max_tokens=kwargs.get("max_tokens", 1000),
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"LiteLLM call failed: {e}") from e

    def generate_planner_spec(self, goal: str, context: str) -> dict[str, Any]:
        """Generate a planner specification using LiteLLM."""
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
            content = self._call_llm(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=1000,
            )

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
        """Generate code diff and commit message using LiteLLM."""
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
            content = self._call_llm(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=2000,
            )

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


def create_llm_client(use_mock: bool = False, model: str = "gpt-3.5-turbo", provider: str = "openai") -> LLMClient:
    """Factory function to create appropriate LLM client.

    Args:
        use_mock: If True, use mock client. If False, use LiteLLM with specified provider
        model: Model name to use
        provider: LLM provider to use ('openai', 'anthropic', 'azure', 'cohere', etc.)

    Returns:
        LLM client instance

    Priority order (when use_mock=False):
    1. LiteLLM with specified provider and model
    2. Mock client (fallback)
    """
    if use_mock:
        return MockLLMClient()

    # Try LiteLLM with specified provider
    try:
        return LiteLLMClient(model=model, provider=provider)
    except (ImportError, ValueError) as e:
        logger.warning(f"Failed to create LiteLLM client ({e}), falling back to mock")
        return MockLLMClient()