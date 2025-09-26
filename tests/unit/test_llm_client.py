"""Unit tests for LLM client functionality."""

import os
from unittest.mock import patch, MagicMock

import pytest

from src.coda.core.llm_client import (
    LiteLLMClient,
    MockLLMClient,
    create_llm_client,
)


class TestMockLLMClient:
    """Test cases for MockLLMClient."""

    def test_mock_client_initialization(self):
        """Test MockLLMClient initializes correctly."""
        client = MockLLMClient()
        assert client is not None
        assert hasattr(client, 'responses')
        assert 'health_endpoint' in client.responses

    def test_mock_planner_spec_health_goal(self):
        """Test mock planner spec generation for health endpoint goal."""
        client = MockLLMClient()
        context = "FastAPI application with basic endpoints"
        
        result = client.generate_planner_spec("Add /health endpoint", context)
        
        assert 'tasks' in result
        assert 'context' in result
        assert 'estimated_changes' in result
        assert len(result['tasks']) == 2
        assert result['tasks'][0]['id'] == 'add_health_endpoint'
        assert 'health' in result['tasks'][0]['description'].lower()

    def test_mock_planner_spec_generic_goal(self):
        """Test mock planner spec generation for generic goal."""
        client = MockLLMClient()
        context = "Python application"
        
        result = client.generate_planner_spec("Add new feature", context)
        
        assert 'tasks' in result
        assert result['context'] == "Working on goal: Add new feature"
        assert len(result['tasks']) == 1
        assert result['tasks'][0]['description'] == "Implement: Add new feature"

    def test_mock_code_diff_health_endpoint(self):
        """Test mock code diff generation for health endpoint."""
        client = MockLLMClient()
        planner_spec = {
            'tasks': [
                {'id': 'add_health', 'description': 'Add /health endpoint', 'files_to_modify': ['app/main.py']}
            ]
        }
        context = "FastAPI application"
        
        result = client.generate_code_diff(planner_spec, context)
        
        assert 'diff' in result
        assert 'commit_message' in result
        assert 'explanation' in result
        assert '@app.get("/health")' in result['diff']
        assert 'health' in result['commit_message'].lower()

    def test_mock_code_diff_generic(self):
        """Test mock code diff generation for generic task."""
        client = MockLLMClient()
        planner_spec = {
            'tasks': [
                {'id': 'add_feature', 'description': 'Add new feature', 'files_to_modify': ['app/main.py']}
            ]
        }
        context = "Python application"
        
        result = client.generate_code_diff(planner_spec, context)
        
        assert 'diff' in result
        assert 'commit_message' in result
        assert 'explanation' in result
        assert '@app.get("/new-feature")' in result['diff']


class TestLiteLLMClient:
    """Test cases for LiteLLMClient."""

    @patch('src.coda.core.llm_client.LITELLM_AVAILABLE', True)
    @patch('src.coda.core.llm_client.completion')
    def test_litellm_client_initialization_openai(self, mock_completion):
        """Test LiteLLMClient initialization with OpenAI provider."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            client = LiteLLMClient(model='gpt-3.5-turbo', provider='openai')
            assert client.model == 'gpt-3.5-turbo'
            assert client.provider == 'openai'

    @patch('src.coda.core.llm_client.LITELLM_AVAILABLE', True)
    def test_litellm_client_initialization_anthropic(self):
        """Test LiteLLMClient initialization with Anthropic provider."""
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test-key'}):
            client = LiteLLMClient(model='claude-3-sonnet', provider='anthropic')
            assert client.model == 'claude-3-sonnet'
            assert client.provider == 'anthropic'

    @patch('src.coda.core.llm_client.LITELLM_AVAILABLE', True)
    def test_litellm_client_missing_api_key(self):
        """Test LiteLLMClient raises error when API key is missing."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="OPENAI_API_KEY environment variable is required"):
                LiteLLMClient(model='gpt-3.5-turbo', provider='openai')

    @patch('src.coda.core.llm_client.LITELLM_AVAILABLE', False)
    def test_litellm_client_not_available(self):
        """Test LiteLLMClient raises error when LiteLLM is not available."""
        with pytest.raises(ImportError, match="LiteLLM package not available"):
            LiteLLMClient(model='gpt-3.5-turbo', provider='openai')

    @patch('src.coda.core.llm_client.LITELLM_AVAILABLE', True)
    @patch('src.coda.core.llm_client.completion')
    def test_litellm_planner_spec_generation(self, mock_completion):
        """Test LiteLLM planner spec generation."""
        # Mock the completion response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"tasks": [{"id": "test", "description": "Test task"}], "context": "Test context", "estimated_changes": ["Change 1"]}'
        mock_completion.return_value = mock_response
        
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            client = LiteLLMClient(model='gpt-3.5-turbo', provider='openai')
            result = client.generate_planner_spec("Test goal", "Test context")
            
            assert 'tasks' in result
            assert 'context' in result
            assert 'estimated_changes' in result
            mock_completion.assert_called_once()

    @patch('src.coda.core.llm_client.LITELLM_AVAILABLE', True)
    @patch('src.coda.core.llm_client.completion')
    def test_litellm_code_diff_generation(self, mock_completion):
        """Test LiteLLM code diff generation."""
        # Mock the completion response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"diff": "--- a/file.py", "commit_message": "Test commit", "explanation": "Test explanation"}'
        mock_completion.return_value = mock_response
        
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            client = LiteLLMClient(model='gpt-3.5-turbo', provider='openai')
            planner_spec = {'tasks': [{'id': 'test', 'description': 'Test task'}]}
            result = client.generate_code_diff(planner_spec, "Test context")
            
            assert 'diff' in result
            assert 'commit_message' in result
            assert 'explanation' in result
            mock_completion.assert_called_once()

    @patch('src.coda.core.llm_client.LITELLM_AVAILABLE', True)
    @patch('src.coda.core.llm_client.completion')
    def test_litellm_json_parse_error_fallback(self, mock_completion):
        """Test LiteLLM fallback when JSON parsing fails."""
        # Mock the completion response with invalid JSON
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = 'Invalid JSON response'
        mock_completion.return_value = mock_response
        
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            client = LiteLLMClient(model='gpt-3.5-turbo', provider='openai')
            result = client.generate_planner_spec("Test goal", "Test context")
            
            # Should fallback to default structure
            assert 'tasks' in result
            assert result['context'] == "Implementing: Test goal"


class TestCreateLLMClient:
    """Test cases for create_llm_client factory function."""

    def test_create_mock_client(self):
        """Test creating mock client."""
        client = create_llm_client(use_mock=True)
        assert isinstance(client, MockLLMClient)

    @patch('src.coda.core.llm_client.LITELLM_AVAILABLE', True)
    def test_create_litellm_client_openai(self):
        """Test creating LiteLLM client with OpenAI."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            client = create_llm_client(use_mock=False, model='gpt-3.5-turbo', provider='openai')
            assert isinstance(client, LiteLLMClient)
            assert client.provider == 'openai'

    @patch('src.coda.core.llm_client.LITELLM_AVAILABLE', True)
    def test_create_litellm_client_anthropic(self):
        """Test creating LiteLLM client with Anthropic."""
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test-key'}):
            client = create_llm_client(use_mock=False, model='claude-3-sonnet', provider='anthropic')
            assert isinstance(client, LiteLLMClient)
            assert client.provider == 'anthropic'

    @patch('src.coda.core.llm_client.LITELLM_AVAILABLE', False)
    def test_create_client_fallback_to_mock(self):
        """Test fallback to mock client when LiteLLM is not available."""
        client = create_llm_client(use_mock=False, model='gpt-3.5-turbo', provider='openai')
        assert isinstance(client, MockLLMClient)

    def test_create_client_with_missing_credentials(self):
        """Test fallback to mock client when credentials are missing."""
        with patch.dict(os.environ, {}, clear=True):
            client = create_llm_client(use_mock=False, model='gpt-3.5-turbo', provider='openai')
            assert isinstance(client, MockLLMClient)
