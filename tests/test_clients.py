"""
Tests for LLM Playbook clients.

These tests verify basic functionality of each client.
Note: Full integration tests require valid API keys.

Run tests:
    pytest tests/test_clients.py -v

For integration tests (requires API keys):
    pytest tests/test_clients.py -v --integration
"""

import pytest
import os
from unittest.mock import Mock, patch


class TestBaseClient:
    """Tests for the base client class."""
    
    def test_chat_message_dataclass(self):
        """Test ChatMessage dataclass."""
        from llm_playbook import ChatMessage
        
        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
    
    def test_chat_response_dataclass(self):
        """Test ChatResponse dataclass."""
        from llm_playbook import ChatResponse
        
        response = ChatResponse(
            content="Hello!",
            model="test-model",
            usage={"total_tokens": 10},
            finish_reason="stop"
        )
        assert response.content == "Hello!"
        assert response.model == "test-model"
        assert response.usage["total_tokens"] == 10


class TestOpenAIClient:
    """Tests for OpenAI client."""
    
    def test_default_model(self):
        """Test default model is set correctly."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            from llm_playbook import OpenAIClient
            
            with patch("llm_playbook.openai_client.OpenAI"):
                client = OpenAIClient()
                assert client.model == "gpt-4o-mini"
    
    def test_custom_model(self):
        """Test custom model can be set."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            from llm_playbook import OpenAIClient
            
            with patch("llm_playbook.openai_client.OpenAI"):
                client = OpenAIClient(model="gpt-4o")
                assert client.model == "gpt-4o"
    
    def test_provider_name(self):
        """Test provider name."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            from llm_playbook import OpenAIClient
            
            with patch("llm_playbook.openai_client.OpenAI"):
                client = OpenAIClient()
                assert client.provider_name == "OpenAI"


class TestAnthropicClient:
    """Tests for Anthropic client."""
    
    def test_default_model(self):
        """Test default model is set correctly."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            from llm_playbook import AnthropicClient
            
            with patch("llm_playbook.anthropic_client.Anthropic"):
                client = AnthropicClient()
                assert "claude" in client.model.lower()
    
    def test_provider_name(self):
        """Test provider name."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            from llm_playbook import AnthropicClient
            
            with patch("llm_playbook.anthropic_client.Anthropic"):
                client = AnthropicClient()
                assert client.provider_name == "Anthropic"


class TestGeminiClient:
    """Tests for Gemini client."""
    
    def test_default_model(self):
        """Test default model is set correctly."""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            from llm_playbook import GeminiClient
            
            with patch("llm_playbook.gemini_client.genai"):
                client = GeminiClient()
                assert "gemini" in client.model.lower()
    
    def test_provider_name(self):
        """Test provider name."""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            from llm_playbook import GeminiClient
            
            with patch("llm_playbook.gemini_client.genai"):
                client = GeminiClient()
                assert client.provider_name == "Google Gemini"


class TestGroqClient:
    """Tests for Groq client."""
    
    def test_default_model(self):
        """Test default model is set correctly."""
        with patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}):
            from llm_playbook import GroqClient
            
            with patch("llm_playbook.groq_client.Groq"):
                client = GroqClient()
                assert "llama" in client.model.lower()
    
    def test_provider_name(self):
        """Test provider name."""
        with patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}):
            from llm_playbook import GroqClient
            
            with patch("llm_playbook.groq_client.Groq"):
                client = GroqClient()
                assert client.provider_name == "Groq"
    
    def test_available_models(self):
        """Test available models list."""
        from llm_playbook import GroqClient
        
        models = GroqClient.available_models()
        assert isinstance(models, list)
        assert len(models) > 0


class TestOllamaClient:
    """Tests for Ollama client."""
    
    def test_default_model(self):
        """Test default model is set correctly."""
        from llm_playbook import OllamaClient
        
        with patch("llm_playbook.ollama_client.Client"):
            client = OllamaClient()
            assert client.model == "llama3.2"
    
    def test_provider_name(self):
        """Test provider name."""
        from llm_playbook import OllamaClient
        
        with patch("llm_playbook.ollama_client.Client"):
            client = OllamaClient()
            assert "ollama" in client.provider_name.lower()
    
    def test_popular_models(self):
        """Test popular models list."""
        from llm_playbook import OllamaClient
        
        models = OllamaClient.popular_models()
        assert isinstance(models, list)
        assert "llama3.2" in models


class TestUtils:
    """Tests for utility functions."""
    
    def test_truncate_text_short(self):
        """Test truncate_text with short text."""
        from llm_playbook.utils import truncate_text
        
        result = truncate_text("Hello", max_length=100)
        assert result == "Hello"
    
    def test_truncate_text_long(self):
        """Test truncate_text with long text."""
        from llm_playbook.utils import truncate_text
        
        long_text = "A" * 200
        result = truncate_text(long_text, max_length=100)
        assert len(result) == 100
        assert result.endswith("...")
    
    def test_count_tokens_approximate(self):
        """Test approximate token counting."""
        from llm_playbook.utils import count_tokens_approximate
        
        # ~4 chars per token
        result = count_tokens_approximate("Hello World!")  # 12 chars
        assert result == 3  # 12 // 4
    
    def test_get_api_key_from_env(self):
        """Test getting API key from environment."""
        from llm_playbook.utils import get_api_key
        
        with patch.dict(os.environ, {"TEST_KEY": "my-secret-key"}):
            key = get_api_key("TEST_KEY")
            assert key == "my-secret-key"
    
    def test_get_api_key_missing_required(self):
        """Test missing required API key raises error."""
        from llm_playbook.utils import get_api_key
        
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError):
                get_api_key("NONEXISTENT_KEY", required=True)
    
    def test_get_api_key_missing_optional(self):
        """Test missing optional API key returns None."""
        from llm_playbook.utils import get_api_key
        
        with patch.dict(os.environ, {}, clear=True):
            key = get_api_key("NONEXISTENT_KEY", required=False)
            assert key is None


# Integration tests (require API keys)
@pytest.mark.integration
class TestIntegration:
    """Integration tests that require actual API keys."""
    
    @pytest.fixture(autouse=True)
    def check_api_keys(self):
        """Skip if API keys are not set."""
        pass
    
    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set"
    )
    def test_openai_chat(self):
        """Test actual OpenAI API call."""
        from llm_playbook import OpenAIClient
        
        client = OpenAIClient()
        response = client.chat("Say 'test' and nothing else.")
        assert "test" in response.lower()
    
    @pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY"),
        reason="ANTHROPIC_API_KEY not set"
    )
    def test_anthropic_chat(self):
        """Test actual Anthropic API call."""
        from llm_playbook import AnthropicClient
        
        client = AnthropicClient()
        response = client.chat("Say 'test' and nothing else.")
        assert "test" in response.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
