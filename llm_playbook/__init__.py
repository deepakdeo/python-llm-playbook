"""
Python LLM API Playbook

A unified Python interface for multiple LLM providers.

Example:
    >>> from llm_playbook import OpenAIClient, AnthropicClient
    >>> 
    >>> # Use OpenAI
    >>> openai = OpenAIClient()
    >>> print(openai.chat("Hello!"))
    >>> 
    >>> # Use Anthropic Claude  
    >>> claude = AnthropicClient()
    >>> print(claude.chat("Hello!"))

Supported Providers:
    - OpenAI (GPT-4o, GPT-4, GPT-3.5)
    - Anthropic (Claude 4, Claude 3.5)
    - Google Gemini (Gemini 2.0, 1.5)
    - Groq (Llama, Mixtral - ultra-fast)
    - Ollama (Local LLMs)
"""

__version__ = "0.1.0"
__author__ = "Your Name"

# Import clients for easy access
from .openai_client import OpenAIClient
from .anthropic_client import AnthropicClient
from .gemini_client import GeminiClient
from .groq_client import GroqClient
from .ollama_client import OllamaClient

# Import base classes and types
from .base import BaseLLMClient, ChatMessage, ChatResponse

# Import utilities
from .utils import (
    load_env_file,
    get_api_key,
    LLMError,
    APIKeyError,
    RateLimitError,
    ModelNotFoundError
)

__all__ = [
    # Clients
    "OpenAIClient",
    "AnthropicClient",
    "GeminiClient",
    "GroqClient",
    "OllamaClient",
    
    # Base classes
    "BaseLLMClient",
    "ChatMessage",
    "ChatResponse",
    
    # Utilities
    "load_env_file",
    "get_api_key",
    
    # Exceptions
    "LLMError",
    "APIKeyError",
    "RateLimitError",
    "ModelNotFoundError",
]
