"""
Base class for LLM clients.

Provides a consistent interface that all provider-specific clients must implement.
"""

from abc import ABC, abstractmethod
from typing import Optional, Generator, Any
from dataclasses import dataclass


@dataclass
class ChatMessage:
    """Represents a message in a conversation."""
    role: str  # "system", "user", "assistant"
    content: str


@dataclass
class ChatResponse:
    """Standardized response from any LLM provider."""
    content: str
    model: str
    usage: Optional[dict] = None
    finish_reason: Optional[str] = None
    raw_response: Optional[Any] = None


class BaseLLMClient(ABC):
    """
    Abstract base class for LLM clients.
    
    All provider-specific clients should inherit from this class
    and implement the required methods.
    """
    
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the LLM client.
        
        Args:
            model: Model identifier (provider-specific)
            api_key: API key (if not set via environment variable)
            **kwargs: Additional provider-specific arguments
        """
        self.model = model or self.default_model
        self.api_key = api_key
        self._client = None
    
    @property
    @abstractmethod
    def default_model(self) -> str:
        """Return the default model for this provider."""
        pass
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name."""
        pass
    
    @abstractmethod
    def _initialize_client(self) -> None:
        """Initialize the provider-specific client."""
        pass
    
    @abstractmethod
    def chat(
        self,
        message: str,
        system_prompt: Optional[str] = None,
        history: Optional[list[ChatMessage]] = None,
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Send a chat message and get a response.
        
        Args:
            message: The user's message
            system_prompt: Optional system prompt to set behavior
            history: Optional conversation history
            temperature: Randomness (0.0-2.0)
            max_tokens: Maximum tokens in response
            **kwargs: Additional provider-specific arguments
            
        Returns:
            The assistant's response text
        """
        pass
    
    @abstractmethod
    def chat_with_details(
        self,
        message: str,
        system_prompt: Optional[str] = None,
        history: Optional[list[ChatMessage]] = None,
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> ChatResponse:
        """
        Send a chat message and get a detailed response.
        
        Args:
            message: The user's message
            system_prompt: Optional system prompt to set behavior
            history: Optional conversation history
            temperature: Randomness (0.0-2.0)
            max_tokens: Maximum tokens in response
            **kwargs: Additional provider-specific arguments
            
        Returns:
            ChatResponse with content, usage stats, and metadata
        """
        pass
    
    @abstractmethod
    def stream(
        self,
        message: str,
        system_prompt: Optional[str] = None,
        history: Optional[list[ChatMessage]] = None,
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """
        Stream a chat response token by token.
        
        Args:
            message: The user's message
            system_prompt: Optional system prompt to set behavior
            history: Optional conversation history
            temperature: Randomness (0.0-2.0)
            max_tokens: Maximum tokens in response
            **kwargs: Additional provider-specific arguments
            
        Yields:
            Response tokens as they are generated
        """
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model='{self.model}')"
