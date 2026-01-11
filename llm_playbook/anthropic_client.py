"""
Anthropic client implementation.

Provides access to Anthropic's Claude models (Claude 4, Claude 3.5 Sonnet, etc.)

Documentation: https://docs.anthropic.com/en/api/getting-started
"""

from typing import Optional, Generator
from .base import BaseLLMClient, ChatMessage, ChatResponse
from .utils import get_api_key, logger


class AnthropicClient(BaseLLMClient):
    """
    Client for Anthropic's Claude models.
    
    Example:
        >>> client = AnthropicClient()
        >>> response = client.chat("What is machine learning?")
        >>> print(response)
    """
    
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the Anthropic client.
        
        Args:
            model: Model to use (default: claude-sonnet-4-20250514)
            api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var)
            **kwargs: Additional arguments passed to Anthropic client
        """
        super().__init__(model=model, api_key=api_key, **kwargs)
        self._kwargs = kwargs
        self._initialize_client()
    
    @property
    def default_model(self) -> str:
        return "claude-sonnet-4-20250514"
    
    @property
    def provider_name(self) -> str:
        return "Anthropic"
    
    def _initialize_client(self) -> None:
        """Initialize the Anthropic client."""
        from anthropic import Anthropic
        
        api_key = get_api_key("ANTHROPIC_API_KEY", self.api_key)
        self._client = Anthropic(api_key=api_key, **self._kwargs)
        logger.debug(f"Initialized Anthropic client with model: {self.model}")
    
    def _build_messages(
        self,
        message: str,
        history: Optional[list[ChatMessage]] = None
    ) -> list[dict]:
        """Build the messages array for the API call."""
        messages = []
        
        # Add history if provided
        if history:
            for msg in history:
                # Anthropic uses "user" and "assistant" roles
                role = msg.role if msg.role != "system" else "user"
                messages.append({"role": role, "content": msg.content})
        
        # Add current message
        messages.append({"role": "user", "content": message})
        
        return messages
    
    def chat(
        self,
        message: str,
        system_prompt: Optional[str] = None,
        history: Optional[list[ChatMessage]] = None,
        temperature: float = 1.0,
        max_tokens: Optional[int] = 1024,
        **kwargs
    ) -> str:
        """
        Send a chat message and get a response.
        
        Args:
            message: The user's message
            system_prompt: Optional system prompt to set behavior
            history: Optional conversation history
            temperature: Randomness (0.0-1.0, default: 1.0)
            max_tokens: Maximum tokens in response (default: 1024)
            **kwargs: Additional Anthropic-specific arguments
            
        Returns:
            The assistant's response text
            
        Example:
            >>> client = AnthropicClient()
            >>> response = client.chat(
            ...     "Explain quantum computing",
            ...     system_prompt="You are a physics teacher",
            ...     temperature=0.7
            ... )
        """
        messages = self._build_messages(message, history)
        
        params = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs
        }
        
        if system_prompt:
            params["system"] = system_prompt
        
        response = self._client.messages.create(**params)
        
        return response.content[0].text
    
    def chat_with_details(
        self,
        message: str,
        system_prompt: Optional[str] = None,
        history: Optional[list[ChatMessage]] = None,
        temperature: float = 1.0,
        max_tokens: Optional[int] = 1024,
        **kwargs
    ) -> ChatResponse:
        """
        Send a chat message and get a detailed response.
        
        Args:
            message: The user's message
            system_prompt: Optional system prompt to set behavior
            history: Optional conversation history
            temperature: Randomness (0.0-1.0, default: 1.0)
            max_tokens: Maximum tokens in response (default: 1024)
            **kwargs: Additional Anthropic-specific arguments
            
        Returns:
            ChatResponse with content, usage stats, and metadata
        """
        messages = self._build_messages(message, history)
        
        params = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs
        }
        
        if system_prompt:
            params["system"] = system_prompt
        
        response = self._client.messages.create(**params)
        
        return ChatResponse(
            content=response.content[0].text,
            model=response.model,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens
            },
            finish_reason=response.stop_reason,
            raw_response=response
        )
    
    def stream(
        self,
        message: str,
        system_prompt: Optional[str] = None,
        history: Optional[list[ChatMessage]] = None,
        temperature: float = 1.0,
        max_tokens: Optional[int] = 1024,
        **kwargs
    ) -> Generator[str, None, None]:
        """
        Stream a chat response token by token.
        
        Args:
            message: The user's message
            system_prompt: Optional system prompt to set behavior
            history: Optional conversation history
            temperature: Randomness (0.0-1.0, default: 1.0)
            max_tokens: Maximum tokens in response (default: 1024)
            **kwargs: Additional Anthropic-specific arguments
            
        Yields:
            Response tokens as they are generated
            
        Example:
            >>> client = AnthropicClient()
            >>> for token in client.stream("Write a poem"):
            ...     print(token, end="", flush=True)
        """
        messages = self._build_messages(message, history)
        
        params = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs
        }
        
        if system_prompt:
            params["system"] = system_prompt
        
        with self._client.messages.stream(**params) as stream:
            for text in stream.text_stream:
                yield text
