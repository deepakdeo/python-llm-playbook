"""
Groq client implementation.

Provides access to Groq's ultra-fast inference API for open-source models
(Llama, Mixtral, Gemma, etc.)

Documentation: https://console.groq.com/docs/quickstart
"""

from typing import Optional, Generator
from .base import BaseLLMClient, ChatMessage, ChatResponse
from .utils import get_api_key, logger


class GroqClient(BaseLLMClient):
    """
    Client for Groq's ultra-fast inference API.
    
    Groq provides extremely fast inference for open-source models like
    Llama 3, Mixtral, and Gemma.
    
    Example:
        >>> client = GroqClient()
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
        Initialize the Groq client.
        
        Args:
            model: Model to use (default: llama-3.3-70b-versatile)
            api_key: Groq API key (or set GROQ_API_KEY env var)
            **kwargs: Additional arguments passed to Groq client
        """
        super().__init__(model=model, api_key=api_key, **kwargs)
        self._kwargs = kwargs
        self._initialize_client()
    
    @property
    def default_model(self) -> str:
        return "llama-3.3-70b-versatile"
    
    @property
    def provider_name(self) -> str:
        return "Groq"
    
    def _initialize_client(self) -> None:
        """Initialize the Groq client."""
        from groq import Groq
        
        api_key = get_api_key("GROQ_API_KEY", self.api_key)
        self._client = Groq(api_key=api_key, **self._kwargs)
        logger.debug(f"Initialized Groq client with model: {self.model}")
    
    def _build_messages(
        self,
        message: str,
        system_prompt: Optional[str] = None,
        history: Optional[list[ChatMessage]] = None
    ) -> list[dict]:
        """Build the messages array for the API call."""
        messages = []
        
        # Add system prompt if provided
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add history if provided
        if history:
            for msg in history:
                messages.append({"role": msg.role, "content": msg.content})
        
        # Add current message
        messages.append({"role": "user", "content": message})
        
        return messages
    
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
            temperature: Randomness (0.0-2.0, default: 1.0)
            max_tokens: Maximum tokens in response
            **kwargs: Additional Groq-specific arguments
            
        Returns:
            The assistant's response text
            
        Example:
            >>> client = GroqClient()
            >>> response = client.chat(
            ...     "Explain quantum computing",
            ...     system_prompt="You are a physics teacher",
            ...     temperature=0.7
            ... )
        """
        messages = self._build_messages(message, system_prompt, history)
        
        params = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            **kwargs
        }
        
        if max_tokens:
            params["max_tokens"] = max_tokens
        
        response = self._client.chat.completions.create(**params)
        
        return response.choices[0].message.content
    
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
            temperature: Randomness (0.0-2.0, default: 1.0)
            max_tokens: Maximum tokens in response
            **kwargs: Additional Groq-specific arguments
            
        Returns:
            ChatResponse with content, usage stats, and metadata
        """
        messages = self._build_messages(message, system_prompt, history)
        
        params = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            **kwargs
        }
        
        if max_tokens:
            params["max_tokens"] = max_tokens
        
        response = self._client.chat.completions.create(**params)
        
        return ChatResponse(
            content=response.choices[0].message.content,
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
                "queue_time": response.usage.queue_time,
                "prompt_time": response.usage.prompt_time,
                "completion_time": response.usage.completion_time,
                "total_time": response.usage.total_time
            } if response.usage else None,
            finish_reason=response.choices[0].finish_reason,
            raw_response=response
        )
    
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
            temperature: Randomness (0.0-2.0, default: 1.0)
            max_tokens: Maximum tokens in response
            **kwargs: Additional Groq-specific arguments
            
        Yields:
            Response tokens as they are generated
            
        Example:
            >>> client = GroqClient()
            >>> for token in client.stream("Write a poem"):
            ...     print(token, end="", flush=True)
        """
        messages = self._build_messages(message, system_prompt, history)
        
        params = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
            **kwargs
        }
        
        if max_tokens:
            params["max_tokens"] = max_tokens
        
        stream = self._client.chat.completions.create(**params)
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    
    @staticmethod
    def available_models() -> list[str]:
        """
        Return a list of commonly available Groq models.
        
        Note: Check https://console.groq.com/docs/models for the latest list.
        
        Returns:
            List of model identifiers
        """
        return [
            "llama-3.3-70b-versatile",
            "llama-3.1-70b-versatile",
            "llama-3.1-8b-instant",
            "llama3-70b-8192",
            "llama3-8b-8192",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
        ]
