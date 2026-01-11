"""
OpenAI client implementation.

Provides access to OpenAI's GPT models (GPT-4o, GPT-4, GPT-3.5, etc.)

Documentation: https://platform.openai.com/docs/api-reference
"""

from typing import Optional, Generator
from .base import BaseLLMClient, ChatMessage, ChatResponse
from .utils import get_api_key, logger


class OpenAIClient(BaseLLMClient):
    """
    Client for OpenAI's GPT models.
    
    Example:
        >>> client = OpenAIClient()
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
        Initialize the OpenAI client.
        
        Args:
            model: Model to use (default: gpt-4o-mini)
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            **kwargs: Additional arguments passed to OpenAI client
        """
        super().__init__(model=model, api_key=api_key, **kwargs)
        self._kwargs = kwargs
        self._initialize_client()
    
    @property
    def default_model(self) -> str:
        return "gpt-4o-mini"
    
    @property
    def provider_name(self) -> str:
        return "OpenAI"
    
    def _initialize_client(self) -> None:
        """Initialize the OpenAI client."""
        from openai import OpenAI
        
        api_key = get_api_key("OPENAI_API_KEY", self.api_key)
        self._client = OpenAI(api_key=api_key, **self._kwargs)
        logger.debug(f"Initialized OpenAI client with model: {self.model}")
    
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
            **kwargs: Additional OpenAI-specific arguments
            
        Returns:
            The assistant's response text
            
        Example:
            >>> client = OpenAIClient()
            >>> response = client.chat(
            ...     "Explain quantum computing",
            ...     system_prompt="You are a physics teacher",
            ...     temperature=0.7
            ... )
        """
        messages = self._build_messages(message, system_prompt, history)
        
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
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
            **kwargs: Additional OpenAI-specific arguments
            
        Returns:
            ChatResponse with content, usage stats, and metadata
        """
        messages = self._build_messages(message, system_prompt, history)
        
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        return ChatResponse(
            content=response.choices[0].message.content,
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            },
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
            **kwargs: Additional OpenAI-specific arguments
            
        Yields:
            Response tokens as they are generated
            
        Example:
            >>> client = OpenAIClient()
            >>> for token in client.stream("Write a poem"):
            ...     print(token, end="", flush=True)
        """
        messages = self._build_messages(message, system_prompt, history)
        
        stream = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            **kwargs
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    
    def responses_create(
        self,
        input_text: str,
        tools: Optional[list[dict]] = None,
        **kwargs
    ) -> str:
        """
        Use OpenAI's Responses API for agentic tasks.
        
        The Responses API is designed for single-turn tasks with
        built-in tool support (web search, file search, code interpreter).
        
        Args:
            input_text: The input prompt
            tools: List of tools to enable (e.g., [{"type": "web_search_preview"}])
            **kwargs: Additional arguments
            
        Returns:
            The response text
            
        Example:
            >>> client = OpenAIClient()
            >>> response = client.responses_create(
            ...     "What's the latest news about AI?",
            ...     tools=[{"type": "web_search_preview"}]
            ... )
        """
        params = {
            "model": self.model,
            "input": input_text,
            **kwargs
        }
        
        if tools:
            params["tools"] = tools
        
        response = self._client.responses.create(**params)
        return response.output_text
