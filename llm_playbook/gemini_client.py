"""
Google Gemini client implementation.

Provides access to Google's Gemini models (Gemini 2.0, 1.5 Pro, Flash, etc.)

Documentation: https://ai.google.dev/gemini-api/docs
"""

from typing import Optional, Generator
from .base import BaseLLMClient, ChatMessage, ChatResponse
from .utils import get_api_key, logger


class GeminiClient(BaseLLMClient):
    """
    Client for Google's Gemini models.
    
    Example:
        >>> client = GeminiClient()
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
        Initialize the Gemini client.
        
        Args:
            model: Model to use (default: gemini-2.0-flash)
            api_key: Google API key (or set GOOGLE_API_KEY env var)
            **kwargs: Additional arguments
        """
        super().__init__(model=model, api_key=api_key, **kwargs)
        self._kwargs = kwargs
        self._initialize_client()
    
    @property
    def default_model(self) -> str:
        return "gemini-2.0-flash"
    
    @property
    def provider_name(self) -> str:
        return "Google Gemini"
    
    def _initialize_client(self) -> None:
        """Initialize the Gemini client."""
        from google import genai
        import os
        
        api_key = get_api_key("GOOGLE_API_KEY", self.api_key)
        # Set env var for the client to pick up
        os.environ["GOOGLE_API_KEY"] = api_key
        self._client = genai.Client()
        logger.debug(f"Initialized Gemini client with model: {self.model}")
    
    def _build_contents(
        self,
        message: str,
        history: Optional[list[ChatMessage]] = None
    ) -> list:
        """Build the contents array for the API call."""
        from google.genai import types
        
        contents = []
        
        # Add history if provided
        if history:
            for msg in history:
                # Gemini uses "user" and "model" roles
                role = "model" if msg.role == "assistant" else msg.role
                if role == "system":
                    role = "user"  # System messages go as user in contents
                contents.append(
                    types.Content(
                        role=role,
                        parts=[types.Part(text=msg.content)]
                    )
                )
        
        # Add current message
        contents.append(
            types.Content(
                role="user",
                parts=[types.Part(text=message)]
            )
        )
        
        return contents
    
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
            **kwargs: Additional Gemini-specific arguments
            
        Returns:
            The assistant's response text
            
        Example:
            >>> client = GeminiClient()
            >>> response = client.chat(
            ...     "Explain quantum computing",
            ...     system_prompt="You are a physics teacher",
            ...     temperature=0.7
            ... )
        """
        from google.genai import types
        
        contents = self._build_contents(message, history)
        
        # Build config
        config_params = {"temperature": temperature}
        if max_tokens:
            config_params["max_output_tokens"] = max_tokens
        if system_prompt:
            config_params["system_instruction"] = system_prompt
        
        config = types.GenerateContentConfig(**config_params)
        
        response = self._client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
            **kwargs
        )
        
        return response.text
    
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
            **kwargs: Additional Gemini-specific arguments
            
        Returns:
            ChatResponse with content, usage stats, and metadata
        """
        from google.genai import types
        
        contents = self._build_contents(message, history)
        
        # Build config
        config_params = {"temperature": temperature}
        if max_tokens:
            config_params["max_output_tokens"] = max_tokens
        if system_prompt:
            config_params["system_instruction"] = system_prompt
        
        config = types.GenerateContentConfig(**config_params)
        
        response = self._client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
            **kwargs
        )
        
        # Extract usage if available
        usage = None
        if response.usage_metadata:
            usage = {
                "prompt_tokens": response.usage_metadata.prompt_token_count,
                "completion_tokens": response.usage_metadata.candidates_token_count,
                "total_tokens": response.usage_metadata.total_token_count
            }
        
        return ChatResponse(
            content=response.text,
            model=self.model,
            usage=usage,
            finish_reason=str(response.candidates[0].finish_reason) if response.candidates else None,
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
            **kwargs: Additional Gemini-specific arguments
            
        Yields:
            Response tokens as they are generated
            
        Example:
            >>> client = GeminiClient()
            >>> for token in client.stream("Write a poem"):
            ...     print(token, end="", flush=True)
        """
        from google.genai import types
        
        contents = self._build_contents(message, history)
        
        # Build config
        config_params = {"temperature": temperature}
        if max_tokens:
            config_params["max_output_tokens"] = max_tokens
        if system_prompt:
            config_params["system_instruction"] = system_prompt
        
        config = types.GenerateContentConfig(**config_params)
        
        for chunk in self._client.models.generate_content_stream(
            model=self.model,
            contents=contents,
            config=config,
            **kwargs
        ):
            if chunk.text:
                yield chunk.text
