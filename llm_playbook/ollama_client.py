"""
Ollama client implementation.

Provides access to local LLMs running via Ollama (Llama, Mistral, Phi, etc.)

Ollama must be installed and running locally: https://ollama.ai

Documentation: https://github.com/ollama/ollama-python
"""

from typing import Optional, Generator
from .base import BaseLLMClient, ChatMessage, ChatResponse
from .utils import logger


class OllamaClient(BaseLLMClient):
    """
    Client for local LLMs via Ollama.
    
    Ollama allows you to run LLMs locally without API keys or internet.
    Models run entirely on your machine for privacy and offline use.
    
    Prerequisites:
        1. Install Ollama: https://ollama.ai
        2. Pull a model: `ollama pull llama3.2`
        3. Ollama runs automatically, or start with: `ollama serve`
    
    Example:
        >>> client = OllamaClient()
        >>> response = client.chat("What is machine learning?")
        >>> print(response)
    """
    
    def __init__(
        self,
        model: Optional[str] = None,
        host: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the Ollama client.
        
        Args:
            model: Model to use (default: llama3.2)
            host: Ollama server host (default: http://localhost:11434)
            **kwargs: Additional arguments
        """
        # Ollama doesn't need an API key
        super().__init__(model=model, api_key=None, **kwargs)
        self._host = host
        self._kwargs = kwargs
        self._initialize_client()
    
    @property
    def default_model(self) -> str:
        return "llama3.2"
    
    @property
    def provider_name(self) -> str:
        return "Ollama (Local)"
    
    def _initialize_client(self) -> None:
        """Initialize the Ollama client."""
        from ollama import Client
        
        if self._host:
            self._client = Client(host=self._host)
        else:
            self._client = Client()
        
        logger.debug(f"Initialized Ollama client with model: {self.model}")
    
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
        temperature: float = 0.8,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Send a chat message and get a response.
        
        Args:
            message: The user's message
            system_prompt: Optional system prompt to set behavior
            history: Optional conversation history
            temperature: Randomness (0.0-2.0, default: 0.8)
            max_tokens: Maximum tokens in response (num_predict)
            **kwargs: Additional Ollama-specific arguments
            
        Returns:
            The assistant's response text
            
        Example:
            >>> client = OllamaClient()
            >>> response = client.chat(
            ...     "Explain quantum computing",
            ...     system_prompt="You are a physics teacher",
            ...     temperature=0.7
            ... )
        """
        messages = self._build_messages(message, system_prompt, history)
        
        options = {"temperature": temperature}
        if max_tokens:
            options["num_predict"] = max_tokens
        
        response = self._client.chat(
            model=self.model,
            messages=messages,
            options=options,
            **kwargs
        )
        
        return response["message"]["content"]
    
    def chat_with_details(
        self,
        message: str,
        system_prompt: Optional[str] = None,
        history: Optional[list[ChatMessage]] = None,
        temperature: float = 0.8,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> ChatResponse:
        """
        Send a chat message and get a detailed response.
        
        Args:
            message: The user's message
            system_prompt: Optional system prompt to set behavior
            history: Optional conversation history
            temperature: Randomness (0.0-2.0, default: 0.8)
            max_tokens: Maximum tokens in response (num_predict)
            **kwargs: Additional Ollama-specific arguments
            
        Returns:
            ChatResponse with content, usage stats, and metadata
        """
        messages = self._build_messages(message, system_prompt, history)
        
        options = {"temperature": temperature}
        if max_tokens:
            options["num_predict"] = max_tokens
        
        response = self._client.chat(
            model=self.model,
            messages=messages,
            options=options,
            **kwargs
        )
        
        return ChatResponse(
            content=response["message"]["content"],
            model=response.get("model", self.model),
            usage={
                "prompt_tokens": response.get("prompt_eval_count"),
                "completion_tokens": response.get("eval_count"),
                "total_duration_ns": response.get("total_duration"),
                "load_duration_ns": response.get("load_duration"),
                "prompt_eval_duration_ns": response.get("prompt_eval_duration"),
                "eval_duration_ns": response.get("eval_duration")
            },
            finish_reason=response.get("done_reason"),
            raw_response=response
        )
    
    def stream(
        self,
        message: str,
        system_prompt: Optional[str] = None,
        history: Optional[list[ChatMessage]] = None,
        temperature: float = 0.8,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """
        Stream a chat response token by token.
        
        Args:
            message: The user's message
            system_prompt: Optional system prompt to set behavior
            history: Optional conversation history
            temperature: Randomness (0.0-2.0, default: 0.8)
            max_tokens: Maximum tokens in response (num_predict)
            **kwargs: Additional Ollama-specific arguments
            
        Yields:
            Response tokens as they are generated
            
        Example:
            >>> client = OllamaClient()
            >>> for token in client.stream("Write a poem"):
            ...     print(token, end="", flush=True)
        """
        messages = self._build_messages(message, system_prompt, history)
        
        options = {"temperature": temperature}
        if max_tokens:
            options["num_predict"] = max_tokens
        
        stream = self._client.chat(
            model=self.model,
            messages=messages,
            options=options,
            stream=True,
            **kwargs
        )
        
        for chunk in stream:
            if chunk["message"]["content"]:
                yield chunk["message"]["content"]
    
    def list_local_models(self) -> list[str]:
        """
        List all locally available models.
        
        Returns:
            List of model names that are downloaded locally
            
        Example:
            >>> client = OllamaClient()
            >>> models = client.list_local_models()
            >>> print(models)
            ['llama3.2:latest', 'mistral:latest', 'codellama:latest']
        """
        response = self._client.list()
        return [model["name"] for model in response.get("models", [])]
    
    def pull_model(self, model_name: str) -> None:
        """
        Download a model from the Ollama library.
        
        Args:
            model_name: Name of the model to download (e.g., 'llama3.2', 'mistral')
            
        Example:
            >>> client = OllamaClient()
            >>> client.pull_model("llama3.2")
        """
        logger.info(f"Pulling model: {model_name}")
        self._client.pull(model_name)
        logger.info(f"Successfully pulled: {model_name}")
    
    @staticmethod
    def popular_models() -> list[str]:
        """
        Return a list of popular Ollama models.
        
        These can be pulled with `ollama pull <model_name>` or
        using the pull_model() method.
        
        Returns:
            List of popular model names
        """
        return [
            "llama3.2",          # Meta's latest Llama model
            "llama3.2:1b",       # Smaller 1B parameter version
            "llama3.1",          # Previous Llama version
            "mistral",           # Mistral 7B
            "mixtral",           # Mixtral 8x7B
            "codellama",         # Code-specialized Llama
            "phi3",              # Microsoft Phi-3
            "gemma2",            # Google Gemma 2
            "qwen2.5",           # Alibaba Qwen 2.5
            "deepseek-coder",    # DeepSeek for coding
        ]
