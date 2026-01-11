"""
Utility functions for the LLM Playbook.
"""

import os
import logging
from typing import Optional
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("llm_playbook")


def load_env_file(env_path: Optional[str] = None) -> None:
    """
    Load environment variables from a .env file.
    
    Args:
        env_path: Path to .env file. If None, looks for .env in current directory.
    """
    try:
        from dotenv import load_dotenv
        
        if env_path:
            load_dotenv(env_path)
        else:
            # Look for .env in current directory and parent directories
            current = Path.cwd()
            for path in [current, *current.parents]:
                env_file = path / ".env"
                if env_file.exists():
                    load_dotenv(env_file)
                    logger.debug(f"Loaded environment from {env_file}")
                    return
            # Try default location
            load_dotenv()
    except ImportError:
        logger.warning("python-dotenv not installed. Install with: pip install python-dotenv")


def get_api_key(
    env_var: str,
    api_key: Optional[str] = None,
    required: bool = True
) -> Optional[str]:
    """
    Get an API key from environment variable or passed value.
    
    Args:
        env_var: Name of the environment variable
        api_key: Explicitly passed API key (takes precedence)
        required: Whether to raise an error if key is not found
        
    Returns:
        The API key string
        
    Raises:
        ValueError: If required=True and no key is found
    """
    key = api_key or os.environ.get(env_var)
    
    if not key and required:
        raise ValueError(
            f"API key not found. Either:\n"
            f"  1. Set the {env_var} environment variable\n"
            f"  2. Pass api_key parameter to the client\n"
            f"  3. Create a .env file with {env_var}=your-key"
        )
    
    return key


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to a maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length (including suffix)
        suffix: String to append if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def count_tokens_approximate(text: str) -> int:
    """
    Approximate token count (rough estimate: ~4 chars per token).
    
    For accurate counts, use the provider's tokenizer.
    
    Args:
        text: Text to count tokens for
        
    Returns:
        Approximate token count
    """
    return len(text) // 4


def format_messages_for_display(messages: list, max_content_length: int = 50) -> str:
    """
    Format a list of messages for debugging display.
    
    Args:
        messages: List of message dictionaries
        max_content_length: Maximum content length to display
        
    Returns:
        Formatted string representation
    """
    lines = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        truncated = truncate_text(content, max_content_length)
        lines.append(f"  [{role}]: {truncated}")
    return "\n".join(lines)


class LLMError(Exception):
    """Base exception for LLM-related errors."""
    pass


class APIKeyError(LLMError):
    """Raised when an API key is missing or invalid."""
    pass


class RateLimitError(LLMError):
    """Raised when rate limit is exceeded."""
    pass


class ModelNotFoundError(LLMError):
    """Raised when the specified model is not found."""
    pass
