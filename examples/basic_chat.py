#!/usr/bin/env python3
"""
Basic Chat Example

Demonstrates simple chat interactions with each LLM provider.

Usage:
    python examples/basic_chat.py
    
Make sure you have set your API keys in .env or environment variables.
"""

from llm_playbook import (
    OpenAIClient,
    AnthropicClient,
    GeminiClient,
    GroqClient,
    OllamaClient,
    load_env_file
)


def main():
    # Load environment variables from .env file
    load_env_file()
    
    prompt = "What is the capital of France? Answer in one sentence."
    
    print("=" * 60)
    print("Basic Chat Example - Same prompt to all providers")
    print("=" * 60)
    print(f"\nPrompt: {prompt}\n")
    
    # OpenAI
    print("-" * 40)
    print("OpenAI (GPT-4o-mini):")
    try:
        client = OpenAIClient()
        response = client.chat(prompt)
        print(f"  {response}")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Anthropic
    print("-" * 40)
    print("Anthropic (Claude Sonnet):")
    try:
        client = AnthropicClient()
        response = client.chat(prompt)
        print(f"  {response}")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Gemini
    print("-" * 40)
    print("Google Gemini (2.0 Flash):")
    try:
        client = GeminiClient()
        response = client.chat(prompt)
        print(f"  {response}")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Groq
    print("-" * 40)
    print("Groq (Llama 3.3 70B):")
    try:
        client = GroqClient()
        response = client.chat(prompt)
        print(f"  {response}")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Ollama (local)
    print("-" * 40)
    print("Ollama (Local - llama3.2):")
    try:
        client = OllamaClient()
        response = client.chat(prompt)
        print(f"  {response}")
    except Exception as e:
        print(f"  Error: {e}")
        print("  (Make sure Ollama is running: https://ollama.ai)")
    
    print("\n" + "=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
