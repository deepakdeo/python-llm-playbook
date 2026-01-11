#!/usr/bin/env python3
"""
Streaming Example

Demonstrates how to stream responses token-by-token for real-time output.

Usage:
    python examples/streaming.py
"""

from llm_playbook import (
    OpenAIClient,
    AnthropicClient,
    GeminiClient,
    GroqClient,
    load_env_file
)


def main():
    load_env_file()
    
    print("=" * 60)
    print("Streaming Example - Watch responses appear in real-time")
    print("=" * 60)
    
    prompt = "Write a short poem about programming."
    
    # OpenAI Streaming
    print("\n--- OpenAI (streaming) ---")
    try:
        client = OpenAIClient()
        for token in client.stream(prompt):
            print(token, end="", flush=True)
        print("\n")
    except Exception as e:
        print(f"Error: {e}\n")
    
    # Anthropic Streaming
    print("--- Anthropic (streaming) ---")
    try:
        client = AnthropicClient()
        for token in client.stream(prompt):
            print(token, end="", flush=True)
        print("\n")
    except Exception as e:
        print(f"Error: {e}\n")
    
    # Gemini Streaming
    print("--- Gemini (streaming) ---")
    try:
        client = GeminiClient()
        for token in client.stream(prompt):
            print(token, end="", flush=True)
        print("\n")
    except Exception as e:
        print(f"Error: {e}\n")
    
    # Groq Streaming (ultra-fast!)
    print("--- Groq (streaming - watch how fast!) ---")
    try:
        client = GroqClient()
        for token in client.stream(prompt):
            print(token, end="", flush=True)
        print("\n")
    except Exception as e:
        print(f"Error: {e}\n")
    
    print("=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
