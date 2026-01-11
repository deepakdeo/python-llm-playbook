#!/usr/bin/env python3
"""
Local LLMs with Ollama Example

Demonstrates running LLMs locally without any API keys or internet connection.

Prerequisites:
    1. Install Ollama: https://ollama.ai
    2. Pull a model: ollama pull llama3.2
    3. Ollama auto-starts, or run: ollama serve

Usage:
    python examples/local_ollama.py
"""

from llm_playbook import OllamaClient


def main():
    print("=" * 60)
    print("Local LLMs with Ollama")
    print("=" * 60)
    print("\nNo API keys needed - runs entirely on your machine!")
    print("Privacy-first: your data never leaves your computer.\n")
    
    try:
        client = OllamaClient()
        
        # List available models
        print("Checking local models...")
        local_models = client.list_local_models()
        
        if local_models:
            print(f"Found {len(local_models)} local model(s):")
            for model in local_models:
                print(f"  - {model}")
        else:
            print("No models found. Pull one with: ollama pull llama3.2")
            print("\nPopular models you can try:")
            for model in client.popular_models()[:5]:
                print(f"  - ollama pull {model}")
            return
        
        # Basic chat
        print("\n" + "-" * 40)
        print("Basic Chat:")
        print("-" * 40)
        
        response = client.chat(
            "What is Python? Answer in 2 sentences.",
            temperature=0.7
        )
        print(f"Response: {response}")
        
        # With system prompt
        print("\n" + "-" * 40)
        print("With System Prompt:")
        print("-" * 40)
        
        response = client.chat(
            "What is recursion?",
            system_prompt="You are a coding tutor for beginners. Use simple language and analogies.",
            temperature=0.7
        )
        print(f"Response: {response}")
        
        # Detailed response with stats
        print("\n" + "-" * 40)
        print("Detailed Response (with stats):")
        print("-" * 40)
        
        detailed = client.chat_with_details(
            "Name three programming languages and their main use cases.",
            max_tokens=200
        )
        print(f"Response: {detailed.content}")
        print(f"\nModel: {detailed.model}")
        if detailed.usage:
            print(f"Prompt tokens: {detailed.usage.get('prompt_tokens', 'N/A')}")
            print(f"Completion tokens: {detailed.usage.get('completion_tokens', 'N/A')}")
            
            # Convert nanoseconds to seconds for duration
            total_ns = detailed.usage.get('total_duration_ns')
            if total_ns:
                print(f"Total time: {total_ns / 1e9:.2f} seconds")
        
        # Streaming
        print("\n" + "-" * 40)
        print("Streaming Response:")
        print("-" * 40)
        print("Response: ", end="", flush=True)
        
        for token in client.stream("Write a haiku about coding."):
            print(token, end="", flush=True)
        print("\n")
        
    except Exception as e:
        error_msg = str(e)
        if "Connection refused" in error_msg or "connection" in error_msg.lower():
            print("❌ Could not connect to Ollama.")
            print("\nTo fix this:")
            print("  1. Install Ollama: https://ollama.ai")
            print("  2. Make sure it's running (it auto-starts on install)")
            print("  3. Or manually start with: ollama serve")
        else:
            print(f"❌ Error: {e}")
    
    print("\n" + "=" * 60)
    print("Benefits of Local LLMs:")
    print("  ✓ Free (no API costs)")
    print("  ✓ Private (data stays on your machine)")
    print("  ✓ Offline (works without internet)")
    print("  ✓ Fast (no network latency)")
    print("=" * 60)


if __name__ == "__main__":
    main()
