#!/usr/bin/env python3
"""
Compare Providers Example

Sends the same prompt to multiple providers and compares:
- Response quality
- Token usage
- Response time

Usage:
    python examples/compare_providers.py
"""

import time
from llm_playbook import (
    OpenAIClient,
    AnthropicClient,
    GeminiClient,
    GroqClient,
    load_env_file
)


def benchmark_provider(name: str, client, prompt: str) -> dict:
    """Benchmark a single provider."""
    try:
        start_time = time.time()
        response = client.chat_with_details(prompt, temperature=0.7)
        elapsed = time.time() - start_time
        
        return {
            "provider": name,
            "model": response.model,
            "response": response.content[:200] + "..." if len(response.content) > 200 else response.content,
            "usage": response.usage,
            "time_seconds": round(elapsed, 2),
            "success": True
        }
    except Exception as e:
        return {
            "provider": name,
            "error": str(e),
            "success": False
        }


def main():
    load_env_file()
    
    print("=" * 70)
    print("Provider Comparison - Same prompt, different LLMs")
    print("=" * 70)
    
    prompt = "Explain the concept of recursion in programming. Give a simple example."
    print(f"\nPrompt: {prompt}\n")
    
    # Initialize clients
    providers = [
        ("OpenAI", OpenAIClient()),
        ("Anthropic", AnthropicClient()),
        ("Gemini", GeminiClient()),
        ("Groq", GroqClient()),
    ]
    
    results = []
    
    for name, client in providers:
        print(f"Testing {name}...", end=" ", flush=True)
        result = benchmark_provider(name, client, prompt)
        results.append(result)
        if result["success"]:
            print(f"✓ ({result['time_seconds']}s)")
        else:
            print(f"✗ ({result['error'][:50]}...)")
    
    # Display results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    for result in results:
        print(f"\n{'─' * 50}")
        print(f"Provider: {result['provider']}")
        
        if result["success"]:
            print(f"Model: {result['model']}")
            print(f"Time: {result['time_seconds']} seconds")
            if result.get("usage"):
                usage = result["usage"]
                if "total_tokens" in usage:
                    print(f"Tokens: {usage.get('total_tokens', 'N/A')}")
                elif "input_tokens" in usage:
                    total = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
                    print(f"Tokens: {total}")
            print(f"\nResponse:\n{result['response']}")
        else:
            print(f"Error: {result['error']}")
    
    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Provider':<15} {'Time (s)':<12} {'Status'}")
    print("-" * 40)
    
    for result in results:
        status = "✓ Success" if result["success"] else "✗ Failed"
        time_str = str(result.get("time_seconds", "N/A"))
        print(f"{result['provider']:<15} {time_str:<12} {status}")


if __name__ == "__main__":
    main()
