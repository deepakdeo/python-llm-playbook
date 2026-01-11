#!/usr/bin/env python3
"""
Multi-turn Conversation Example

Demonstrates how to maintain conversation history across multiple exchanges.

Usage:
    python examples/multi_turn.py
"""

from llm_playbook import OpenAIClient, ChatMessage, load_env_file


def main():
    load_env_file()
    
    print("=" * 60)
    print("Multi-turn Conversation Example")
    print("=" * 60)
    
    client = OpenAIClient()
    
    # System prompt sets the assistant's behavior
    system_prompt = "You are a helpful astronomy expert. Be concise but informative."
    
    # We'll manually track conversation history
    history = []
    
    # Turn 1
    print("\n[User]: What's the closest star to Earth?")
    response1 = client.chat(
        message="What's the closest star to Earth?",
        system_prompt=system_prompt,
        history=history
    )
    print(f"[Assistant]: {response1}")
    
    # Add to history
    history.append(ChatMessage(role="user", content="What's the closest star to Earth?"))
    history.append(ChatMessage(role="assistant", content=response1))
    
    # Turn 2 - follows up on previous context
    print("\n[User]: Does it have any planets?")
    response2 = client.chat(
        message="Does it have any planets?",
        system_prompt=system_prompt,
        history=history
    )
    print(f"[Assistant]: {response2}")
    
    # Add to history
    history.append(ChatMessage(role="user", content="Does it have any planets?"))
    history.append(ChatMessage(role="assistant", content=response2))
    
    # Turn 3 - continues the conversation
    print("\n[User]: Could humans ever travel there?")
    response3 = client.chat(
        message="Could humans ever travel there?",
        system_prompt=system_prompt,
        history=history
    )
    print(f"[Assistant]: {response3}")
    
    print("\n" + "=" * 60)
    print(f"Conversation had {len(history) // 2 + 1} turns")
    print("=" * 60)


if __name__ == "__main__":
    main()
