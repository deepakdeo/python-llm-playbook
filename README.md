# Python LLM API Playbook

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/license/mit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A unified, modular Python interface for working with multiple Large Language Model (LLM) providers. This project provides consistent patterns for accessing OpenAI, Anthropic (Claude), Google Gemini, Groq, and local models via Ollama.

## Features

- **Unified Interface**: Consistent patterns across all providers
- **Production Ready**: Type hints, error handling, and logging
- **Well Documented**: Examples, notebooks, and inline documentation
- **Extensible**: Easy to add new providers
- **No Lock-in**: Swap providers with minimal code changes

## Supported Providers

| Provider | Models | Highlights |
|----------|--------|------------|
| **OpenAI** | GPT-4o, GPT-4o-mini, GPT-4 | Industry standard, function calling |
| **Anthropic** | Claude 4, Claude 3.5 Sonnet | Excellent reasoning, long context |
| **Google Gemini** | Gemini 2.0, 1.5 Pro/Flash | Multimodal, generous free tier |
| **Groq** | Llama, Mixtral, Gemma | Ultra-fast inference, free tier |
| **Ollama** | Llama, Mistral, Phi, etc. | Local/offline, privacy-first |

## Installation

### From Source

```bash
git clone https://github.com/deepakdeo/python-llm-playbook.git
cd python-llm-playbook
pip install -e .
```

### Dependencies Only

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Set Up API Keys

Copy the example environment file and add your API keys:

```bash
cp .env.example .env
# Edit .env with your API keys
```

Or export them directly:

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="AI..."
export GROQ_API_KEY="gsk_..."
```

### 2. Basic Usage

```python
from llm_playbook import OpenAIClient, AnthropicClient, GeminiClient, GroqClient

# OpenAI
openai_client = OpenAIClient()
response = openai_client.chat("What is machine learning?")
print(response)

# Anthropic (Claude)
claude_client = AnthropicClient()
response = claude_client.chat("Explain quantum computing simply.")
print(response)

# Google Gemini
gemini_client = GeminiClient()
response = gemini_client.chat("Write a haiku about Python.")
print(response)

# Groq (ultra-fast)
groq_client = GroqClient()
response = groq_client.chat("What is the capital of France?")
print(response)
```

### 3. With System Prompts and Parameters

```python
from llm_playbook import OpenAIClient

client = OpenAIClient()

response = client.chat(
    message="Explain APIs to a 10-year-old.",
    system_prompt="You are a friendly teacher who uses simple analogies.",
    temperature=0.7,
    max_tokens=200
)
print(response)
```

### 4. Multi-turn Conversations

```python
from llm_playbook import OpenAIClient

client = OpenAIClient()

# Build conversation history
history = []
history = client.chat(
    message="What's the closest star to Earth?",
    history=history,
    return_history=True
)
# Continue the conversation
history = client.chat(
    message="Does it have any planets?",
    history=history,
    return_history=True
)
```

### 5. Local LLMs with Ollama

```python
from llm_playbook import OllamaClient

# Requires Ollama running locally: https://ollama.com
client = OllamaClient(model="llama3.2")
response = client.chat("What is Python?")
print(response)
```

## Project Structure

```
python-llm-playbook/
├── README.md                 # This file
├── requirements.txt          # Dependencies
├── setup.py                  # Package installation
├── .env.example              # API key template
├── .gitignore                # Git ignore rules
│
├── llm_playbook/             # Main package
│   ├── __init__.py           # Package exports
│   ├── base.py               # Abstract base class
│   ├── openai_client.py      # OpenAI implementation
│   ├── anthropic_client.py   # Anthropic implementation
│   ├── gemini_client.py      # Google Gemini implementation
│   ├── groq_client.py        # Groq implementation
│   ├── ollama_client.py      # Ollama implementation
│   └── utils.py              # Shared utilities
│
├── examples/                 # Standalone example scripts
│   ├── basic_chat.py
│   ├── multi_turn.py
│   ├── streaming.py
│   ├── compare_providers.py
│   └── local_ollama.py
│
├── notebooks/                # Interactive Jupyter notebooks
│   ├── 01_openai.ipynb
│   ├── 02_anthropic.ipynb
│   ├── 03_gemini.ipynb
│   ├── 04_groq.ipynb
│   ├── 05_ollama.ipynb
│   └── 06_comparison.ipynb
│
├── tests/                    # Unit tests
│   └── test_clients.py
│
└── docs/                     # Additional documentation
    └── getting_api_keys.md
```

## Getting API Keys

| Provider | Sign Up | Free Tier |
|----------|---------|-----------|
| OpenAI | [platform.openai.com](https://platform.openai.com/signup) | Credits may apply (see pricing) |
| Anthropic | [console.anthropic.com](https://console.anthropic.com/) | Free credits for new users (see console) |
| Google Gemini | [aistudio.google.com](https://aistudio.google.com/) | Generous free tier |
| Groq | [console.groq.com](https://console.groq.com/) | Free tier available |
| Ollama | [ollama.com](https://ollama.com/) | Free (runs locally) |

See [docs/getting_api_keys.md](docs/getting_api_keys.md) for detailed setup instructions.

## Notebooks

Interactive tutorials are available in the `notebooks/` directory. Run them locally or in Google Colab:

| Notebook | Description | Colab |
|----------|-------------|-------|
| `01_openai.ipynb` | OpenAI GPT models | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/deepakdeo/python-llm-playbook/blob/main/notebooks/01_openai.ipynb) |
| `02_anthropic.ipynb` | Anthropic Claude | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/deepakdeo/python-llm-playbook/blob/main/notebooks/02_anthropic.ipynb) |
| `03_gemini.ipynb` | Google Gemini | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/deepakdeo/python-llm-playbook/blob/main/notebooks/03_gemini.ipynb) |
| `04_groq.ipynb` | Groq (fast inference) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/deepakdeo/python-llm-playbook/blob/main/notebooks/04_groq.ipynb) |
| `05_ollama.ipynb` | Local LLMs | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/deepakdeo/python-llm-playbook/blob/main/notebooks/05_ollama.ipynb) |
| `06_comparison.ipynb` | Side-by-side comparison | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/deepakdeo/python-llm-playbook/blob/main/notebooks/06_comparison.ipynb) |

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

**Note:** Models listed in the supported providers table are examples only—model availability changes frequently. Refer to each provider’s documentation for the latest models and versions. Free tier or promotional credits may vary over time.

- [OpenAI](https://openai.com/) for GPT models
- [Anthropic](https://anthropic.com/) for Claude
- [Google](https://ai.google.dev/) for Gemini
- [Groq](https://groq.com/) for ultra-fast inference
- [Ollama](https://ollama.com/) for local LLM support
