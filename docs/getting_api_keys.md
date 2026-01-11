# Getting API Keys

This guide walks you through obtaining API keys from each supported LLM provider.

## Table of Contents

- [OpenAI](#openai)
- [Anthropic](#anthropic)
- [Google Gemini](#google-gemini)
- [Groq](#groq)
- [Ollama](#ollama-local)

---

## OpenAI

**Website:** https://platform.openai.com

**Free Tier:** $5 credit for new accounts (expires after 3 months)

### Steps:

1. Go to [platform.openai.com](https://platform.openai.com)
2. Click "Sign Up" (or "Log In" if you have an account)
3. Navigate to **API Keys** in the left sidebar
4. Click **"Create new secret key"**
5. Give it a name (e.g., "LLM Playbook")
6. Copy the key immediately (you won't see it again!)

### Environment Variable:
```bash
export OPENAI_API_KEY="sk-..."
```

### Available Models:
- `gpt-4o` - Latest flagship model
- `gpt-4o-mini` - Fast and cheap (recommended for testing)
- `gpt-4-turbo` - Previous generation
- `gpt-3.5-turbo` - Fast and affordable

### Pricing:
See [openai.com/pricing](https://openai.com/pricing)

---

## Anthropic

**Website:** https://console.anthropic.com

**Free Tier:** $5 credit for new accounts

### Steps:

1. Go to [console.anthropic.com](https://console.anthropic.com)
2. Click "Sign Up" or "Log In"
3. Navigate to **Settings** → **API Keys**
4. Click **"Create Key"**
5. Copy the key (starts with `sk-ant-`)

### Environment Variable:
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

### Available Models:
- `claude-sonnet-4-20250514` - Latest Sonnet (balanced)
- `claude-opus-4-20250514` - Most capable
- `claude-3-5-sonnet-20241022` - Previous Sonnet
- `claude-3-haiku-20240307` - Fast and cheap

### Pricing:
See [anthropic.com/pricing](https://www.anthropic.com/pricing)

---

## Google Gemini

**Website:** https://aistudio.google.com

**Free Tier:** Generous free tier with high rate limits

### Steps:

1. Go to [aistudio.google.com](https://aistudio.google.com)
2. Sign in with your Google account
3. Click **"Get API Key"** in the top right
4. Click **"Create API Key"**
5. Select a Google Cloud project (or create one)
6. Copy the key

### Environment Variable:
```bash
export GOOGLE_API_KEY="AI..."
```

### Available Models:
- `gemini-2.0-flash` - Latest fast model (recommended)
- `gemini-1.5-pro` - Most capable
- `gemini-1.5-flash` - Fast and efficient
- `gemini-1.5-flash-8b` - Smallest/fastest

### Pricing:
See [ai.google.dev/pricing](https://ai.google.dev/pricing)

---

## Groq

**Website:** https://console.groq.com

**Free Tier:** Free tier with generous rate limits

### Steps:

1. Go to [console.groq.com](https://console.groq.com)
2. Click "Sign Up" with Google, GitHub, or email
3. Navigate to **API Keys** in the sidebar
4. Click **"Create API Key"**
5. Copy the key (starts with `gsk_`)

### Environment Variable:
```bash
export GROQ_API_KEY="gsk_..."
```

### Available Models:
- `llama-3.3-70b-versatile` - Latest Llama (recommended)
- `llama-3.1-70b-versatile` - Previous Llama
- `llama-3.1-8b-instant` - Fast small model
- `mixtral-8x7b-32768` - Mixtral MoE
- `gemma2-9b-it` - Google Gemma

### Why Groq?
Groq specializes in ultra-fast inference using custom LPU hardware. Response times are often 10x faster than other providers, making it ideal for real-time applications.

### Pricing:
See [groq.com/pricing](https://groq.com/pricing)

---

## Ollama (Local)

**Website:** https://ollama.ai

**Free:** Completely free (runs on your hardware)

### Steps:

1. **Download and Install:**
   - **macOS:** `brew install ollama` or download from [ollama.ai](https://ollama.ai)
   - **Linux:** `curl -fsSL https://ollama.ai/install.sh | sh`
   - **Windows:** Download installer from [ollama.ai](https://ollama.ai)

2. **Pull a model:**
   ```bash
   ollama pull llama3.2
   ```

3. **Start using** (Ollama auto-starts on install):
   ```bash
   # Or manually start
   ollama serve
   ```

### No API Key Needed!
Ollama runs entirely on your local machine. No API key, no internet, no costs.

### Popular Models:
```bash
ollama pull llama3.2        # Meta's latest (3B, 8B)
ollama pull llama3.2:1b     # Smallest/fastest
ollama pull mistral         # Mistral 7B
ollama pull codellama       # Code-specialized
ollama pull phi3            # Microsoft Phi-3
ollama pull gemma2          # Google Gemma 2
```

### Hardware Requirements:
- **Minimum:** 8GB RAM for 7B models
- **Recommended:** 16GB+ RAM for 13B+ models
- **GPU:** Optional but significantly faster (NVIDIA, Apple Silicon)

---

## Setting Up Your Environment

### Option 1: .env File (Recommended)

Create a `.env` file in your project root:

```bash
cp .env.example .env
```

Edit `.env` with your keys:

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AI...
GROQ_API_KEY=gsk_...
```

### Option 2: Export in Shell

Add to your `~/.bashrc`, `~/.zshrc`, or equivalent:

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="AI..."
export GROQ_API_KEY="gsk_..."
```

Then reload:
```bash
source ~/.bashrc  # or ~/.zshrc
```

### Option 3: Google Colab Secrets

In Google Colab:
1. Click the 🔑 key icon in the left sidebar
2. Add each key as a secret
3. Access in code:
   ```python
   from google.colab import userdata
   import os
   
   os.environ['OPENAI_API_KEY'] = userdata.get('OPENAI_API_KEY')
   ```

---

## Security Best Practices

1. **Never commit API keys to Git**
   - Use `.env` files (already in `.gitignore`)
   - Use environment variables

2. **Rotate keys regularly**
   - Delete unused keys
   - Create new keys for different projects

3. **Set usage limits**
   - Most providers allow setting spending limits
   - Set alerts for unusual usage

4. **Use separate keys for dev/prod**
   - Easier to track usage
   - Limit blast radius if compromised
