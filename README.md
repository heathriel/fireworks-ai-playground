# Fireworks AI Playground

Example scripts and tests for the [Fireworks AI](https://fireworks.ai) API: chat, streaming, tool calling, structured output, reasoning, and vision.

## Prerequisites

- Python 3.9+
- A [Fireworks](https://fireworks.ai) account and API key

## Setup

**Get your API key:** In the [Fireworks dashboard](https://app.fireworks.ai/settings/users/api-keys), go to **Settings → API keys**, click **Create API key**, and copy the key. Store it somewhere safe (it’s shown only once).

Then install and set the key:

```bash
pip install -r requirements.txt
export FIREWORKS_API_KEY="your-api-key"
```

## Scripts

| Script | Description |
|--------|-------------|
| `fireworks_chat.py` | Tool calling: ask for weather, run a stub `get_weather` tool, get a natural-language answer (no real API). |
| `fireworks_chat_real_weather.py` | Same flow with real weather via [Open-Meteo](https://open-meteo.com) (geocoding + current conditions). |
| `fireworks_stream.py` | Streaming chat completion with a safe chunk guard. |
| `fireworks_structured.py` | Structured JSON output using `response_format` and a JSON schema (e.g. extract name/age). |
| `fireworks_reasoning.py` | Reasoning model (DeepSeek v3p2) with `reasoning_effort`; prints reasoning and final answer. |
| `fireworks_vision.py` | Vision model (Kimi K2.5) with an image URL; describes the image. |

Run any script:

```bash
python3 fireworks_chat.py
python3 fireworks_vision.py
# etc.
```

## Tests

Pytest suite for the same features (chat, streaming, tools, structured output, reasoning, vision, and error handling):

```bash
# Run all tests (requires FIREWORKS_API_KEY)
pytest tests/test_fireworks.py -v

# Without an API key: 7 tests are skipped, 1 (401 invalid key) runs
pytest tests/test_fireworks.py -v
```

Tests use serverless models. Without `FIREWORKS_API_KEY`, only `test_error_401_invalid_api_key` runs; the rest are skipped.

## Links

- [Fireworks docs](https://docs.fireworks.ai)
- [Which model to use](https://docs.fireworks.ai/guides/recommended-models)
- [API reference](https://docs.fireworks.ai/api-reference/post-chatcompletions)

## Docs & troubleshooting

- **[Inference error codes](https://docs.fireworks.ai/guides/inference-error-codes)** — 400, 401, 404, 429, 503, etc., and how to fix them.
- **[Rate limits & quotas](https://docs.fireworks.ai/guides/quotas_usage/rate-limits)** — Serverless limits, `x-ratelimit-remaining-requests` header, and when 429 means “capacity” vs “quota”.
- **[OpenAI compatibility](https://docs.fireworks.ai/tools-sdks/openai-compatibility)** — Use the OpenAI SDK with Fireworks by setting `base_url` and `api_key`.

Other guides worth knowing: [Batch API](https://docs.fireworks.ai/guides/batch-inference) (async, lower cost), [Responses API](https://docs.fireworks.ai/guides/response-api) (stateful + MCP tools), [prompt caching](https://docs.fireworks.ai/guides/prompt-caching), [video & audio inputs](https://docs.fireworks.ai/guides/video-audio-inputs), [embeddings & reranking](https://docs.fireworks.ai/guides/querying-embeddings-models), [speech-to-text](https://docs.fireworks.ai/guides/querying-asr-models).
