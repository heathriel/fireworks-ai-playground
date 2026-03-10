"""Tests for Fireworks API: chat, streaming, tools, structured output, reasoning, vision, errors."""
import json
import os

import pytest

from fireworks import Fireworks

# Models used by scripts (must be available on serverless)
MODEL_CHAT = "accounts/fireworks/models/deepseek-v3p1"
MODEL_TOOLS = "accounts/fireworks/models/kimi-k2-instruct-0905"
MODEL_STRUCTURED = "accounts/fireworks/models/deepseek-v3p1"
MODEL_REASONING = "accounts/fireworks/models/deepseek-v3p2"
MODEL_VISION = "accounts/fireworks/models/kimi-k2p5"

# Skip tests that need a valid API key (error tests run without it or with invalid key)
_has_key = bool(os.environ.get("FIREWORKS_API_KEY"))
skip_no_key = pytest.mark.skipif(not _has_key, reason="FIREWORKS_API_KEY not set")


def _client():
    return Fireworks()


# --- 1. Chat ---
@skip_no_key
def test_chat_returns_non_empty_content():
    client = _client()
    response = client.chat.completions.create(
        model=MODEL_CHAT,
        messages=[{"role": "user", "content": "Say hello in one word."}],
    )
    assert response.choices
    msg = response.choices[0].message
    assert msg.content is not None
    assert isinstance(msg.content, str)
    assert len(msg.content.strip()) > 0


# --- 2. Streaming ---
@skip_no_key
def test_streaming_yields_chunks_and_final_usage():
    client = _client()
    stream = client.chat.completions.create(
        model=MODEL_CHAT,
        messages=[{"role": "user", "content": "Say hi."}],
        stream=True,
    )
    chunks = list(stream)
    assert len(chunks) >= 1
    # Guard: only read delta when choices present
    for chunk in chunks:
        if chunk.choices and chunk.choices[0].delta.content:
            assert isinstance(chunk.choices[0].delta.content, str)
    # Final chunk typically has usage
    last = chunks[-1]
    if getattr(last, "usage", None):
        assert last.usage.total_tokens >= 0


# --- 3. Tool calling ---
@skip_no_key
def test_tool_calling_flow():
    client = _client()

    def get_weather(location: str) -> str:
        return f"{location}: 18°C, partly cloudy."

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City name, e.g. San Francisco",
                        }
                    },
                    "required": ["location"],
                },
            },
        },
    ]
    messages = [{"role": "user", "content": "What's the weather in Paris?"}]

    response = client.chat.completions.create(
        model=MODEL_TOOLS,
        messages=messages,
        tools=tools,
    )
    msg = response.choices[0].message
    assert msg.tool_calls, "Model should request get_weather tool"
    messages.append(
        {"role": "assistant", "content": msg.content or "", "tool_calls": msg.tool_calls}
    )
    for tc in msg.tool_calls:
        args = json.loads(tc.function.arguments)
        result = get_weather(args.get("location", ""))
        messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    response2 = client.chat.completions.create(
        model=MODEL_TOOLS,
        messages=messages,
        tools=tools,
    )
    final = response2.choices[0].message
    assert final.content is not None
    assert len(final.content.strip()) > 0


# --- 4. Structured output ---
@skip_no_key
def test_structured_output_json_schema():
    client = _client()
    response = client.chat.completions.create(
        model=MODEL_STRUCTURED,
        messages=[
            {
                "role": "user",
                "content": "Extract the name and age from: John is 30 years old",
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "person",
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "number"},
                    },
                    "required": ["name", "age"],
                },
            },
        },
    )
    raw = response.choices[0].message.content
    assert raw is not None
    data = json.loads(raw.strip())
    assert "name" in data
    assert "age" in data
    assert isinstance(data["name"], str)
    assert isinstance(data["age"], (int, float))


# --- 5. Reasoning ---
@skip_no_key
def test_reasoning_returns_reasoning_and_answer():
    client = _client()
    response = client.chat.completions.create(
        model=MODEL_REASONING,
        messages=[{"role": "user", "content": "What is 25 * 37? Show your work."}],
        reasoning_effort="medium",
    )
    msg = response.choices[0].message
    # At least one of reasoning or content should be present
    assert msg.reasoning_content is not None or msg.content is not None
    if msg.content:
        assert "925" in msg.content


# --- 6. Vision ---
@skip_no_key
def test_vision_image_url_returns_content():
    client = _client()
    response = client.chat.completions.create(
        model=MODEL_VISION,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://storage.googleapis.com/fireworks-public/image_assets/fireworks-ai-wordmark-color-dark.png"
                        },
                    },
                ],
            }
        ],
    )
    msg = response.choices[0].message
    assert msg.content is not None
    assert len(msg.content.strip()) > 0


# --- 7. Errors ---
@skip_no_key
def test_error_404_invalid_model():
    from fireworks import NotFoundError

    client = _client()
    with pytest.raises(NotFoundError):
        client.chat.completions.create(
            model="accounts/fireworks/models/nonexistent-model-404",
            messages=[{"role": "user", "content": "Hi"}],
        )


def test_error_401_invalid_api_key():
    from fireworks import AuthenticationError

    client = Fireworks(api_key="invalid-key-for-test")
    with pytest.raises(AuthenticationError):
        client.chat.completions.create(
            model=MODEL_CHAT,
            messages=[{"role": "user", "content": "Hi"}],
        )
