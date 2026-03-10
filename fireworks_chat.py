import json
from fireworks import Fireworks

client = Fireworks()


def get_weather(location: str) -> str:
    """Return a simple weather string for the model to parse."""
    # Just give Kimi something to parse — no API needed
    return f"{location}: 18°C, partly cloudy."


messages = [
    {"role": "user", "content": "What's the weather in Paris?"}
]

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

# First call: model may return tool_calls
response = client.chat.completions.create(
    model="accounts/fireworks/models/kimi-k2-instruct-0905",
    messages=messages,
    tools=tools,
)

msg = response.choices[0].message
messages.append(
    {"role": "assistant", "content": msg.content or "", "tool_calls": msg.tool_calls}
)

# Execute tool(s) and append tool results
if msg.tool_calls:
    for tc in msg.tool_calls:
        args = json.loads(tc.function.arguments)
        location = args.get("location", "")
        result = get_weather(location)
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            }
        )

    # Second call: model uses tool results to answer
    response2 = client.chat.completions.create(
        model="accounts/fireworks/models/kimi-k2-instruct-0905",
        messages=messages,
        tools=tools,
    )
    final = response2.choices[0].message
    print(final.content)
else:
    print(msg.content or "(no content)")