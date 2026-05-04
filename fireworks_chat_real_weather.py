"""Load API key from .env if available."""
from dotenv import load_dotenv
load_dotenv()


import json
import urllib.request
from fireworks import Fireworks

client = Fireworks()


def get_weather(location: str) -> str:
    """Fetch current weather for a location using Open-Meteo (no API key)."""
    try:
        # Geocode city name to lat/lon
        name = urllib.parse.quote(location)
        with urllib.request.urlopen(
            f"https://geocoding-api.open-meteo.com/v1/search?name={name}&count=1",
            timeout=10,
        ) as r:
            data = json.loads(r.read().decode())
        if not data.get("results"):
            return f"Location '{location}' not found."
        lat = data["results"][0]["latitude"]
        lon = data["results"][0]["longitude"]
        # Get current weather
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}&current=temperature_2m,weather_code,relative_humidity_2m"
        )
        with urllib.request.urlopen(url, timeout=10) as r:
            w = json.loads(r.read().decode())["current"]
        temp = w["temperature_2m"]
        code = w["weather_code"]
        humidity = w.get("relative_humidity_2m", "?")
        # Simple code -> description (WMO codes)
        desc = {
            0: "clear",
            1: "mainly clear",
            2: "partly cloudy",
            3: "overcast",
            45: "foggy",
            48: "depositing rime fog",
            51: "light drizzle",
            61: "rain",
            80: "rain showers",
            95: "thunderstorm",
        }.get(code, "variable")
        return f"{location}: {temp}°C, {desc}, humidity {humidity}%."
    except Exception as e:
        return f"Error getting weather for {location}: {e}"


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
    model="accounts/fireworks/models/qwen3-8b",
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
        model="accounts/fireworks/models/qwen3-8b",
        messages=messages,
        tools=tools,
    )
    final = response2.choices[0].message
    print(final.content)
else:
    print(msg.content or "(no content)")
