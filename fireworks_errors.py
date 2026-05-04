"""Error handling and retry patterns for Fireworks AI API.

Production applications must handle transient failures gracefully:
- 429 Rate limit exceeded (retry with exponential backoff)
- 503 Service unavailable (retry with longer backoff)
- 401 Authentication error (check API key, do not retry)
- 404 Invalid model (check model ID, do not retry)
- 400 Bad request (check request payload, do not retry)
"""
from dotenv import load_dotenv
load_dotenv()

import os
import time
import random
from fireworks import Fireworks, RateLimitError, ServiceUnavailableError

client = Fireworks()


def chat_with_retry(
    messages,
    model="accounts/fireworks/models/llama-v3p3-70b-instruct",
    max_retries=3,
    base_delay=1.0,
):
    """Send a chat completion with exponential backoff for 429/503 errors."""
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
            )
            return response
        except RateLimitError as e:
            if attempt == max_retries:
                raise
            # Exponential backoff with jitter
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            print(f"Rate limited (attempt {attempt + 1}). Retrying in {delay:.1f}s...")
            time.sleep(delay)
        except ServiceUnavailableError as e:
            if attempt == max_retries:
                raise
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            print(f"Service unavailable (attempt {attempt + 1}). Retrying in {delay:.1f}s...")
            time.sleep(delay)


# Example usage
if __name__ == "__main__":
    try:
        response = chat_with_retry(
            messages=[{"role": "user", "content": "Say hello in one word."}],
        )
        print("Success:", response.choices[0].message.content)
    except Exception as e:
        print(f"Failed after retries: {type(e).__name__}: {e}")

    # Demonstrate error types
    print("\n--- Error type demonstrations ---")

    # 401: Invalid API key
    try:
        bad_client = Fireworks(api_key="invalid-key")
        bad_client.chat.completions.create(
            model="accounts/fireworks/models/llama-v3p3-70b-instruct",
            messages=[{"role": "user", "content": "Hi"}],
        )
    except Exception as e:
        print(f"Invalid key: {type(e).__name__}: {e}")

    # 404: Invalid model
    try:
        client.chat.completions.create(
            model="accounts/fireworks/models/nonexistent-model-404",
            messages=[{"role": "user", "content": "Hi"}],
        )
    except Exception as e:
        print(f"Invalid model: {type(e).__name__}: {e}")
